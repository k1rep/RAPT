import os
import torch
import datetime
import logging
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup
from common.data_structures import Examples
from common.utils import format_batch_input, write_tensor_board, save_check_point, evaluate_classification, \
    evaluate_retrival, load_check_point, format_triplet_batch_input

logger = logging.getLogger(__name__)


def log_train_info(args, example_num, train_steps):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", example_num)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", train_steps)


def get_exp_name(args):
    exp_name = "{}_{}"
    time = datetime.datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exp_name.format(time, base_model)


def train_with_neg_sampling_triplet(args, model, train_examples: Examples, valid_examples: Examples, optimizer,
                            scheduler, tb_writer, step_bar, skip_n_steps):
    """
    Create training dataset at epoch level.
    """

    tr_loss, tr_ac = 0, 0
    batch_size = args.per_gpu_train_batch_size
    train_dataloader = train_examples.online_neg_sampling_dataloader(batch_size=int(batch_size / 2))  # (n_id,p_id,1)

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        if skip_n_steps > 0:
            skip_n_steps -= 1
            continue
        batch = train_examples.make_online_triplet_sampling_batch(batch, model)
        if batch == ():
            continue
        model.train()
        inputs = format_triplet_batch_input(batch, train_examples, model)
        outputs = model(**inputs)
        loss = outputs['loss']

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            try:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        else:
            loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            args.global_step += 1
            step_bar.update()

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                tb_data = {
                    'lr': scheduler.get_last_lr()[0],
                    'acc': tr_ac / args.logging_steps / (
                            args.train_batch_size * args.gradient_accumulation_steps),
                    'loss': tr_loss / args.logging_steps
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
                tr_loss = 0.0
                tr_ac = 0.0

            if args.valid_step > 0 and args.global_step % args.valid_step == 1:
                valid_accuracy = evaluate_classification(valid_examples, model,
                                                                     args.per_gpu_eval_batch_size,
                                                                     "result/train/{}".format(
                                                                         args.data_name))
                pk, best_f1, map = evaluate_retrival(model, valid_examples, args.per_gpu_eval_batch_size,
                                                     "result/train/{}".format(args.data_name))
                tb_data = {
                    "valid_accuracy": valid_accuracy,
                    "precision@3": pk,
                    "best_f1": best_f1,
                    "MAP": map
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
        args.steps_trained_in_current_epoch += 1


def get_optimizer_scheduler(args, model, train_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=train_steps
    )
    return optimizer, scheduler


def train(args, train_examples, valid_examples, model, train_iter_method):
    """

    :param args:
    :param train_examples:
    :param valid_examples:
    :param model:
    :param train_iter_method: method use for training in each iteration
    :return:
    """
    model_output = os.path.join('./output', args.data_name)
    if not args.exp_name:
        exp_name = get_exp_name(args)
    else:
        exp_name = args.exp_name

    #    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="../runs/{}".format(exp_name))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    example_num = 2 * len(train_examples)  # 50/50 of pos/neg
    epoch_batch_num = example_num / args.train_batch_size  # 每一个epoch有多少batch
    t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs  # 总共多少次参数更新
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    log_train_info(args, example_num, t_total)

    args.global_step = 0
    args.epochs_trained = 0
    args.steps_trained_in_current_epoch = 0
    if args.model_path and os.path.exists(args.model_path):
        ckpt = load_check_point(model, args.model_path)
        model = ckpt["model"]
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch {}, global step {}".format(args.epochs_trained, args.global_step))
    else:
        logger.info("Start a new training")
    skip_n_steps_in_epoch = args.steps_trained_in_current_epoch  # in case we resume training
    model.zero_grad()
    train_iterator = trange(args.epochs_trained, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    step_bar = tqdm(initial=args.epochs_trained, total=t_total, desc="Steps")
    for _ in train_iterator:
        params = (
            args, model, train_examples, valid_examples, optimizer, scheduler, tb_writer, step_bar,
            skip_n_steps_in_epoch)

        train_iter_method(*params)
        args.epochs_trained += 1
        skip_n_steps_in_epoch = 0
        args.steps_trained_in_current_epoch = 0

    save_check_point(model, model_output, args, optimizer, scheduler)
    if args.local_rank in [-1, 0]:
        tb_writer.close()
