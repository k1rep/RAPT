import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from common.metrices import metrics

MODEL_FNAME = "t_bert.pt"
OPTIMIZER_FNAME = "optimizer.pt"
SCHED_FNAME = "scheduler.pt"
ARG_FNAME = "training_args.bin"

logger = logging.getLogger(__name__)
map_file={}
map_iss={}

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

def format_batch_input_for_single_bert(batch, examples, model):
    tokenizer = model.tokenizer
    nl_ids, pl_ids, labels = batch[0].tolist(), batch[1].tolist(), batch[2].tolist()
    input_ids = []
    att_masks = []
    tk_types = []
    for nid, pid, lb in zip(nl_ids, pl_ids, labels):
        encode = examples._gen_seq_pair_feature(nid, pid, tokenizer)
        input_ids.append(torch.tensor(encode["input_ids"], dtype=torch.long))
        att_masks.append(torch.tensor(encode["attention_mask"], dtype=torch.long))
        tk_types.append(torch.tensor(encode["token_type_ids"], dtype=torch.long))
    input_tensor = torch.stack(input_ids)
    att_tensor = torch.stack(att_masks)
    tk_type_tensor = torch.stack(tk_types)
    features = [input_tensor, att_tensor, tk_type_tensor]
    features = [t.to(model.device) for t in features]
    inputs = {
        'input_ids': features[0],
        'attention_mask': features[1],
        'token_type_ids': features[2],
    }
    return inputs


def format_batch_input(batch, examples, model):
    """
    move tensors to device
    :param batch:
    :param examples:
    :return:
    """
    nl_ids, pl_ids, labels = batch[0], batch[1], batch[2]
    features = examples.id_pair_to_feature_pair(nl_ids, pl_ids)
    features = [t.to(model.device) for t in features]
    nl_in, nl_att, pl_in, pl_att = features
    inputs = {
        "text_ids": nl_in,
        "text_attention_mask": nl_att,
        "code_ids": pl_in,
        "code_attention_mask": pl_att,
    }
    return inputs



def format_triplet_batch_input(batch, examples, model):
    """
    move tensors to device
    :param batch:
    :param examples:
    :param model:
    :return:
    """
    anchor_ids, pos_cid, neg_cid = batch[0], batch[1], batch[2]
    features = examples.id_triplet_to_feature_triplet(nl_id_tensor=anchor_ids, pos_pl_id_tensor=pos_cid, neg_pl_id_tensor=neg_cid)
    features = [t.to(model.device) for t in features]
    nl_in, nl_att, pos_pl_in, pos_pl_att, neg_pl_in, neg_pl_att = features
    inputs = {
        "text_ids": nl_in,
        "text_attention_mask": nl_att,
        "pos_code_ids": pos_pl_in,
        "pos_code_attention_mask": pos_pl_att,
        "neg_code_ids": neg_pl_in,
        "neg_code_attention_mask": neg_pl_att,
    }
    return inputs



def write_tensor_board(tb_writer, data, step):
    for att_name in data.keys():
        att_value = data[att_name]
        tb_writer.add_scalar(att_name, att_value, step)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def save_examples(exampls, output_file):
    nl = []
    pl = []
    df = pd.DataFrame()
    for exmp in exampls:
        nl.append(exmp['NL'])
        pl.append(exmp['PL'])
    df['NL'] = nl
    df['PL'] = pl
    df.to_csv(output_file)


def save_check_point(model, ckpt_dir, args, optimizer, scheduler):
    logger.info("Saving checkpoint to %s", ckpt_dir)
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    peft_model_id = r'./output/'+args.data_name+'/lora'
    # torch.save(model.state_dict(), os.path.join(ckpt_dir, MODEL_FNAME))
    # torch.save(args, os.path.join(ckpt_dir, ARG_FNAME))
    # torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, OPTIMIZER_FNAME))
    # torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, SCHED_FNAME))
    if not os.path.exists(peft_model_id):
        os.makedirs(peft_model_id)
    model.cbert.save_pretrained(peft_model_id)


def load_check_point(model, ckpt_dir):
    logger.info(
        "Loading checkpoint from {}".format(ckpt_dir))
    model_path = os.path.join(ckpt_dir, MODEL_FNAME)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.cbert = get_peft_model(model.cbert, peft_config)


    return {'model': model}


def results_to_df(res: List[Tuple]) -> DataFrame:
    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    return df


def evaluate_classification(eval_examples, model, batch_size, output_dir):
    """

    :param eval_examples:
    :param model:
    :param batch_size:
    :param output_dir:
    :param append_label: append label to calculate evaluation_loss
    :return:
    """
    eval_dataloader = eval_examples.random_neg_sampling_dataloader(batch_size=batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    # Eval!

    clsfy_res = []
    num_correct = 0
    eval_num = 0

    for batch in tqdm(eval_dataloader, desc="Classify Evaluating"):
        with torch.no_grad():
            model.eval()
            labels = batch[2].to(model.device)
            inputs = format_batch_input(batch, eval_examples, model)
            text_hidden = model.create_nl_embd(inputs['text_ids'], inputs['text_attention_mask'])[0]
            code_hidden = model.create_pl_embd(inputs['code_ids'], inputs['code_attention_mask'])[0]
            y_pred = model.get_sim_score(text_hidden, code_hidden) > 0.5
            batch_correct = y_pred.eq(labels).long().sum().item()
            num_correct += batch_correct
            eval_num += y_pred.size()[0]
            clsfy_res.append((y_pred, labels, batch_correct))

    accuracy = num_correct / eval_num
    tqdm.write("\nevaluate accuracy={}\n".format(accuracy))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    res_file = os.path.join(output_dir, "raw_classify_res.txt")
    with open(res_file, 'w') as fout:
        for res in clsfy_res:
            fout.write(
                "pred:{}, label:{}, num_correct:{}\n".format(str(res[0].tolist()), str(res[1].tolist()), str(res[2])))
    return accuracy


def evaluate_retrival(model, eval_examples, batch_size, res_dir):
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    retr_res_path = os.path.join(res_dir, "raw_result.csv")
    summary_path = os.path.join(res_dir, "summary.txt")
    retrival_dataloader = eval_examples.get_retrivial_task_dataloader(batch_size)
    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        with torch.no_grad():
            model.eval()
            nl_ids = batch[0]
            pl_ids = batch[1]
            labels = batch[2].to(model.device)
            inputs = format_batch_input(batch, eval_examples, model)
            text_hidden = model.create_nl_embd(inputs['text_ids'], inputs['text_attention_mask'])[0]
            code_hidden = model.create_pl_embd(inputs['code_ids'], inputs['code_attention_mask'])[0]
            sim_score = model.get_sim_score(text_hidden=text_hidden, code_hidden=code_hidden).cpu()
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), sim_score.tolist(), labels.tolist()):
                res.append((n, p, prd, lb))

    df = results_to_df(res)
    df.to_csv(retr_res_path)
    m = metrics(df, output_dir=res_dir)

    pk = m.precision_at_K(3)
    best_f1, best_f2, details, _ = m.precision_recall_curve("pr_curve.png")
    map = m.MAP_at_K(3)

    summary = "\nprecision@3={}, best_f1 = {}, best_f2={}， MAP={}\n".format(pk, best_f1, best_f2, map)
    with open(summary_path, 'w') as fout:
        fout.write(summary)
        fout.write(str(details))
    return pk, best_f1, map
