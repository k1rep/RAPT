import logging
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import BertConfig

sys.path.append("..")
sys.path.append("../../")

from common.init_eval import get_eval_args, test

from common.data_loader import load_examples
# from common.utils import MODEL_FNAME
MODEL_FNAME = 't_bert.pt'
from common.models import TBertT
if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if not os.path.exists("./cache"):
        os.makedirs("./cache")
    cached_file = os.path.join("./cache", "test_{}.dat".format(args.data_name))
    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model = TBertT(BertConfig(), args)
    # lora
    # if args.model_path and os.path.exists(args.model_path):
    #     model_path = os.path.join(args.model_path, MODEL_FNAME)
    #     model.load_state_dict(torch.load(model_path), strict=False)
    #     peft_model_id = r'./output/' + args.data_name + '/lora'
    #     model.cbert = PeftModel.from_pretrained(model.cbert, peft_model_id)
    # bert
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        raise Exception("evaluation model not found")
    logger.info("model loaded")
    print(model)
    start_time = time.time()
    test_dir = os.path.join(args.data_dir, "test")
    test_examples = load_examples(test_dir, model=model, num_limit=args.test_num, type='test')
    test_examples.update_embd(model)
    m = test(args, model, test_examples)
    exe_time = time.time() - start_time
    m.write_summary(exe_time)


