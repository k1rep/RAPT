import logging
import os
import sys

sys.path.append("..")
sys.path.append("../..")

from common.init_train import get_train_args, init_train_env
from common.train_utils import train_with_neg_sampling_triplet, train, logger
from common.data_loader import load_examples

logger = logging.getLogger(__name__)


def main():
    args = get_train_args()
    print(args.data_name)
    model = init_train_env(args)
    train_dir = os.path.join(args.data_dir, "train")
    # valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(train_dir, model=model, num_limit=args.train_num)
    # valid_examples = load_examples(valid_dir, model=model, num_limit=args.valid_num)
    train(args, train_examples, model, train_with_neg_sampling_triplet)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
