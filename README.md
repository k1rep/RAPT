# RAPT-replication-package

This repository contains source code that we used to perform experiment in paper titled "xxxx".

Please follow the steps below to reproduce the result.

## Environment Setup

- Python >= 3.7 
- pytorch/1.1.0
- 1 GPU with CUDA 10.2 or 11.1

Run the following command in terminal (or command line) to install python packages.

```
pip install -U pip setuptools 
pip install -r requirement.txt
```

## Experiment Result Replication Guide

### Train

```bash
cd trace/main
python train_trace_rapt.py \
    --data_dir ../data/Teiid \
    --output_dir ./output \
    --model_path ./model \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --logging_steps 50 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 5 \
    --learning_rate 4e-5 \
    --valid_step 1000
```

### Evaluate

```bash
python eval_trace_rapt.py \
        --data_dir ../data/Teiid \
        --model_path ./output/Teiid \
        --per_gpu_eval_batch_size 4 
```

## Dataset

The data we used to train and test is attached in the `trace/data` directory. You can replace the data with your own. The easiest way to do this is formatting the data into the following csv schema. After formatting the data, you can use the train/eval scripts in to conduct training and evaluatoin.

-----------------------

**commit_file:**

commit_id: unique id of the code artifact

diff: the actual content of the code file in string, in our case is the code change set

summary: not used

commit_time: not used

files: not used

----

**issue_file:**

issue_id: unique id of the NL artifact

issue_desc: not used

issue_comments: string of the content

created_at: not used

closed_at: not used

---

**link_file:**

issue_id: ids from issue_file

commit_id: ids from commit_file

----
