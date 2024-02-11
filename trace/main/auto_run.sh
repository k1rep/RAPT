#!/bin/bash
set -e   # 错误监测
cd /nvme3n1/LiYworks/RAPT/trace/main
echo "auto run start !" | tee /nvme3n1/LiYworks/RAPT/log.txt
export CUDA_VISIBLE_DEVICES=0

for PROJ_NAME in Infinispan JBossTransactionManager JGroups ModeShape Quarkus RESTEasy Teiid Undertow Weld WildFlyElytron
# Debezium HAL Infinispan JBossTransactionManager JGroups ModeShape Quarkus RESTEasy Teiid Undertow Weld Wildfly WildFlyCore WildFlyElytron
do
    echo -e "\n----------- processing proj $PROJ_NAME -----------" | tee -a /nvme3n1/LiYworks/RAPT/log.txt
    
    # 执行训练
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] Train start..." | tee -a /nvme3n1/LiYworks/RAPT/log.txt
    python train_trace_rapt.py \
    --data_dir /nvme3n1/LiYworks/RAPT/trace/data_no/$PROJ_NAME \
    --output_dir ./output \
    --model_path /nvme3n1/LiYworks/RAPT/trace/main/model \
    --per_gpu_train_batch_size 8\
    --per_gpu_eval_batch_size 8\
    --logging_steps 50 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 4e-5 \
    --valid_step 1000

    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] Train finished..." | tee -a /nvme3n1/LiYworks/RAPT/log.txt
    
    # 执行测试
    python eval_trace_rapt.py \
        --data_dir /nvme3n1/LiYworks/RAPT/trace/data_no/$PROJ_NAME \
        --model_path ./output/$PROJ_NAME \
        --per_gpu_eval_batch_size 4 

    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] Test finished..." | tee -a /nvme3n1/LiYworks/RAPT/log.txt
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] $PROJ_NAME successfully finished..." | tee -a /nvme3n1/LiYworks/RAPT/log.txt

    # 删除遗留文件
    rm -rf /nvme3n1/LiYworks/RAPT/trace/main/output/$PROJ_NAME
done

cd /nvme3n1/LiYworks/RAPT/trace/main