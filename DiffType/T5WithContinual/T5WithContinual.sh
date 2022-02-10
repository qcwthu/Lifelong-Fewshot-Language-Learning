learnrate=(5e-1)
alllambda=(0.10)
allstartindex=(0 1 2)
alltask=(1 2)
for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
    for onestartindex in ${allstartindex[@]}
    do
      for onetask in ${alltask[@]}
      do
              echo "------------------------------"
              python -m torch.distributed.launch --nproc_per_node 2 --master_port 29953 T5WithContinual.py \
                --cuda 5,6 \
                --lr $onerate \
                --lm_lambda $onelambda \
                --startindex $onestartindex \
                --taskindex $onetask \
                --weight_decay 1e-5 \
                --max_grad_norm 1.0 \
                --batch_size_per_gpu 2 \
                --valid_size_per_gpu 16 \
                --test_size_per_gpu 16 \
                --gradient_accumulation_steps 4 \
                --max_epoch 384 \
                --num_workers 0 \
                --save_step 100000 \
                --eval_step 100000 \
                --seed 42 \
                --model_name google/t5-v1_1-large \
                --train_sample \
                --max_length 128 \
                --adam_epsilon 1e-8 \
                --warmup_steps 0.01 \
                --use_lm_adapted 1 \
                --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
                --cache_path /data/qin/cache/ \
                --prompt_number 300 \
                --ifckpt_onlymodel 1
          echo "++++++++++++++++++++++++++++++"
          ps aux | grep T5WithContinual.py | awk '{print $2}' | xargs kill -9
      done
    done
  done
done



