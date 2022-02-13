learnrate=(5e-1)
alllambda=(0.10)
allkd=(0.06 0.03 0.05)

for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
      for onekd in ${allkd[@]}
      do
          echo "------------------------------"
          python -m torch.distributed.launch --nproc_per_node 2 --master_port 29528 NER_Onto2Conll_1.py \
            --cuda 4,5 \
            --lr $onerate \
            --lm_lambda $onelambda \
            --kd_lamda $onekd \
            --weight_decay 1e-5 \
            --max_grad_norm 1.0 \
            --batch_size_per_gpu 2 \
            --valid_size_per_gpu 16 \
            --test_size_per_gpu 16 \
            --gradient_accumulation_steps 4 \
            --max_epoch 640 \
            --num_workers 4 \
            --save_step 100000 \
            --eval_step 100000 \
            --save_dir t5ner_pseudo_ckpt \
            --seed 42 \
            --model T5NER \
            --model_name google/t5-v1_1-large \
            --train_file_name ./conll_fewshot/train.txt \
            --valid_file_name ./conll_fewshot/valid.txt \
            --test_file_name ./conll_fewshot/test.txt \
            --train_sample \
            --max_length 128 \
            --adam_epsilon 1e-8 \
            --warmup_steps 0.01 \
            --load_ckpt 0 \
            --use_lm_adapted 1 \
            --lm_adapted_path  /data/qin/lm_adapted_t5model/torch_ckpt/large/pytorch_model.bin \
            --cache_path /data/qin/cache/ \
            --prompt_number 300 \
            --ifckpt_onlymodel 1 \
            --use_pre_prompt 1 \
            --pre_prompt_path  ./onto_ckpt/onto_ckpt
          echo "++++++++++++++++++++++++++++++"
          ps aux | grep NER_Onto2Conll_1.py | awk '{print $2}' | xargs kill -9
    done
  done
done



