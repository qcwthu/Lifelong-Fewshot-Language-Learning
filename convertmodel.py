import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import sys
import argparse
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, load_tf_weights_in_t5
from transformers.utils import logging
from pprint import pprint
import tensorflow as tf
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path, model_name):
    # Initialise PyTorch model
    config = T5Config.from_pretrained(model_name)
    print(f"Building PyTorch model from configuration: {config}")
    model = T5ForConditionalGeneration(config)
    # n_params = sum(p.numel() for p in model.parameters())
    # print("Total # of parameters: {}".format(n_params))
    # #ckpttorecover = torch.load(pytorch_dump_path + "/pytorch_model.bin")
    # ckpttorecover = torch.load("/data/qin/T5/t5_ckpt_1/t5_ckpt/ckpt_of_step_100000")
    # model.load_state_dict(ckpttorecover['t5-base-prefixlm'])
    # n_params = sum(p.numel() for p in model.parameters())
    # print("Total # of parameters: {}".format(n_params))

    # Load weights from tf checkpoint
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # ckpt = {
    #     't5-model': model.state_dict(),
    # }
    # torch.save(ckpt, os.path.join(pytorch_dump_path, "pytorch_model.bin"))

if __name__ == "__main__":
    savepath_prefix = ["/data/qin/lm_adapted_t5model/torch_ckpt/small","/data/qin/lm_adapted_t5model/torch_ckpt/base",
                       "/data/qin/lm_adapted_t5model/torch_ckpt/large","/data/qin/lm_adapted_t5model/torch_ckpt/xl",
                       "/data/qin/lm_adapted_t5model/torch_ckpt/xxl"]
    for path in savepath_prefix:
        if not os.path.exists(path):
            os.mkdir(path)
    modeltype = ["google/t5-v1_1-small", "google/t5-v1_1-base", "google/t5-v1_1-large", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]
    loadpath_prefix = "/data/qin/lm_adapted_t5model/"
    ckptpath = [loadpath_prefix+"t5.1.1.lm100k.small/",loadpath_prefix+"t5.1.1.lm100k.base/",loadpath_prefix+"t5.1.1.lm100k.large/",
                loadpath_prefix+"t5.1.1.lm100k.xl/",loadpath_prefix+"t5.1.1.lm100k.xxl/"]
    # tf_path = os.path.abspath('/data/qin/lm_adapted_t5model/t5.1.1.lm100k.small/model.ckpt-*.data-*')  # Path to our TensorFlow checkpoint
    for i in range(len(modeltype)):
        print(i)
        tf_vars = tf.train.list_variables(ckptpath[i])
        #pprint(tf_vars)
        #config = T5Config.from_pretrained(modeltype[i])
        #print(f"Building PyTorch model from configuration: {config}")
        #model = T5ForConditionalGeneration(config)
        #for k,v in model.named_parameters():
        #    print(k)
        convert_tf_checkpoint_to_pytorch(ckptpath[i], savepath_prefix[i], modeltype[i])
