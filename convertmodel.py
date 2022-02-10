import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, load_tf_weights_in_t5
from transformers.utils import logging
import tensorflow as tf
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path, model_name):
    config = T5Config.from_pretrained(model_name)
    print(f"Building PyTorch model from configuration: {config}")
    model = T5ForConditionalGeneration(config)
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

if __name__ == "__main__":
    # savepath_prefix = ["/data/qin/lm_adapted_t5model/torch_ckpt/small","/data/qin/lm_adapted_t5model/torch_ckpt/base",
    #                    "/data/qin/lm_adapted_t5model/torch_ckpt/large","/data/qin/lm_adapted_t5model/torch_ckpt/xl",
    #                    "/data/qin/lm_adapted_t5model/torch_ckpt/xxl"]
    savepath_prefix = ["./lm_adapted_t5model/torch_ckpt/large"]
    for path in savepath_prefix:
        if not os.path.exists(path):
            os.mkdir(path)
    #modeltype = ["google/t5-v1_1-small", "google/t5-v1_1-base", "google/t5-v1_1-large", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]
    modeltype = ["google/t5-v1_1-large"]
    loadpath_prefix = "./lm_adapted_t5model/"
    # ckptpath = [loadpath_prefix+"t5.1.1.lm100k.small/",loadpath_prefix+"t5.1.1.lm100k.base/",loadpath_prefix+"t5.1.1.lm100k.large/",
    #             loadpath_prefix+"t5.1.1.lm100k.xl/",loadpath_prefix+"t5.1.1.lm100k.xxl/"]
    ckptpath = [loadpath_prefix + "t5.1.1.lm100k.large/"]
    for i in range(len(modeltype)):
        print(i)
        tf_vars = tf.train.list_variables(ckptpath[i])
        convert_tf_checkpoint_to_pytorch(ckptpath[i], savepath_prefix[i], modeltype[i])
