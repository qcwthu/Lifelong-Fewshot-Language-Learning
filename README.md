# [ICLR2022]  LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5 

The repo is the source code for [LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5](https://openreview.net/forum?id=HCRVf71PMF)

Chengwei Qin, Shafiq Joty

Accepted at 10th  International Conference on Learning Representations (ICLR'22).


## Setup

### 1. Download the code

```
git clone git@github.com:qcwthu/Lifelong-Fewshot-Language-Learning.git
cd Lifelong-Fewshot-Language-Learning
```

### 2. Install dependencies

#### 2.1. Create conda environment

For NER & Classification:

```
conda create --name lfll_1 python=3.9.4
conda activate lfll_1
pip install gpustat
pip install seqeval==1.2.2
pip install numpy==1.19.5
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
cd transformers; pip install .; cd ..
pip install tensorflow==2.5.0
pip install fairscale==0.3.7
pip install datasets==1.11.0
pip install sentencepiece==0.1.95
pip install matplotlib==3.3.4
```
For Summarization & Different task types:

```
conda create --name lfll_2 python=3.9.4
conda activate lfll_2
pip install gpustat
pip install seqeval==1.2.2
```

#### 2.2. Download LM-adapted T5 model

To run lifelong few-shot language learning with prompt tuning, you should download the LM-adapted T5 model from [LM-Adapted: t5.1.1.lm100k](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#lm-adapted-t511lm100k). Then you should use `convertmodel.py` to convert Tensorflow checkpoint to Pytorch checkpoint.

```
mkdir lm_adapted_t5model
cd lm_adapted_t5model
```
Then you should follow [gsutil_install](https://cloud.google.com/storage/docs/gsutil_install) to install gsutil. 

After installing gsutil, run 

```
gsutil -m cp \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/checkpoint" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model-info.txt" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.data-00000-of-00004" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.data-00001-of-00004" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.data-00002-of-00004" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.data-00003-of-00004" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.index" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/model.ckpt-1100000.meta" \
  "gs://t5-data/pretrained_models/t5.1.1.lm100k.large/operative_config.gin" \
  .
```
to download LM-adapted T5-large checkpoint [T5-large TF ckpt](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5.1.1.lm100k.large;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). Finally,

```
cd ..
mkdir lm_adapted_t5model/torch_ckpt
python convertmodel.py
```

### 3. Run code

**Remember to change `lm_adapted_path` and `cache_path` in script files to the right path on your server!!!**

```
lm_adapted_path: the path of lm_adapted t5 model (Pytorch ckpt)
cache_path: the path of huggingface cache
```

#### 3.1. NER:
```
cd NER
bash runall_1.sh
```

#### 3.2. Classification
```
cd Classification
bash Classification.sh
```

#### 3.3. Summarization
```
cd Summarization
bash Summarization.sh
```

#### 3.4. Different task types
Without forward knowledge transfer (FKD):

```
cd DiffType/T5NoContinual
bash T5NoContinual.sh
```

With forward knowledge transfer (FKD):

```
cd DiffType/T5WithContinual
bash T5WithContinual.sh
```




## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@inproceedings{
qin2022lfpt,
title={{LFPT}5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5},
author={Chengwei Qin and Shafiq Joty},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=HCRVf71PMF}
}

```