# Distilling Chain-of-Thought Reasoning from code-davinci-002 to FlanT5

Implementation of Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot. _Specializing Smaller Language Models towards Multi-Step Reasoning_. ICML 2023. [[Arxiv](https://arxiv.org/abs/2301.12726)]

Download data at [Google Drive](https://drive.google.com/drive/folders/1BOXcUTnEyvQia_ypHcaUnUbLsN4HzqmQ?usp=sharing)

After downloading the data, put it under `processed_data/` folder because all data are processed and stored as `.pkl` files. 

A lot of the engineering efforts in this work is not modeling, but data engineering, mostly about processing the data into the four following formats that is important for imbuing the model with in-context and zero-shot abilities. See figure 1B in the paper for details.
* in-context answer-only
* in-context chain-of-thought
* zero-shot answer-only
* zero-shot chain-of-thought

We strongly recommend runing `notebooks/inspect_processed_data.ipynb` to get a sense at what the data looks like. It gives an example about how `in-context chain-of-thought` data looks like.

The actual training script is pretty simple `train_distill_simple.py`. Most of the efforts go to data engineering, hyperparameter search, and evaluation.

The following is a quickstart code using FlanT5 base model. We did not have time to implement DeepSpeed/ FairScale/ Pytorch FSDP because we were in a rush when developing this work. Yet wrapping the model with DeepSpeed should be pretty straightforward. If you have done this, please submit a pull request and we will be happy to merge it :)

Quickstart:
```bash
pip install -r requirements.txt

# inspect data 
# see notebooks/inspect_processed_data.ipynb

# run a small model 
model_version=0.0.5.0 # base model FlanT5 780m
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0\'\
    base_model=\'google/flan-t5-base\'\
    batch_size=250m\
    grad_accum_steps=3\
    save_per_step=1000\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log
```

Notebooks for inspecting the processed data
* `inspect_processed_data.ipynb`: an example about how `in-context chain-of-thought` data looks like.

Notebooks for visualization
* `dev_align_codex_to_flan_t5_dtw.ipynb`: notebook for aligning codex and flan t5 tokenized outputs using dynamic time warping
* `dev_process_codex_outputs.ipynb`: visualize the output probability of codex
* `dev_process_flan_t5_outputs.ipynb`: visualize the output probability of codex

Notebooks for prompting FlanT5
* `flan_t5_3b_asdiv.ipynb`: prompting FlanT5 3B on ASDIV dataset 
* `flan_t5_3b_gsm8k.ipynb`: prompting FlanT5 3B on GSM8K dataset
* `flan_t5_3b_multiarith.ipynb`: prompting FlanT5 3B on MultiArith dataset
* `flan_t5_3b_svamp.ipynb`: prompting FlanT5 3B on SVAMP dataset
* `flan_t5_11b_GSM8K.ipynb`: prompting FlanT5 11B on GSM8K dataset

Scripts
* `codex_decode_gsm8k.py`: decode gsm8k training set with codex
* `flan_t5_decode_gsm8k.py`: decode gsm8k training set with Flan-T5
* `flan_t5_verifier_decode_gsm8k.py`: decode gsm8k training set with Flan-T5 + verifier (TBC)

Distillation 
* `train_distill_t5.py`: train the distillation algorithm 
* `trainer_distill.py`: trainer for emergent ability distillation 

TODO:
* [x] Add preprocessed data
* [ ] Add DeepSpeed integration 
* [x] Add requirements.txt -- but generally this repo only requires transformers and pytorch
* [ ] Code Cleaning
* [ ] Example dynamic programming code for matching different tokenizers
