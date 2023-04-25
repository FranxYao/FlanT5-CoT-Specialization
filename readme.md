# Language Model Emergent Ability Distillation. 

Implementation of Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot. _Specializing Smaller Language Models towards Multi-Step Reasoning_. ICML 2023

Code preview. Data and model checkpoints coming soon. 

TODO:
* [ ] Add preprocessed data
* [ ] Add DeepSpeed integration 
* [ ] Add requirements.txt -- but generally this repo only requires transformers and pytorch

Quickstart:
```bash
model_version=0.0.5.0 # base model change to FlanT5 780m
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


Notebooks 
* `flan_t5_gsm8k.ipynb`: run GSM8K on Flan-T5
* `flan_t5_gsm8k_verifier.ipynb`: run GSM8K on Flan-T5 with verifier
* `flan_t5_gsm8k_vis.ipynb`: visualization of decoding probability of Flan-T5
* `flan_t5_mmlu.ipynb`: run mmlu on Flan-T5
* `flan_t5_multiarith.ipynb`: run multiarith on Flan-T5
* `opt_gsm8k.ipynb`: run multiarith on OPT (have not gone through this yet)

Scripts
* `codex_decode_gsm8k.py`: decode gsm8k training set with codex
* `flan_t5_decode_gsm8k.py`: decode gsm8k training set with Flan-T5
* `flan_t5_verifier_decode_gsm8k.py`: decode gsm8k training set with Flan-T5 + verifier (TBC)

Distillation 
* `train_distill_t5.py`: train the distillation algorithm 
* `trainer_distill.py`: trainer for emergent ability distillation 

