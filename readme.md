# Language Model Emergent Ability Distillation. 

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

Train
```bash 
model_version=0.0.2.2.1
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'2,3\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=100\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.4.1
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'6,7\'\
    base_model=\'google/flan-t5-xl\'\
    data_formats=contrastive\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=10\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log
```

Test
```bash
python test_distill.py\
    base_model=/mnt/data_10t/flan_t5_distill/checkpoints/0.0.2.1_epoch_1_end\
    test_data=gsm8k_dev\
    batch_size=80\
    gpu_id=7
```