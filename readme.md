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
model_version=0.0.2.2.1 ## BUGGY THIS ONE
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

model_version=0.0.2.2.2 # match distribution + same question in batch
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.4.1 # contrastive loss, not good 
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

model_version=0.0.2.3 # match sample + same question in batch
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'2,3\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    loss_type=match_distribution\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.7 # match sample
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'6,7\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    batch_mix_mode=fully_random\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.8 # match distribution
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'4,5\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    loss_type=match_distribution\
    batch_mix_mode=fully_random\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.6 # only use in-context training instances
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    data_formats=in_context\
    loss_type=match_sample\
    batch_mix_mode=fully_random\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.2.9 # only use zero-shot training instances
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'2,3\'\
    base_model=\'google/flan-t5-xl\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    data_formats=zero_shot\
    loss_type=match_sample\
    batch_mix_mode=fully_random\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.3.0 # base model change to T5 3b
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1\'\
    base_model=\'t5-3b\'\
    batch_sizes=3b\
    device_map=3b\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.3.1 # base model change to T5 3b, loss type change to match distribution
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'6,7\'\
    base_model=\'t5-3b\'\
    batch_sizes=3b\
    device_map=3b\
    loss_type=match_distribution\
    grad_accum_steps=30\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.3.2 # base model change to T5 780m
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'2\'\
    base_model=\'t5-large\'\
    batch_sizes=780m\
    grad_accum_steps=20\
    save_per_step=3000\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.0.4.0 # base model change to FlanT5 780m
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'3\'\
    base_model=\'google/flan-t5-large\'\
    batch_sizes=780m\
    grad_accum_steps=20\
    save_per_step=3000\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log
```

Test
```bash
python test_distill.py\
    base_model=/mnt/data_10t/flan_t5_distill/checkpoints/0.0.2.8_epoch_0_iter_60000\
    test_data=gsm8k_test\
    batch_size=80\
    gpu_id=0

python test_distill.py\
    base_model=/mnt/data_10t/flan_t5_distill/checkpoints/0.0.2.8_epoch_0_iter_60000\
    test_data=multiarith_test\
    batch_size=80\
    gpu_id=0

python test_distill_multiple.py\
    model_version=0.0.2.6\
    test_data=multiarith_test\
    epoch=0\
    gpu_id=5

python test_distill_multiple.py\
    model_version=0.0.2.9\
    test_data=asdiv_test\
    epoch=5\
    gpu_id=4

python test_distill_multiple.py\
    model_version=0.0.2.6\
    test_data=asdiv_test\
    prompt_mode=zero_shot_cot\
    epoch=1\
    gpu_id=4

python test_distill_multiple.py\
    model_version=0.0.3.0\
    batch_size=80\
    test_data=gsm8k_test\
    tokenizer=t5-3b\
    epoch=0\
    gpu_id=0
```