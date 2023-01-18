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

output_path=/mnt/data_20t/flan_t5_distill/outputs/
batch_size_fixed=80
dataset=svamp_test
python test_distill.py\
    base_model=google/flan-t5-large\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    gpu_id=3

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

model_version=0.0.3.2
dataset=svamp_test
epoch=4
gpu_id=7
nohup python test_distill_multiple.py\
    model_version=${model_version}\
    test_data=${dataset}\
    tokenizer=t5-large\
    batch_size=150\
    iter=780m\
    epoch=${epoch}\
    gpu_id=${gpu_id}\
    &> logs/beta_${model_version}_${dataset}_e${epoch}_eval.log &
tail -f logs/beta_${model_version}_${dataset}_e${epoch}_eval.log
```