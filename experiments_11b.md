train 
```bash
model_version=0.1.0.0 # base model change to FlanT5 11B
loss_type=match_distribution
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1,2,3,4,5\'\
    base_model=\'google/flan-t5-xxl\'\
    batch_size=11b\
    grad_accum_steps=20\
    save_per_step=3000\
    loss_type=${loss_type}\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.1.2.0 # base model change to FlanT5 11B
loss_type=match_distribution
save_path=/mnt/data/flan_t5_distill/checkpoints/
grad_accum_steps=6
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1,2,3,4,5,6,7\'\
    base_model=\'google/flan-t5-xxl\'\
    grad_accum_steps=${grad_accum_steps}\
    save_per_step=2000\
    loss_type=${loss_type}\
    log_interval=2\
    lr=0.0001\
    device_map=11b_8cards.yaml\
    batch_size=11b_8cards.yaml\
    save_path=${save_path}\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.1.3.0 # base model change to FlanT5 11B
loss_type=match_distribution
save_path=/mnt/data_20t/flan_t5_distill/checkpoints/
grad_accum_steps=20
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1,2,3,4,5,6,7\'\
    base_model=\'google/flan-t5-xxl\'\
    grad_accum_steps=${grad_accum_steps}\
    save_per_step=2000\
    loss_type=${loss_type}\
    log_interval=2\
    lr=0.0002\
    device_map=11b_8cards.yaml\
    batch_size=11b_8cards.yaml\
    save_path=${save_path}\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log

model_version=0.2.0.0 # base model change to FlanT5 11B + deepspeed -- but did not work yet
loss_type=match_distribution
save_path=/mnt/data_20t/flan_t5_distill/checkpoints/
nohup python -u train_distill_lightning.py\
    model_version=${model_version}\
    gpu_id=\'0,1,2,3\'\
    base_model=\'google/flan-t5-xxl\'\
    batch_size=11b_deepspeed\
    grad_accum_steps=20\
    save_per_step=3000\
    loss_type=${loss_type}\
    log_interval=2\
    lr=0.0005\
    &> logs/beta_${model_version}.log &
tail -f logs/beta_${model_version}.log
```

test
```bash
output_path=/mnt/data_20t/flan_t5_distill/outputs/
base_model=/mnt/data_20t/flan_t5_distill/checkpoints/0.1.0.0_epoch_0_iter_63000
batch_size_fixed=40
dataset=gsm8k_test
gpu_id=\'0,1,2,3\'
python test_distill.py\
    base_model=${base_model}\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    model_size=11b\
    gpu_id=${gpu_id}

output_path=/mnt/data_20t/flan_t5_distill/outputs/
base_model=/mnt/data_20t/flan_t5_distill/checkpoints/0.1.0.0_epoch_0_iter_51000
batch_size_fixed=40
dataset=multiarith_test
gpu_id=\'4,5,6,7\'
python test_distill.py\
    base_model=${base_model}\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    model_size=11b\
    gpu_id=${gpu_id}

output_path=/mnt/data_10t/flan_t5_distill/outputs/
base_model=google/flan-t5-xxl
batch_size_fixed=40
dataset=asdiv_test
gpu_id=\'0,1,2,3\'
python test_distill.py\
    base_model=${base_model}\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    model_size=11b\
    gpu_id=${gpu_id}

output_path=/mnt/data_10t/flan_t5_distill/outputs/
base_model=google/flan-t5-xxl
batch_size_fixed=20
dataset=svamp_test
gpu_id=\'4,5,6\'
python test_distill.py\
    base_model=${base_model}\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    device_map=11b_3cards_inf\
    model_size=11b\
    gpu_id=${gpu_id}

output_path=/mnt/data_20t/flan_t5_distill/outputs/
base_model=google/flan-t5-xxl
batch_size_fixed=40
dataset=multiarith_test
gpu_id=\'4,5,6,7\'
python test_distill.py\
    base_model=${base_model}\
    output_path=${output_path}\
    batch_size_fixed=${batch_size_fixed}\
    test_data=${dataset}\
    model_size=11b\
    gpu_id=${gpu_id}
```