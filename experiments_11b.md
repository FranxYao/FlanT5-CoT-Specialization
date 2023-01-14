train 
```bash
model_version=0.1.0.0 # base model change to FlanT5 780m
loss_type=match_distribution
nohup python -u train_distill_simple.py\
    model_version=${model_version}\
    gpu_id=\'0,1,2,3,4,5\'\
    base_model=\'google/flan-t5-xxl\'\
    batch_sizes=11b\
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

```