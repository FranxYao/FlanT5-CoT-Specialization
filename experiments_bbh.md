# Testing model performance on BBH

```bash
gpu_id=0
base_model=google/flan-t5-xl
nohup python test_bbh.py\
    base_model=${base_model}\
    tokenizer=${base_model}\
    gpu_id=${gpu_id}\
    &> logs/beta_${base_model:7}_bbh_eval.log &
tail -f logs/beta_${base_model:7}_bbh_eval.log

gpu_id=6
base_model=google/flan-t5-xl
prompt_mode=ao
nohup python test_bbh.py\
    base_model=${base_model}\
    tokenizer=${base_model}\
    prompt_mode=${prompt_mode}\
    gpu_id=${gpu_id}\
    &> logs/beta_${base_model:7}_bbh_${prompt_mode}_eval.log &
tail -f logs/beta_${base_model:7}_bbh_${prompt_mode}_eval.log

gpu_id=\'4,5,6,7\'
base_model=google/flan-t5-xxl
prompt_mode=cot
batch_size=bbh_small
device_map=11b_inference
nohup python test_bbh.py\
    base_model=${base_model}\
    tokenizer=${base_model}\
    prompt_mode=${prompt_mode}\
    gpu_id=${gpu_id}\
    device_map=${device_map}\
    batch_size=${batch_size}\
    model_size=11b\
    &> logs/beta_${base_model:7}_bbh_${prompt_mode}_eval.log &
tail -f logs/beta_${base_model:7}_bbh_${prompt_mode}_eval.log

model_version=0.0.3.2
tokenizer=t5-large
epoch=0
gpu_id=1
nohup python test_bbh.py\
    model_version=${model_version}\
    tokenizer=${tokenizer}\
    iter=780m\
    epoch=${epoch}\
    gpu_id=${gpu_id}\
    &> logs/beta_${model_version}_bbh_e${epoch}_eval.log &
tail -f logs/beta_${model_version}_bbh_e${epoch}_eval.log

model_version=0.0.4.0
tokenizer=google/flan-t5-large
epoch=2
prompt_mode=cot
gpu_id=0
nohup python test_bbh.py\
    model_version=${model_version}\
    tokenizer=${tokenizer}\
    iter=780m\
    prompt_mode=${prompt_mode}\
    epoch=${epoch}\
    gpu_id=${gpu_id}\
    &> logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log &
tail -f logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log

model_version=0.0.3.1
tokenizer=google/flan-t5-xl
epoch=0
prompt_mode=cot
gpu_id=6
batch_size=bbh_small
nohup python test_bbh.py\
    model_version=${model_version}\
    tokenizer=${tokenizer}\
    iter=3b\
    batch_size=${batch_size}\
    prompt_mode=${prompt_mode}\
    epoch=${epoch}\
    gpu_id=${gpu_id}\
    &> logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log &
tail -f logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log

model_version=0.0.2.8
tokenizer=google/flan-t5-xl
epoch=0
prompt_mode=cot
gpu_id=7
batch_size=bbh_small
nohup python test_bbh.py\
    model_version=${model_version}\
    tokenizer=${tokenizer}\
    iter=3b\
    batch_size=${batch_size}\
    prompt_mode=${prompt_mode}\
    epoch=${epoch}\
    gpu_id=${gpu_id}\
    &> logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log &
tail -f logs/beta_${model_version}_bbh_e${epoch}_${prompt_mode}_eval.log
```