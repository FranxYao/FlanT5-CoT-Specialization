import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from src.data_utils import GSM8KCodexAugmentedInContextDataset
from omegaconf import DictConfig, OmegaConf

args = OmegaConf.load('src/conf/config.yaml')
args['batch_size'] = OmegaConf.load('src/conf/batch_size/3b.yaml')
args['data_formats'] = OmegaConf.load('src/conf/data_formats/normal.yaml')
args['grad_accum_steps'] = 20

import sys 
sys.path.append('..')

from train_distill_lightning import GSM8KCodexAugDataset, CollateFn

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
tokenizer.decoder_start_token_id = 0 # special treatment 

dataset = GSM8KCodexAugmentedInContextDataset(args)
train_batches = dataset.get_train_batches()

train_set = GSM8KCodexAugDataset(train_batches)
collate_fn = CollateFn(tokenizer)

train_dataloader = DataLoader(train_set, 
                                batch_size=1, 
                                shuffle=False, 
                                collate_fn=collate_fn
                                )
batch = next(iter(train_dataloader))