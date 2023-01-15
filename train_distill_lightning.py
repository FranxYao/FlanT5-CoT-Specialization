import time 
import torch
import re
import argparse
import os
import hydra

import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils import tprint, kl_divergence
from src.data_utils import GSM8KCodexAugmentedInContextDataset
from omegaconf import DictConfig, OmegaConf
from deepspeed.ops.adam import FusedAdam
# from train_distill_simple import compute_loss_match_dist, compute_loss_nll

class GSM8KCodexAugDataset(Dataset):
    def __init__(self, train_batches):
        super().__init__()

        self.batches = train_batches
        return 

    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        return self.batches[idx]


class CollateFn(object):

    def __init__(self, tokenizer, vocab_size=32128):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        return 

    def __call__(self, batch, debug=0):
        assert(len(batch) == 1)
        batch = batch[0]

        tokenizer = self.tokenizer
        vocab_size = self.vocab_size

        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id
        vocab = tokenizer.get_vocab()

        questions = list(b['question'] for b in batch)
        answers = list(b['answer'] for b in batch)
        answer_gold = list(b['answer_gold'] for b in batch)
        questions = tokenizer(questions, padding=True, return_tensors='pt')
        answers = tokenizer(answers, padding=True, return_tensors='pt')

        # answer_ids = answers['input_ids'].masked_fill(1 - answers['attention_mask'], -100)
        targets = answers['input_ids']
        answer_ids = answers['input_ids']
        batch_size = answer_ids.size(0)
        bos = torch.tensor([tokenizer.decoder_start_token_id] * batch_size).view(batch_size, 1)
        answer_ids = torch.cat([bos, answer_ids[:, :-1]], dim=1)

        max_len = answer_ids.size(1)
        target_dist = torch.zeros(batch_size, max_len, vocab_size)
        
        # distribution match
        if('chain_of_thought' in batch[0]['type']):
            target_dist[:, :, pad_id] = 1
            for bi, b in enumerate(batch):
                for sj, step_p in enumerate(b['per_step_probs']):
                    for w in step_p:
                        wid = vocab[w]
                        target_dist[bi, sj, wid] = step_p[w]
                    target_dist[bi, sj, pad_id] = 0
                target_dist[bi, sj + 1, end_id] = 1
                target_dist[bi, sj + 1, pad_id] = 0
        else: target_dist = None
        
        batch_dict = {'questions': questions['input_ids'],
                      'question_mask': questions['attention_mask'],    
                      'answers': answer_ids,
                      'answer_mask': answers['attention_mask'],
                      'targets': targets,
                      'target_dist': target_dist,
                      'answer_gold': answer_gold,
                      'answer_label': batch[0]['answer_label'],
                      'type': batch[0]['type']
                      }

        if(debug and 'chain_of_thought' in batch[0]['type']):
            target_id_from_dist = torch.argmax(target_dist, dim=2)
            batch_dict['target_id_from_dist'] = target_id_from_dist
            mask = answers['attention_mask']
            checksum = ((target_id_from_dist * mask - targets * mask).float() != 0.).sum(-1)
            batch_dict['checksum'] = checksum
        return batch_dict

def compute_loss_nll(lm_logits, targets, mask, answer_label):
    loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1), reduction='none')
    loss = (loss * mask.view(-1)).sum() / mask.sum()

    if(answer_label == 0): loss = -loss # negative sample
    return loss

def compute_loss_match_dist(logits, teacher_dist, mask):
    """Compute loss for the model

    logits: [batch_size, seq_len, vocab_size]
    teacher_dist: [batch_size, seq_len, vocab_size], teacher distribution from Codex
    """
    kld = kl_divergence(teacher_dist, F.softmax(logits, dim=-1))
    loss = (kld * mask).sum() / mask.sum()
    return loss

class DistillFlanT5(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        # TODO: check how to do model parallelism 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        # if(args.base_model in ['t5-3b', 't5-11b', 'google/flan-t5-xl', 'google/flan-t5-xxl']): # Multi-GPU model parallelism
        #     self.model.parallelize(args.device_map)
        # else: # single A100
        #     self.model.to('cuda')
        return 

    # def forward(self):
    #     return 

    # def configure_sharded_model(self):
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    #     return 

    def training_step(self, batch, batch_idx):
        args = self.args
        # device = self.model.device
        out_dict = self.model(input_ids=batch['questions'],
                        attention_mask=batch['question_mask'],
                        decoder_input_ids=batch['answers'],
                        decoder_attention_mask=batch['answer_mask'],
                        return_dict=True
                        )

        lm_logits = out_dict['logits']
        if(args.loss_type == 'match_sample'):
            loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'])
            # total_loss.append(loss.item())
        elif(args.loss_type == 'match_distribution'):
            if('chain_of_thought' in batch['type']):
                loss = compute_loss_match_dist(lm_logits, batch['target_dist'], batch['answer_mask'])
            else: 
                loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'])
            # total_loss.append(loss.item())
        else:
            raise NotImplementedError

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        # TODO: learning rate scheduler, see https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        optimizer = FusedAdam(self.model.parameters(), lr=self.args.lr)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr) 
        # optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.args.lr)
        return optimizer

@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    ## arguments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    pl.seed_everything(15213)

    ## data
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = GSM8KCodexAugmentedInContextDataset(args)
    train_batches = dataset.get_train_batches()

    train_set = GSM8KCodexAugDataset(train_batches)
    collate_fn = CollateFn(tokenizer)

    # TODO: test train dataloader
    train_dataloader = DataLoader(train_set, 
                                   batch_size=1, 
                                   shuffle=False, 
                                   collate_fn=collate_fn
                                   )
    # import ipdb; ipdb.set_trace()
    tprint('Loading the model ... ')
    model = DistillFlanT5(args)
    tokenizer.decoder_start_token_id = model.model.config.decoder_start_token_id

    # TODO: batch size finder
    # TODO: save checkpoint
    # TODO: print log 
    ngpu = len(args.gpu_id.split(','))
    tprint('Start training, %d GPUs ... ' % ngpu)
    trainer = pl.Trainer(accumulate_grad_batches=args.grad_accum_steps, 
                      gradient_clip_val=args.gradient_clip_val,
                      accelerator="gpu", 
                      devices=4, 
                      strategy="deepspeed_stage_3_offload", 
                    #   strategy="fsdp",
                      precision="bf16",
                    #   enable_checkpointing=False,
                      max_epochs=args.num_epoch,
                      )
    trainer.fit(model, train_dataloaders=train_dataloader)

if __name__ == '__main__':
    main()