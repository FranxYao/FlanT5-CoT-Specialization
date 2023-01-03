"""Simple training scripts for distilling T5

This script is used for verifying that distillation from Codex can help FlanT5 improve performance on GSM8K

After the verification, we will transfer the code to huggingface trainer
"""

import time 
import torch
import re
import argparse
import os
import hydra

import numpy as np
import torch.nn.functional as F

# from torch import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils import tprint, kl_divergence
from src.data_utils import GSM8KCodexAugmentedInContextDataset
from omegaconf import DictConfig, OmegaConf


def compute_loss_match_dist(logits, teacher_dist, mask, device):
    """Compute loss for the model

    logits: [batch_size, seq_len, vocab_size]
    teacher_dist: [batch_size, seq_len, vocab_size], teacher distribution from Codex
    """
    kld = kl_divergence(teacher_dist.to(device), F.softmax(logits, dim=-1))
    loss = (kld * mask.to(device)).sum() / mask.to(device).sum()
    return loss

def compute_loss_unlikelihood(lm_logits, targets, mask, answer_label, device):
    """Unlikelihood loss for wrong reasoning chains, loss originally proposed in 
    Welleck et. al. 2019, Neural Text Generation with Unlikelihood Training. 
    """
    # import ipdb; ipdb.set_trace()
    loss = -F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.to(device).view(-1), reduction='none')
    loss = - (1 - loss.exp() + 1e-5).log()
    loss = (loss * mask.to(device).view(-1)).sum() / mask.sum()
    return loss

def compute_loss_nll(lm_logits, targets, mask, answer_label, device):
    loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.to(device).view(-1), reduction='none')
    loss = (loss * mask.to(device).view(-1)).sum() / mask.sum()

    if(answer_label == 0): loss = -loss # negative sample
    return loss

def train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler=None):
    """Training loop"""
    device = model.device
    global_step = 0
    positive_loss, negative_loss, total_loss = [], [], []
    tprint('Start trainig, %d / %d = %d batches in total' % (len(train_batches), args.grad_accum_steps, len(train_batches) // args.grad_accum_steps))
    for e in range(args.num_epoch):

        # training epoch 
        ans_label_cnt = 0
        for i, batch in enumerate(train_batches):
            batch = dataset.process_batch(tokenizer, batch)

            # TODO: check what device will be when using multi-gpu
            out_dict = model(input_ids=batch['questions'].to(device),
                             attention_mask=batch['question_mask'].to(device),
                             decoder_input_ids=batch['answers'].to(device),
                             decoder_attention_mask=batch['answer_mask'].to(device),
                             return_dict=True
                             )

            lm_logits = out_dict['logits']
            
            # TODO: seperate transition loss and in-step loss to check which part is hard to learn
            # TODO: distribution match

            # gradient accumulation
            if(args.loss_type == 'match_sample'):
                loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'], device)
                total_loss.append(loss.item())
            elif(args.loss_type == 'match_distribution'):
                if('chain_of_thought' in batch['type']):
                    loss = compute_loss_match_dist(lm_logits, batch['target_dist'], batch['answer_mask'], device)
                else: 
                    loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'], device)
                total_loss.append(loss.item())
            else:
                raise NotImplementedError

            # if(batch['answer_label'] == 1): 
            #     loss = compute_loss_nll(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'], device)
            #     positive_loss.append(loss.item())
            # else: # negative sample 
            #     loss = args.neg_loss_alpha * compute_loss_unlikelihood(lm_logits, batch['targets'], batch['answer_mask'], batch['answer_label'], device)
            #     negative_loss.append(loss.item())

            # ans_label_cnt += batch['answer_label']
            loss /= args.grad_accum_steps
            loss.backward()

            if ((i + 1) % args.grad_accum_steps == 0) or (i + 1 == len(train_batches)):
                # print(ans_label_cnt, args.grad_accum_steps)
                # assert(ans_label_cnt in [args.grad_accum_steps, 0]) # make sure all batches are positive-only or negative-only
                # ans_label_cnt = 0
                
                optimizer.step()
                optimizer.zero_grad()
                if(scheduler is not None): scheduler.step() 
                global_step += 1
                
                if global_step % args.log_interval == 0:
                    # tprint(f'Epoch: %d, Iter: %d, Global step %d, Lr: %.4g, Positive Loss: %.4f, Negative Loss: %.4f' % 
                    #     (e, i, global_step, scheduler.get_last_lr()[0], 
                    #     np.average(positive_loss), np.average(negative_loss)))
                    tprint(f'Epoch: %d, Iter: %d, Global step %d, Lr: %.4g, Loss: %.4f' % 
                        (e, i, global_step, scheduler.get_last_lr()[0], 
                        np.average(total_loss)))
                    total_loss = []
                    # positive_loss = []
                    # negative_loss = []
                # import ipdb; ipdb.set_trace()
            
            if(i > 0 and i % args.save_per_step == 0):
                save_path = args.save_path + args.model_version + '_epoch_%d_iter_%d' % (e, i)
                tprint('Saving model at %s' % save_path)
                model.save_pretrained(save_path)

        # validation on subset of training data 
        save_path = args.save_path + args.model_version + '_epoch_%d_end' % e
        tprint('Model %s Epoch %d finished, saving model at %s' % (args.model_version, e, save_path))
        model.save_pretrained(save_path)
    return 

@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    ## arguments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ## data
    dataset = GSM8KCodexAugmentedInContextDataset(args)
    train_batches = dataset.get_train_batches()

    ## model
    tprint('Loading the model ... ')
    start_time = time.time()
    # tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # TODO: change code using lightning trainer and FairScale/ DeepSpeed
    # model = T5ForConditionalGeneration.from_pretrained(args.base_model) 
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.parallelize(args.device_map)

    tokenizer.decoder_start_token_id = model.config.decoder_start_token_id # special treatment for T5

    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.num_warmup_steps, 
                                                num_training_steps=10*len(train_batches))

    ## training 
    train(args, tokenizer, model, dataset, train_batches, optimizer, scheduler)
    return 

if __name__ == '__main__':
    main()