"""
Decode GSM8K training data using the T5 model.
TODO: adaptive batch size, such that max_len * batch_size = const 

nohup python -u scripts/flan_t5_decode_gsm8k.py\
  --gpu_id 0,1,2,3\
  --output_path outputs/gsm8k/train_flan_t5_complex.txt\
  --debug 0\
  --num_sample 20\
  --batch_size 10\
  --log_interval 5\
  &> logs/flan_t5_decode_gsm8k.log &

tail -f logs/flan_t5_decode_gsm8k.log

TODO: add evaluation code
"""

import time 
import torch
import re
import argparse
import os
import pytz 

import numpy as np
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

OUTPUT_PATH = 'outputs/gsm8k/train_flan_t5_complex.txt'
PROMPT_PATH = 'lib_prompt/prompt_complex.txt'

def tprint(str):
  timezone = pytz.timezone("America/Vancouver")
  timenow = datetime.now(timezone)
  currenttime= timenow.strftime("%m/%d/%Y, %H:%M:%S")
  print(currenttime + ' ' + str)
  return  

def define_argument():
  ## add commandline arguments, initialized by the default configuration
  parser = argparse.ArgumentParser()   

  # general 
  parser.add_argument("--gpu_id", default='0', type=str)
  parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
  parser.add_argument("--debug", default=0, type=int)
  parser.add_argument("--num_sample", default=50, type=int)
  parser.add_argument("--batch_size", default=10, type=int) # NOTE: num_sample should be divisible by batch_size
  parser.add_argument("--log_interval", default=10, type=int)
  
  args = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  return args

def process_prompt_complex(prompt, question):
  """
  Append the question to the prompt.
  Add "Let's think step by step." to the end of the prompt.
  """
  prompt_q = prompt + '\nQuestion: ' + question + "\nLet's think step by step"
  return prompt_q

def flan_decode(model, tokenizer, prompt, question, batch_size=10, num_sample=50):
  """
  Decode a question using the T5 model.
  Return the decoded answer and the per-token log probabilities.
  """

  prompt_q = process_prompt_complex(prompt, question)
  # import ipdb; ipdb.set_trace()
  out_scores, out_texts = [], [] 
  with torch.no_grad():
    input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to('cuda:0')
    for _ in range(num_sample // batch_size):
      # print('.')
      # import ipdb; ipdb.set_trace()
      out_dict = model.generate(input_ids,
                                do_sample=True, 
                                max_new_tokens=256, 
                                output_scores=True, 
                                return_dict_in_generate=True,
                                num_return_sequences=batch_size
                                )
      # import ipdb; ipdb.set_trace()
      scores_ = [] # [B, T, 6, 6]
      for si in out_dict['sequences']:
        out_text = tokenizer.decode(si)
        out_texts.append(out_text)

      for b, si in enumerate(out_dict['sequences']): # si.size() = [T]
        seq_id_p = []
        for t, sic in enumerate(si): 
          # import ipdb; ipdb.set_trace()
          if(t == 0): continue # TODO: check if the index matches or need to shift
          sicp = F.softmax(out_dict['scores'][t - 1][b], dim=-1)[sic]
          seq_id_p.append([(tokenizer.convert_ids_to_tokens([sic])[0], sicp)])
        scores_.append(seq_id_p)

      for t, s in enumerate(out_dict['scores']): # s.size() = [batch, vocab]
        for b, sc in enumerate(s): # sc.size() = [vocab]
          top_prob, top_ind = F.softmax(sc, dim=-1).topk(5)
          # import ipdb; ipdb.set_trace()
          for ti, tp in zip(top_ind, top_prob):
            scores_[b][t].append((tokenizer.convert_ids_to_tokens([ti])[0], tp))

      out_scores.extend(scores_)

  # out_texts.size = [num_sample, T]
  # out_scores.size = [num_sample, T, 6, 6]
  return out_scores, out_texts

def write_output(out_scores, out_texts, question, qid, answer, fout):
  """Write output to file.
  Output consists of 
    question
    gold answer 
    num_sample * decoded answer
    num_sample * per step top 5 probabilities
  """
  fout.write(('Question %d: ' % qid) + question + '\n')
  fout.write('Answer: ' + answer + '\n')

  for i, (out_topk, out_text) in enumerate(zip(out_scores, out_texts)):
    fout.write(('Model output %d: ' % i) + out_text + '\n')
    fout.write('Per-step decode: ')
    for top_wp in out_topk:
      for w, p in top_wp:
        fout.write('<<' + repr(w) + '>>' + ' ' + '%.4f' % p + ' ')
      fout.write(' ||| ')
    fout.write('\n')

  fout.write('\n\n\n\n')
  return 

def main():
  args = define_argument()

  # load the dataset
  gsm8k = load_dataset('gsm8k', 'main')

  # load the model
  tprint('Loading the model ... ')
  start_time = time.time()
  tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
  model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map='auto')
  tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

  # load the prompt
  prompt_complex = open(PROMPT_PATH).read()
  
  # decode the dataset
  if(args.debug): end_id = 3
  else: end_id = len(gsm8k['train'])

  tprint('Start decoding ... ')
  with open(args.output_path, 'w') as fout:
    start_time = time.time()

    for i, (q, a) in enumerate(zip(gsm8k['train']['question'][:end_id], gsm8k['train']['answer'][:end_id])):
      # pass
      # import ipdb; ipdb.set_trace()

      out_scores, out_texts = flan_decode(model, tokenizer, prompt_complex, q, args.batch_size, args.num_sample)
      write_output(out_scores, out_texts, q, i, a, fout)
    
      if(i % args.log_interval == 0): 
        tprint('Decoded %d / %d questions. time %.1fs' % (i, end_id, time.time() - start_time))
  return 

if __name__ == '__main__':
  main()