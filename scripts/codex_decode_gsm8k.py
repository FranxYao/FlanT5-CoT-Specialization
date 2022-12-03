"""
Decode GSM8K training data using Codex

API_KEY=
nohup python -u scripts/codex_decode_gsm8k.py\
  --output_path outputs/gsm8k/train_codex_complex_from_2489.txt\
  --debug 0\
  --num_sample 20\
  --batch_size 10\
  --log_interval 1\
  --api_key $API_KEY\
  --from_index 2489\
  &> logs/codex_decode_gsm8k_from_2489.log &

tail -f logs/codex_decode_gsm8k_from_2489.log

TODO: add evaluation code
"""

import time 
import torch
import re
import pytz
import argparse
import openai

import numpy as np

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed 


OUTPUT_PATH = 'outputs/gsm8k/train_codex_complex.txt'
PROMPT_PATH = 'lib_prompt/prompt_complex.txt'


def define_argument():
  ## add commandline arguments, initialized by the default configuration
  parser = argparse.ArgumentParser()   

  # general 
  parser.add_argument("--output_path", default=OUTPUT_PATH, type=str)
  parser.add_argument("--debug", default=0, type=int)
  parser.add_argument("--num_sample", default=50, type=int)
  parser.add_argument("--batch_size", default=25, type=int)
  parser.add_argument("--api_key", default="", type=str)
  parser.add_argument("--log_interval", default=10, type=int)
  parser.add_argument("--from_index", default=0, type=int)
  
  args = parser.parse_args()
  openai.api_key = args.api_key
  return args


def tprint(str):
  timezone = pytz.timezone("America/Vancouver")
  timenow = datetime.now(timezone)
  currenttime= timenow.strftime("%m/%d/%Y, %H:%M:%S")
  print(currenttime + ' ' + str)
  return  

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]), stop=stop_after_attempt(1000))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def process_prompt_complex(prompt, question):
  """
  Append the question to the prompt.
  Add "Let's think step by step." to the end of the prompt.
  """
  prompt_q = prompt + '\nQuestion: ' + question + "\nLet's think step by step"
  return prompt_q

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

def codex_decode(prompt, question, num_sample=1, batch_size=1):
  """
  Decode a question using the T5 model.
  Return the decoded answer and the per-token log probabilities.
  """

  prompt_q = process_prompt_complex(prompt, question)
  # import ipdb; ipdb.set_trace()
  # response = openai.Completion.create(model="code-davinci-002", 
  #                                 prompt=prompt_q, 
  #                                 temperature=0.5, 
  #                                 max_tokens=256,
  #                                 n=num_sample,
  #                                 logprobs=5)
  response = []
  for _ in range(num_sample // batch_size):
    r = completion_with_backoff(model="code-davinci-002", 
                                    prompt=prompt_q, 
                                    temperature=0.5, 
                                    max_tokens=256,
                                    n=num_sample,
                                    logprobs=5
                                    )
    response.extend(r['choices'])
                                
  out_texts = []
  out_dicts = []
  for ans in response: 
    ans_str_, residual = extract_ans(ans['text'])
    out_texts.append(ans_str_)
    out_dict = {}
    out_dict['tokens'] = ans['logprobs']['tokens']
    out_dict['token_logprobs'] = ans['logprobs']['token_logprobs']
    out_dict['top_logprobs'] = ans['logprobs']['top_logprobs']
    out_dicts.append(out_dict)
  return out_dicts, out_texts

def write_output(out_dicts, out_texts, question, qid, answer, fout):
  """Write output to file.
  Output consists of 
    question
    gold answer 
    num_sample * decoded answer
    num_sample * per step top 5 probabilities
  """
  fout.write(('Question %d: ' % qid) + question + '\n')
  fout.write('Answer: ' + answer + '\n')
  for i, (out_dict, out_text) in enumerate(zip(out_dicts, out_texts)):
    fout.write(('Model output %d: ' % i) + out_text + '\n')
    fout.write('Per-step decode: ')
    for ti, pi, top5 in zip(out_dict['tokens'], out_dict['token_logprobs'], out_dict['top_logprobs']):
      fout.write('<<' + repr(ti) + '>>' + ' %.4f' % np.exp(pi) + ' ')
      # import ipdb; ipdb.set_trace()
      for t_ in top5:
        p_ = top5[t_]
        fout.write('<<' + repr(t_) + '>>' + ' %.4f' % np.exp(p_) + ' ')
      fout.write('||| ')
    fout.write('\n')

  fout.write('\n\n\n\n')
  return 

def main():
  args = define_argument()

  # load the dataset
  gsm8k = load_dataset('gsm8k', 'main')

  # load the prompt
  prompt_complex = open(PROMPT_PATH).read()
  
  # decode the dataset
  if(args.debug): end_id = 10
  else: end_id = len(gsm8k['train'])

  tprint('Start decoding ... ')
  with open(args.output_path, 'w', buffering=1) as fout:
    start_time = time.time()

    for i, (q, a) in enumerate(zip(gsm8k['train']['question'][:end_id], gsm8k['train']['answer'][:end_id])):
      # pass
      # import ipdb; ipdb.set_trace()
      if(i < args.from_index): continue

      out_dicts, out_texts = codex_decode(prompt_complex, q, args.num_sample, args.batch_size)
      write_output(out_dicts, out_texts, q, i, a, fout)
    
      if(i % args.log_interval == 0): 
        tprint('Decoded %d / %d questions. time %.1fs' % (i, end_id, time.time() - start_time))
  return 

if __name__ == '__main__':
  main()