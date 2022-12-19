"""Data utils for the gsm8k dataset"""

import torch
import time
import pickle

from .utils import tprint

CODEX_QUESTIONS_IDX_PATH = 'processed_data/codex_questions_idx.pkl'
CODEX_ANSWERS_PATH = 'processed_data/codex_answers.pkl'
CODEX_PER_STEP_PROBS_IDX_PATH = 'processed_data/codex_per_step_probs_idx.pkl'
CODEX_PREDICTION_LABELS_PATH = 'processed_data/codex_prediction_labels.pkl'
CODEX_MASK_AFTER_TRANSFORM_PATH = 'processed_data/codex_mask_after_transform.pkl'
PERMUTED_IDX_PATH = 'processed_data/permuted_idx.pkl'

class GSM8KCodexAugmentedDataset(object):
    """GSM8K dataset with Codex augmented data
    
    TODO: extend this class to support FlanT5 data and original data
    """

    def __init__(self, 
        dataset_name='gsm8k_codex_augmented',
        vocab_size=32128, # FlanT5 tokenizer vocab size
        batch_size=20,
        base_path='',
        codex_questions_idx_path=CODEX_QUESTIONS_IDX_PATH, 
        codex_answers_path=CODEX_ANSWERS_PATH,
        codex_per_step_probs_idx_path=CODEX_PER_STEP_PROBS_IDX_PATH,
        codex_prediction_labels_path=CODEX_PREDICTION_LABELS_PATH,
        codex_mask_after_transform_path=CODEX_MASK_AFTER_TRANSFORM_PATH,
        permuted_idx_path=PERMUTED_IDX_PATH
        ):
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name

        tprint('Loading dataset...')
        questions = pickle.load(open(base_path + codex_questions_idx_path, 'rb'))
        gold_answers = pickle.load(open(base_path + codex_answers_path, 'rb'))
        codex_per_step_probs = pickle.load(open(base_path + codex_per_step_probs_idx_path, 'rb'))
        codex_prediction_labels = pickle.load(open(base_path + codex_prediction_labels_path, 'rb'))
        codex_mask_after_transform = pickle.load(open(base_path + codex_mask_after_transform_path, 'rb'))
        train_perm_idx = pickle.load(open(base_path + permuted_idx_path, 'rb')) # permute the training data
        tprint('.. finished')

        self.questions = [questions[i] for i in train_perm_idx]
        self.answers = [gold_answers[i] for i in train_perm_idx]
        self.per_step_probs = [codex_per_step_probs[i] for i in train_perm_idx]
        self.per_step_mask = [codex_mask_after_transform[i] for i in train_perm_idx]
        self.prediction_labels = [codex_prediction_labels[i] for i in train_perm_idx]
        return 

    # def get_train_batches_with_in_context_example(self, batch_size):
    #     """One question per batch, prepend different in-context example before the question"""
    #     return 

    def decode_batch(self, batch, tokenizer):
        """"Decode a batch input-output to string"""
        decoded_batch = []
        for q, a, tp, m in zip(batch['questions'], batch['answers'], batch['per_step_probs'], batch['per_step_mask']):
            q_token = tokenizer.convert_ids_to_tokens(q)
            a_token = a
            tpm = []
            for tp_i, m in zip(tp, m):
                tmp_i = {'mask': m, 'tokens':[], 'probs':[]}
                for t, p in tp_i:
                    tmp_i['tokens'].extend(tokenizer.convert_ids_to_tokens([t]))
                    tmp_i['probs'].append(p)
                tpm.append(tmp_i)
            decoded_b = {'question': q_token, 'answer': a_token, 'per_step_probs': tpm}
            decoded_batch.append(decoded_b)
        return decoded_batch

    def process_batch(self, 
                      tokenizer, 
                      batch, 
                      src_prefix='Question: ', 
                      tgt_prefix="Let's think step by step: "
                      ):
        """Pad the batch to the same length, and convert to tensor

        Add "Question: " before the question
        Add "Let's think step by step: " before the answer
        """
        # construct prefix
        src_prefix_idx = tokenizer(src_prefix)['input_ids'][:-1]
        tgt_prefix_idx = tokenizer(tgt_prefix)['input_ids'][:-1]
        tgt_prefix_targets = self.construct_label_prefix(tgt_prefix_idx)

        # construct source input
        src_input_ids = []
        for q in batch['questions']:
            inputs = list(src_prefix_idx + q)
            src_input_ids.append(inputs)
    
        # construct target input output
        tgt_input_ids = []
        tgt_targets = []
        underscore_id = tokenizer.get_vocab()['▁']
        for tp_i in batch['per_step_probs']:
            tgt_input_i = []
            tgt_target_i = []
            # import ipdb; ipdb.set_trace()
            for tp_ij in tp_i:
                t_ij = tp_ij[0][0]
                tgt_input_i.append(t_ij)
            tgt_target_i = self.construct_label(tp_i, underscore_id)

            tgt_input_i = list(tgt_prefix_idx + tgt_input_i)
            tgt_target_i = list(tgt_prefix_targets + tgt_target_i)

            tgt_input_ids.append(tgt_input_i)
            tgt_targets.append(tgt_target_i)

        # construct tgt_mask
        # import ipdb; ipdb.set_trace()
        tgt_mask = list(batch['per_step_mask'])
        tgt_mask_ = []
        for m in tgt_mask:
            tgt_mask_.append([1] * len(tgt_prefix_idx) + list(m))
        tgt_mask = tgt_mask_

        # import ipdb; ipdb.set_trace()
        assert(len(tgt_mask[0]) == len(tgt_input_ids[0]) == len(tgt_targets[0]))

        # pad tgt_input_ids and tgt_targets
        pad_id = tokenizer.pad_token_id
        max_tgt_len = max([len(tgt_i) for tgt_i in tgt_input_ids])
        for tgt_input_i, tgt_mask_i in zip(tgt_input_ids, tgt_mask):
            tgt_input_i.extend([pad_id] * (max_tgt_len - len(tgt_input_i)))
            tgt_mask_i.extend([0] * (max_tgt_len - len(tgt_mask_i)))

        # import ipdb; ipdb.set_trace()

        pad_array = [0] * self.vocab_size
        pad_array[pad_id] = 1
        for tgt_target_i in tgt_targets:
            len_diff = max_tgt_len - len(tgt_target_i)
            for _ in range(len_diff):
                tgt_target_i.append(list(pad_array))
        assert(len(tgt_mask[0]) == len(tgt_input_ids[0]) == len(tgt_targets[0]))

        # NOTE: T5 implementation automatically shifts the tgt_input_ids to the right by 1
        out_dict = {"src_input_ids": torch.tensor(src_input_ids),
                    "tgt_input_ids": torch.tensor(tgt_input_ids), # inside transformer, there will be a start_token added to the input
                    "tgt_targets": torch.tensor(tgt_targets),
                    "tgt_mask": torch.tensor(tgt_mask)
                    }
        return out_dict

    def construct_label_prefix(self, prefix_idx):
        per_step_p = []
        for idx in prefix_idx:
            step_j_target = [0] * self.vocab_size
            step_j_target[idx] = 1
            per_step_p.append(step_j_target)
        return per_step_p

    def construct_label(self, tp, pad_id):
        per_step_p = []
        for tp_j in tp:
            step_j_target = [0] * self.vocab_size
            add_prob = 0
            for t, p in tp_j:
                step_j_target[t] = p
                add_prob += p 
            if(add_prob == 0): # take care of the padded token '_'
                step_j_target[pad_id] = 1
            per_step_p.append(step_j_target)
        return per_step_p

    def get_train_batches(self, 
                          batch_size, 
                          target_answer_label=1,
                          questions=None, 
                          answers=None, 
                          per_step_probs=None, 
                          per_step_mask=None, 
                          prediction_labels=None,
                          max_token_in_batch=-1,
                          ):
        """One question per batch, only retain correct answer for the question

        Args:
            batch_size (int): batch size
            target_answer_label (int, optional): whether only using correct answer or using all answers
                1 for correct answer, 0 for incorrect answer, 2 for no filter. Defaults to 1.
        """
        def _compare_label(target_answer_label, l):
            if(target_answer_label == 2): return True 
            else: return l == target_answer_label

        batches = []
        for q, a, tp, m, l in zip(questions, answers, per_step_probs, per_step_mask, prediction_labels):
            batch_q = []
            batch_a = []
            batch_tp = []
            batch_m = []
            for tp_i, m_i, l_i in zip(tp, m, l):
                if(_compare_label(target_answer_label, l_i)):
                    batch_q.append(q)
                    batch_a.append(a)
                    batch_tp.append(tp_i)
                    if(self.dataset_name == 'gsm8k_codex_augmented'): m_i.append(1)
                    batch_m.append(m_i)
                if(len(batch_q) >= batch_size): break

            batch = {'questions': batch_q,
                    'answers': batch_a,
                    'per_step_probs': batch_tp,
                    'per_step_mask': batch_m
                    }
            if(len(batch['questions']) > 0):
                batches.append(batch)
        return batches 


class GSM8KCodexAugmentedInContextDataset(object):

    def __init__(self):
        self.gsm8k_train = ... 
        self.questions = ... 
        self.answers = ... 
        self.codex_predictions = ...
        self.prediction_labels = ... 
        return 

    def process_batch(self):
        return 

    def get_train_batches(self):
        # Attention: add the start token to the input -- previously it is not added

        # Step 1. Build zero-shot answer-only batches 
        # Add prefix "Q: " and "A: "

        # Step 2. Build in-context answer-only batches
        # sample 4 in-context demonstrations from the GSM8K training set

        # Step 3. Build zero-shot chain-of-thought batches 

        # Step 4. Build in-context chain-of-thought batches

        # Step 5. Mix all batches together
        return 