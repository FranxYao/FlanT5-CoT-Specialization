"""Customized trainer for distillation."""

from transformers import Seq2SeqTrainer

class DistillTrainer(Seq2SeqTrainer):

    def compute_loss():
        return # TBC