from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseGenerator


class EncoderDecoderGenerator(BaseGenerator):
    BASE_MODELS_WITH_PREFIX = {'t5'}

    def __init__(self, pretrained_model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path)
        super().__init__(tokenizer, 1024, True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_path)
        self.source_prefix = self.get_prefix(self.model)
        self.gpu_enabled = torch.cuda.is_available()
        if self.gpu_enabled:
            self.model.cuda()

    @classmethod
    def get_prefix(cls, model) -> str:
        return 'summarize: ' if model.config.model_type in cls.BASE_MODELS_WITH_PREFIX else ""

    @staticmethod
    def preprocess(
            tokenizer, source_prefix: str, inputs: List[str], max_source_length,
            padding, return_tensors=None):
        inputs = [source_prefix + inp for inp in inputs]
        return tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True,
            return_tensors=return_tensors)

    def generate(self, context: str, prompt: str, max_length: int) -> Tuple[str, str]:
        context_tokens = self.preprocess(
            self.tokenizer, self.source_prefix, [context], None, False,
            return_tensors='pt').input_ids
        decoder_start_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.gpu_enabled:
            context_tokens = context_tokens.cuda()
        input_len = len(decoder_start_ids)
        # We pass mapping from index to str as there is a bug in current
        # huggingface implementation.
        # ttps://github.com/huggingface/transformers/issues/19602
        # We start index from 1 as index 0 is always a special token
        outputs = self.model.generate(
            context_tokens,
            max_length=input_len + max_length,
            min_length=input_len + min(10, max_length),  # Avoid falling back to naive generation of EOS
            do_sample=False,
            num_beams=10,
            no_repeat_ngram_size=4,
            forced_decoder_ids={i + 1: id for i, id in enumerate(decoder_start_ids)}
        )
        # input_len + 1 to skip the first special token
        text = self.tokenizer.decode(outputs[0][input_len + 1:], skip_special_tokens=True)
        if len(prompt) > 0 and prompt[-1] not in [' ', '\n', '\t']:
            prompt += ' '
        return prompt + text, text

    def calculate_perplexity(self, context: str, content: str, num_content_tokens: int) -> float:
        # FIXME: Implement batched calculation for better performance
        assert len(content) > 0
        context_tokens = self.preprocess(
            self.tokenizer, self.source_prefix, [context], None,
            False, return_tensors='pt').input_ids
        content_tokens = self.tokenizer.encode(
            content, return_tensors='pt', verbose=False,
            add_special_tokens=False)[:, :num_content_tokens]
        with torch.no_grad():
            # +1 to ignore the first token. This is because
            # OpenAI API returns null for the first token, thus this breaks when
            # token_offset is 0. We want to make sure we do fair comparison.
            content_tokens[0, 0] = -100
            if self.gpu_enabled:
                context_tokens = context_tokens.cuda()
                content_tokens = content_tokens.cuda()
            outputs = self.model(input_ids=context_tokens, labels=content_tokens)
        return float(np.exp(outputs.loss.item()))
