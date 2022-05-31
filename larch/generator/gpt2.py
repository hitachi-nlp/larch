from typing import Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .base import BaseGenerator


class GPT2Generator(BaseGenerator):

    def __init__(self, pretrained_model_name_or_path: str):
        tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path)
        super().__init__(tokenizer, 1024, False)
        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path, pad_token_id=self.tokenizer.eos_token_id)
        self.gpu_enabled = torch.cuda.is_available()
        if self.gpu_enabled:
            self.model.cuda()

    def generate(self, context: str, prompt: str, max_length: int) -> Tuple[str, str]:
        inputs = self.tokenizer.encode(context + prompt, return_tensors='pt')
        if self.gpu_enabled:
            inputs = inputs.cuda()
        input_len = len(inputs[0])
        outputs = self.model.generate(
            inputs,
            max_length=input_len + max_length,
            min_length=input_len + min(10, max_length),  # Avoid falling back to naive generation of EOS
            do_sample=False,
            num_beams=10,
            no_repeat_ngram_size=4
        )
        text = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True)
        return prompt + text, text

    def calculate_perplexity(self, context: str, content: str, num_content_tokens: int) -> float:
        # FIXME: Implement batched calculation for better performance
        assert len(content) > 0
        if (num_content_tokens + 1) > self.max_context_length:
            raise ValueError(
                'num_content_tokens must be smaller than model\'s maximum '
                f'contex length ({self.max_context_length}) - 1.')
        content_tokens = self.tokenizer.encode(
            content, return_tensors='pt', verbose=False,
            add_special_tokens=False)[:, :num_content_tokens]

        context_tokens = self.tokenizer.encode(
            context, return_tensors='pt', verbose=False)
        if len(context_tokens[0]) > 0:
            inputs = torch.cat((context_tokens, content_tokens), dim=1)
        else:
            inputs = content_tokens
        assert len(inputs[0]) <= self.max_context_length
        with torch.no_grad():
            labels = inputs.detach().clone()
            # +1 to ignore the first token. This is because
            # OpenAI API returns null for the first token, thus this breaks when
            # token_offset is 0. We want to make sure we do fair comparison.
            labels[0, :len(context_tokens[0]) + 1] = -100
            if self.gpu_enabled:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = self.model(input_ids=inputs, labels=labels)
        return float(np.exp(outputs.loss.item()))


class GPT2BaseGenerator(GPT2Generator):
    def __init__(self):
        super().__init__('gpt2')


class GPT2XLGenerator(GPT2Generator):
    def __init__(self):
        super().__init__('gpt2-xl')
