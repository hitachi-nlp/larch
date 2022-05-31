from typing import Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from transformers import GPT2TokenizerFast

from larch.context_creator.common import truncate_by_token_length
from .base import BaseGenerator


class GPT3Generator(BaseGenerator):
    ENDPOINT = 'https://api.openai.com/v1/completions'

    def __init__(self, model: str, key: str):
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        super().__init__(tokenizer, 4097, False)
        self.model = model
        self.api_key = key

    def _create_header(self) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        return headers

    @staticmethod
    def _create_session():
        session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def generate(self, context: str, prompt: str, max_length: int) -> Tuple[str]:
        headers = self._create_header()
        # FIXME: This will cause error when input is too long
        input = context + prompt
        data = {
            "model": self.model,
            "prompt": input,
            "max_tokens": max_length,
            "temperature": 0.,
            "top_p": 1.,
            "n": 1,
            "stream": False,
            "logprobs": None,
            "stop": None
        }
        session = self._create_session()
        r = session.post(self.ENDPOINT, json=data, headers=headers)
        r.raise_for_status()
        text = r.json()['choices'][0]['text']
        return prompt + text, text

    def calculate_perplexity(self, context: str, content: str, num_content_tokens: int) -> float:
        if (num_content_tokens + 1) > self.max_context_length:
            raise ValueError(
                'num_content_tokens must be smaller than model\'s maximum '
                f'contex length ({self.max_context_length}) - 1.')

        headers = self._create_header()
        content = truncate_by_token_length(
            self.tokenizer, content, num_content_tokens)

        input = context + content
        data = {
            "model": self.model,
            "prompt": input,
            "max_tokens": 0,
            "temperature": 0.,
            "top_p": 1.,
            "n": 1,
            "stream": False,
            "logprobs": 0,
            "stop": None,
            "echo": True
        }
        session = self._create_session()
        r = session.post(self.ENDPOINT, json=data, headers=headers)
        r.raise_for_status()
        logprobs = r.json()['choices'][0]['logprobs']['token_logprobs']
        char_offsets = r.json()['choices'][0]['logprobs']['text_offset']
        text = r.json()['choices'][0]['text']
        assert input[:len(text)] == text

        content_offset = len(context)
        token_offset = max(range(len(char_offsets)),
                           key=lambda i: (char_offsets[i] >= content_offset, -i))
        # token_offset + 1 to ignore the first token. This is because
        # OpenAI API returns null for the first token, thus this breaks when
        # token_offset is 0. We want to make sure we do fair comparison.
        return float(np.exp(-np.average(logprobs[token_offset + 1:])))


class GPT3Davinci002Generator(GPT3Generator):
    def __init__(self, key: str):
        super().__init__(model='text-davinci-002', key=key)


class GPT3Davinci003Generator(GPT3Generator):
    def __init__(self, key: str):
        super().__init__(model='text-davinci-003', key=key)
