from itertools import chain
from typing import Optional, List, Dict, Set

from . import encoder_decoder
from . import gpt2
from . import gpt3
from .base import BaseGenerator

# models that do not require any options to be set
__NO_ARGS_MODELS = {
    'gpt2': gpt2.GPT2BaseGenerator,
    'gpt2-xl': gpt2.GPT2XLGenerator
}
__OPENAI_MODELS = {
    'openai/text-davinci-002': gpt3.GPT3Davinci002Generator,
    'openai/text-davinci-003': gpt3.GPT3Davinci003Generator
}
IMPLEMENTED_MODEL_NAMES = sorted(chain(
    __NO_ARGS_MODELS.keys(),
    __OPENAI_MODELS.keys()))
AVAILABLE_MODELS: Optional[Dict[str, BaseGenerator]] = None


def init_models(
        loaded_models: Optional[List[str]], openai_api_key: Optional[str],
        encoder_decoder_model_paths: Dict[str, str]):
    global AVAILABLE_MODELS
    loaded_models = set(loaded_models) if loaded_models is not None else None
    if loaded_models is not None and len(loaded_models - set(IMPLEMENTED_MODEL_NAMES)) > 0:
        raise ValueError(
            f'Unknown model(s) {loaded_models - set(IMPLEMENTED_MODEL_NAMES)}')

    if loaded_models is not None and len(set(loaded_models) & set(encoder_decoder_model_paths)) > 0:
        raise ValueError(
            'Name collision in loaded_models and encoder_decoder_model_paths '
            f'{set(loaded_models) & set(encoder_decoder_model_paths)}.'
        )

    if loaded_models is not None and len(loaded_models) == 0 and len(encoder_decoder_model_paths) == 0:
        raise ValueError('No models were loaded (both load_models and encoder_decoder_model_paths are empty).')

    assert loaded_models is None or len(loaded_models) > 0 or len(encoder_decoder_model_paths) > 0
    AVAILABLE_MODELS = dict()
    for model_name, model_cls in __NO_ARGS_MODELS.items():
        if loaded_models is None or model_name in loaded_models:
            AVAILABLE_MODELS[model_name] = model_cls()
            if loaded_models is not None:
                loaded_models.remove(model_name)

    for model_name, model_cls in __OPENAI_MODELS.items():
        if loaded_models is None or model_name in loaded_models:
            if loaded_models is not None and openai_api_key is None:
                raise ValueError(
                    f'"{model_name}" was specified in loaded_models but '
                    'openai_api_key was not set.')
            if openai_api_key is not None:
                AVAILABLE_MODELS[model_name] = model_cls(openai_api_key)
                if loaded_models is not None:
                    loaded_models.remove(model_name)

    assert loaded_models is None or len(loaded_models) == 0

    for model_name, model_path in encoder_decoder_model_paths.items():
        AVAILABLE_MODELS[model_name] = encoder_decoder.EncoderDecoderGenerator(model_path)

    assert len(AVAILABLE_MODELS) > 0
