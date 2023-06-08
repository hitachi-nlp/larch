import json
import logging
import os

import click
import uvicorn
from fastapi import FastAPI, HTTPException

import larch
from larch.context_creator import ContextCreatorName
from larch.utils.download import cached_model_download, larch_model_url
from larch.postprocessor import postprocess
from larch.request_models import Response, GenerationRequest, \
    GenerationModel, GenerationModels, Generation, Edit

app = FastAPI(
    title='larch-server'
)
logger = logging.getLogger(__name__)


available_models = set()
_context_creator = None

# FIXME: Stop hardcoding this
_MAX_GENERATION_LENGTH = 512
# context creators may return more tokens than specified due to the nature
# of subword tokenization. We specify slack to prevent an error.
_SLACK_LENGTH = 4


@app.post("/generations", response_model=Response)
async def create_generation(request: GenerationRequest):
    """Generate a readme from a repository"""
    if request.model not in larch.generator.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=503,
            detail=f'Model "{request.model}" is currently not available.')

    generator = larch.generator.AVAILABLE_MODELS[request.model]

    logger.info('Generating context')
    context = _context_creator.create_context(
        request.files,
        use_prompt=not generator.is_pretrained,
        tokenizer=generator.tokenizer,
        max_tokens=generator.max_context_length - _MAX_GENERATION_LENGTH - _SLACK_LENGTH,
        project_name=request.project_name)
    logger.info('Generating readme')
    readme, difference = generator.generate(
        context, request.prompt, max_length=_MAX_GENERATION_LENGTH)
    logger.info('Postprocessing readme')
    readme, difference = postprocess(readme, difference)
    response = Response(
        id='tmp-non-id',
        model=request.model,
        choices=[
            Generation(
                text=readme,
                edits=[
                    Edit(
                        type='insertion',
                        start=len(readme) - len(difference),
                        text=difference
                    )
                ],
                index=0,
                logprobs=0.0
            )
        ]
    )
    return response


@app.get("/models", response_model=GenerationModels)
async def get_models():
    """List available models"""
    models = []
    for model_name in larch.generator.AVAILABLE_MODELS.keys():
        models.append(GenerationModel(
            id=model_name,
            description='',
            owned_by=''
        ))
    return GenerationModels(
        data=models
    )


@app.get("/health")
async def get_health():
    return


@app.on_event("startup")
async def startup_event():
    _startup_event()


def _startup_event():
    global _context_creator
    logger.info('Initializaing generator modules.')
    if 'OPENAI_API_KEY' not in os.environ:
        logger.warn(
            'Environment variable "OPENAI_API_KEY" is not set.'
            f'OpenAI-based models are disabled.')
    if 'LOADED_MODELS' not in os.environ:
        logger.info('Environment variable "LOADED_MODELS" is not set.'
                    f'All available models are loaded.')
        loaded_models = None
    else:
        loaded_models = [
            m for m in os.environ['LOADED_MODELS'].strip().split(',') if len(m) > 0]
    if 'ENCODER_DECODER_MODEL_PATHS' in os.environ:
        try:
            encoder_decoder_model_paths = json.loads(os.environ['ENCODER_DECODER_MODEL_PATHS'])
        except json.JSONDecodeError:
            raise OSError(
                'Invalid environmental variable was set for ENCODER_DECODER_MODEL_PATHS '
                f'({os.environ["LOADED_MODELS"]}). It must be a json-parsable '
                'mapping from model name to model path.')
    else:
        encoder_decoder_model_paths = dict()
    larch.generator.init_models(
        loaded_models, os.environ.get('OPENAI_API_KEY'), encoder_decoder_model_paths
    )
    logger.info(f'Loaded {list(larch.generator.AVAILABLE_MODELS.keys())}')
    logger.info('Initializing context creator.')
    _context_creator = ContextCreatorName('entrypoint').get_module()
    if 'ENTRYPOINT_EXTRACTOR' in os.environ:
        if not os.path.exists(os.environ['ENTRYPOINT_EXTRACTOR']):
            raise OSError(
                'The entrypoint extractor model path "'
                f'{os.environ["ENTRYPOINT_EXTRACTOR"]}" (set by '
                'ENTRYPOINT_EXTRACTOR environmental variable) does not exist.')
        entrypoint_model = os.environ['ENTRYPOINT_EXTRACTOR']
    else:
        entrypoint_model = cached_model_download()
    _context_creator.init(entrypoint_model)

    logger.info(f'Loaded {_context_creator}')


@click.command()
@click.option('--reload', is_flag=True)
@click.option('--log-level', type=str, default='info')
@click.option('--workers', type=int, default=1)
@click.option('--port', type=int, default=8000)
@click.option('--host', type=str, default='0.0.0.0')
def cli(reload: bool, log_level: str, workers: int, port: int, host: str):
    """ Start the larch server.
    You can pass larch config via environemental variables.

      - OPENAI_API_KEY: OpenAI API key to use for GPT-3 models.
      - LOADED_MODELS: A comma-separated list of models to load. All models are
                       loaded if not set.
      - ENCODER_DECODER_MODEL_PATHS: JSON-serializable string of mapping from
                                     a model name to a pretrained model path.
                                     You do not need to specify these model names
                                     in LOADED_MODELS (they will always be loaded).

      - ENTRYPOINT_EXTRACTOR: A path to an entrypoint extractor model
    """
    uvicorn.run(
        'larch.server:app', host=host, port=port, reload=reload, log_level=log_level, workers=workers)


@click.command()
def init_dryrun():
    """Dry-run initialization in order to load huggingface models.
    This is intended to be used when building a docker image.
    """
    _startup_event()


# For debugging & development
if __name__ == '__main__':
    cli()
