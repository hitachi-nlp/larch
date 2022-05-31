import os
from typing import Optional

import click
import requests

import larch
from larch.context_creator import ContextCreatorName, Directory
from larch.postprocessor import postprocess
from larch.request_models import GenerationRequest
from larch.utils import spinner
from larch.utils.download import cached_model_download, larch_model_url


def setup_local_generator(
        endpoint: Optional[str],
        model: Optional[str],
        encoder_decoder_path: Optional[str],
        openai_api_key: Optional[str]) -> larch.generator.BaseGenerator:
    if endpoint is not None:
        raise click.BadParameter(
            'You may not specify --endpoint when --local is set.')
    if (model is None) == (encoder_decoder_path is None):
        raise click.BadParameter(
            'You must specify one of --model or --encoder-decoder-path'
        )
    with spinner('Loading a language model...'):
        larch.generator.init_models(
            [] if model is None else [model],
            openai_api_key,
            {} if encoder_decoder_path is None else {'encoder_decoder': encoder_decoder_path}
        )
    assert len(larch.generator.AVAILABLE_MODELS) == 1
    return list(larch.generator.AVAILABLE_MODELS.values())[0]


# FIXME: Stop hardcoding this
_MAX_GENERATION_LENGTH = 200
# context creators may return more tokens than specified due to the nature
# of subword tokenization. We specify slack to prevent an error.
_SLACK_LENGTH = 4


@click.command()
@click.option('--out', '-o', default='README.md', type=click.Path())
@click.option(
    '--input', '-i', default='./',
    type=click.Path(exists=True, file_okay=False, readable=True))
@click.option(
    '--input-context', default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Utilize a text file for README context instead of interactive creating one.')
@click.option(
    '--endpoint', '-e', default=None, type=str)
@click.option('--local/--no-local', default=False)
@click.option('--model', '-m', type=str, default=None)
@click.option('--encoder-decoder-path', type=str, default=None)
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', type=str,
              default=None)
@click.option('--entrypoint-model', default=None, type=str,
              help='Optional model path for entry point extractor model. '
                   f'Defaults to {larch_model_url}')
def cli(out: str, input: str, input_context: Optional[str],
        endpoint: Optional[str], local: bool, model: Optional[str],
        encoder_decoder_path: Optional[str], openai_api_key: Optional[str],
        entrypoint_model: Optional[str]):
    """Generate a readme from a repository"""
    if local:
        generator = setup_local_generator(
            endpoint, model, encoder_decoder_path, openai_api_key)
        context_creator = ContextCreatorName('entrypoint').get_module()
        if entrypoint_model is None:
            entrypoint_model = cached_model_download()
        context_creator.init(entrypoint_model)
    else:
        if encoder_decoder_path is not None:
            raise click.BadParameter(
                'You may not specify --encoder-decoder-path when --local is not set.')
        if endpoint is None:
            raise click.BadParameter(
                'You must specify --endpoint when --local is not set.')
        if openai_api_key is not None:
            raise click.BadParameter(
                'You may not specify --openai-api-key when --local '
                'is not set.')
        if entrypoint_model is not None:
            raise click.BadParameter(
                'You may not specify --entrypoint-model when --local '
                'is not set.')
        endpoint = endpoint.rstrip('/') + '/'
        response = requests.get(endpoint + 'health')
        if response.status_code != 200:
            raise RuntimeError(
                f'Endpoint "{endpoint}" is either not reachable or dead. Error: {response}')

        response = requests.get(endpoint + 'models')
        if response.status_code != 200:
            raise RuntimeError(
                f'Something went wrong with the server. Error: {response}')
        model_list = {m['id'] for m in response.json()['data']}
        if model not in model_list:
            raise click.BadParameter(
                f'You must choose a model from [{", ".join(model_list)}] at {endpoint}.')

    project_name = os.path.split(os.path.realpath(input))[-1]
    project_name = click.prompt('Project name:', type=str, default=project_name)

    with spinner(f'Collecting files from {input}'):
        dir_tree = Directory.from_directory(input)

    if local:
        with spinner(f'Creating context from files from {input}'):
            context = context_creator.create_context(
                dir_tree,
                use_prompt=not generator.is_pretrained,
                tokenizer=generator.tokenizer,
                max_tokens=generator.max_context_length - _MAX_GENERATION_LENGTH - _SLACK_LENGTH,
                project_name=project_name)

    click.echo(
'''larch will launch your editor to write a readme document.
You can write a partial readme and AI will try to complete it.
You will have chance to quit writing the readme each time after editing.''')
    click.prompt('Press enter to continue.', hide_input=True, default=1,
                 prompt_suffix='', show_default=False)

    def _generate(input_text: str) -> str:
        if local:
            gen_text, diff = generator.generate(
                context, input_text, max_length=_MAX_GENERATION_LENGTH)
            gen_text, _ = postprocess(gen_text, diff)
            return gen_text
        else:
            request_data = GenerationRequest(
                files=dir_tree,
                model=model,
                prompt=input_text,
                project_name=project_name
            )
            response = requests.post(
                endpoint + 'generations', json=request_data.dict())
            if response.status_code != 200:
                raise RuntimeError(
                    f'Something went wrong with the server. Error: {response}')
            return response.json()['choices'][0]['text']

    if input_context is None:
        cur_readme = '# README\n\n'
        while True:
            input_text = click.edit(cur_readme)
            if input_text is None:
                input_text = cur_readme
            if click.confirm('Finish editing?', default=None):
                break
            with spinner('Generating readme...'):
                cur_readme = _generate(input_text)
        if cur_readme == {'# README\n\n', ''}:
            click.echo('An empty or unchaged readme was given. Quitting without saving...')
            return
    else:
        with open(input_context) as fin:
            input_text = fin.read()
        with spinner('Generating readme...'):
            cur_readme = _generate(input_text)

    with spinner(f'Writing generated README to {out}...'):
        with open(out, 'w') as fout:
            fout.write(cur_readme)


# For debugging & development
if __name__ == '__main__':
    cli()
