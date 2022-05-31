import datetime
import json
import time
from typing import Optional

import click
import numpy as np
import tqdm

from larch.cli import setup_local_generator
from larch.context_creator import ContextCreatorName, Directory, File
from larch.context_creator.model import remove_readme, remove_setuppy


@click.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, file_okay=True, readable=True))
@click.option('--model', '-m', type=str, default=None)
@click.option('--encoder-decoder-path', type=str, default=None)
@click.option('--max-context-length', type=int, default=500)
@click.option('--num-eval-tokens', '-n', type=int, default=512)
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', type=str,
              default=None)
@click.option('--out', '-o', default=None, type=click.Path(exists=False))
@click.option('--entrypoint-model', default=None, type=click.Path(exists=True, dir_okay=False))
def cli(input: str, model: str, encoder_decoder_path: Optional[str],
        max_context_length: int, num_eval_tokens: int,
        openai_api_key: Optional[str], out: Optional[str],
        entrypoint_model: Optional[str]):
    generator = setup_local_generator(
        None, model, encoder_decoder_path, openai_api_key)

    if entrypoint_model is not None:
        ContextCreatorName('entrypoint').get_module().init(entrypoint_model)

    perplexities = {}
    with open(input) as fin:
        for line in tqdm.tqdm(fin):
            data = json.loads(line.strip())
            reference = File.parse_obj(data['references'][0])
            dir_tree = Directory.parse_obj(data['repo'])
            dir_tree, _ = remove_readme(dir_tree)
            dir_tree, _ = remove_setuppy(dir_tree)
            if entrypoint_model is not None:
                context = ContextCreatorName('entrypoint').get_module().create_context(
                    dir_tree,
                    use_prompt=not generator.is_pretrained,
                    tokenizer=generator.tokenizer,
                    max_tokens=max_context_length,
                    project_name=data['meta']['name'])
            else:
                context = ContextCreatorName('random_file').get_module().create_context(
                    dir_tree,
                    use_prompt=not generator.is_pretrained,
                    tokenizer=generator.tokenizer,
                    max_tokens=max_context_length,
                    project_name=data['meta']['name'])
            perplexity = generator.calculate_perplexity(
                context, reference.content, num_eval_tokens)
            data_id = data['meta']['full_name']
            assert data_id not in perplexities
            perplexities[data_id] = perplexity
            time.sleep(3)
    ppl_ave = float(np.average(list(perplexities.values())))
    ppl_std = float(np.std(list(perplexities.values())))
    click.echo(f'average: {ppl_ave}, std: {ppl_std}')
    if out is not None:
        metrics = {
            'perplexity': {
                'average': ppl_ave,
                'std': ppl_std,
                'values': perplexities
            },
            'experiment_info': {
                'model': model,
                'num_eval_tokens': num_eval_tokens,
                'max_context_length': max_context_length,
                'utc_date': datetime.datetime.utcnow().isoformat()
            }
        }
        with open(out, 'w') as fout:
            json.dump(metrics, fout, indent=2)
        click.echo(f'Written metrics to "{out}"')


if __name__ == '__main__':
    cli()
