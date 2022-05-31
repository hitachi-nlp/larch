import html
import json
import os
import random
import time
from typing import Optional, List

import click
import spacy
import tqdm

from larch.cli import setup_local_generator
from larch.context_creator import ContextCreatorName, Directory, File
from larch.context_creator.model import remove_readme, remove_setuppy
from larch.evaluation import calc_all_metrics
from larch.postprocessor import postprocess

nlp = spacy.load("en_core_web_sm")


def truncate(reference: str, input_tokens: int):
    if input_tokens == 0:
        return ''
    doc = nlp(reference)
    if len(doc) < input_tokens:
        return reference
    last_char_idx = doc[input_tokens - 1].idx + len(doc[input_tokens - 1])
    return reference[:last_char_idx]


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--model', '-m', type=str, default=None)
@click.option('--encoder-decoder-path', type=str, default=None)
@click.option('--max-generation-length', type=int, default=512)
@click.option('--max-context-length', type=int, default=512)
@click.option('--input-tokens', type=int, default=0)
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', type=str,
              default=None)
@click.option('--out', '-o', default=None, type=click.Path(exists=False))
@click.option('--entrypoint-model', default=None, type=click.Path(exists=True, dir_okay=False))
def predict(input: str, model: Optional[str], encoder_decoder_path: Optional[str],
            max_generation_length: int, max_context_length: int, input_tokens: int,
            openai_api_key: Optional[str], out: Optional[str],
            entrypoint_model: Optional[str]):
    if not (isinstance(input_tokens, int) and input_tokens >= 0):
        raise click.BadOptionUsage(
            '--input-tokens', '--input-tokens must be a positive integer')
    click.echo(
        f'Making prediction on {input} with model={model}, '
        f'encoder_decoder_path={encoder_decoder_path}, '
        f'max_generation_length={max_generation_length}, '
        f'max_context_length={max_context_length}, '
        f'input_tokens={input_tokens}, '
        f'entrypoint_model={entrypoint_model}.')

    generator = setup_local_generator(
        None, model, encoder_decoder_path, openai_api_key)

    if entrypoint_model is not None:
        ContextCreatorName('entrypoint').get_module().init(entrypoint_model)

    with open(input) as fin, open(out, 'w') as fout:
        for line in tqdm.tqdm(fin):
            data = json.loads(line.strip())
            reference = File.parse_obj(data['references'][0])
            input_text = truncate(reference.content, input_tokens)
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
            gen_text, diff = generator.generate(
                context, input_text, max_length=max_generation_length)
            gen_text, _ = postprocess(gen_text, diff)
            result = {
                'input': data,
                'result': gen_text,
                'reference': reference.content,
                'input_text': input_text,
                'context': context
            }
            fout.write(json.dumps(result) + '\n')
            time.sleep(3)
    click.echo(f'Done prediction on {input} and written results to {out}.')


@cli.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--out', '-o', default=None, type=click.Path())
def evaluate(input: str, out: Optional[str]):
    click.echo(f'Running evaluation on {input}.')
    predictions = []
    references = []
    with open(input) as fin:
        for line in fin:
            data = json.loads(line)
            predictions.append(data['result'])
            references.append(data['reference'])
    metrics = calc_all_metrics(predictions, references)
    if out is not None:
        with open(out, 'w') as fout:
            json.dump(metrics, fout, indent=2)
    print(json.dumps(metrics, indent=2))
    click.echo(f'Done evaluation on {input} and written metrics to {out}.')


def _escape(text: str):
    return html.escape(text).replace('\n', '<br/>')


@cli.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--out', '-o', default=None, type=click.Path(exists=False))
def visualize(input: str, out: str):
    click.echo(f'Running visualization of {input}.')

    os.makedirs(out, exist_ok=True)
    with open(input) as fin:
        for line in fin:
            data = json.loads(line)
            name = data['input']['meta']['full_name']
            url = data['input']['meta']['html_url']
            description = data['input']['meta']['description']
            if description is None:
                description = ''
            input_text = data['input_text']
            result = data['result']
            assert result[:len(input_text)] == input_text
            result = result[len(input_text):]
            result_html = ''.join((
                f'<span style="color:#2B547E">{_escape(input_text)}</span>',
                _escape(result)
            ))
            formatted_html = ''.join((
                '<html><body>',
                '<h2>larch - generation result</h2>',
                '<hr>',
                f'<h4><a href="{url}">{name}</a></h4>',
                _escape(description),
                '<hr>',
                '<h4>Generated readme</h4>',
                result_html,
                '<hr>',
                '<h4>Reference</h4>',
                _escape(data['reference']),
                '<hr>',
                '<h4>Context</h4>',
                _escape(data['context']),
                '</body></html>'
            ))
            with open(os.path.join(out, f'{name.replace("/", ".")}.html'), 'w') as fout:
                fout.write(formatted_html)
    click.echo(f'Done visualization of {input} and written htmls to {out}.')


@cli.command('ab-test')
@click.option(
    '--input', '-i', 'inputs', required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--out', '-o', default=None, type=click.Path(exists=False))
def ab_test(inputs: List[str], out: str):
    click.echo(f'Running visualization of {inputs}.')

    fins = [open(input) for input in inputs]
    mappings = []

    os.makedirs(out, exist_ok=True)
    for i, lines in enumerate(zip(*fins)):
        all_data = [json.loads(line) for line in lines]
        if len(set([d['input']['meta']['full_name'] for d in all_data])) != 1:
            raise ValueError(f'Repo mismatch in {i}-th row of inputs')
        name = all_data[0]['input']['meta']['full_name']
        url = all_data[0]['input']['meta']['html_url']
        description = all_data[0]['input']['meta']['description']
        if description is None:
            description = ''
        indices = list(range(len(inputs)))
        random.shuffle(indices)
        results_html = []
        for j_ind, j in enumerate(indices):
            input_text = all_data[j]['input_text']
            result = all_data[j]['result']
            assert result[:len(input_text)] == input_text
            result = result[len(input_text):]
            results_html.append(''.join((
                f'<h4>Generated readme {j_ind}</h4>',
                f'<span style="color:#2B547E">{_escape(input_text)}</span>',
                _escape(result)
            )))
        formatted_html = ''.join((
            '<html><body>',
            '<h2>larch - generation result</h2>',
            '<hr>',
            f'<h4><a href="{url}">{name}</a></h4>',
            _escape(description),
            '<hr>',
            '<hr>'.join(results_html),
            '<hr>',
            '<h4>Reference</h4>',
            _escape(all_data[0]['reference']),
            '</body></html>'
        ))
        with open(os.path.join(out, f'{i:0>4}_{name.replace("/", ".")}.html'), 'w') as fout:
            fout.write(formatted_html)
        mappings.append({
            inputs[j]: j_ind
            for j_ind, j in enumerate(indices)
        })
    click.echo(f'Done visualization of {inputs} and written htmls to {out}.')
    mappings_path = os.path.join(out, 'mappings.json')
    with open(mappings_path, 'w') as fout:
        json.dump({'mappings': mappings}, fout, indent=2)
    click.echo(f'Written mappings to to {mappings_path}.')


if __name__ == '__main__':
    cli()
