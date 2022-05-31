import json
from typing import List, Dict, Any

import click
import joblib
import numpy as np
import tqdm
from transformers import GPT2TokenizerFast

from larch.context_creator import Directory, File
from larch.context_creator.common import extract_all_files
from larch.utils import tqdm_joblib, count_lines
from larch.context_creator.model import remove_readme, remove_setuppy
from larch.entrypoint_extractor.features.common import match_python_file


def agg_items(dicts: List[Dict[str, Any]], key: str):
    return [d[key] for d in dicts]


def summarize_statistics(dicts: List[Dict[str, Any]], key: str):
    values = agg_items(dicts, key)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'max': float(np.max(values)),
        'min': float(np.min(values))
    }


@click.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, file_okay=True, readable=True))
@click.option(
    '--output', '-o', default=None, type=click.Path(exists=False))
@click.option('--n-jobs', '-n', type=int, default=8)
def main(input: str, output: str, n_jobs: int):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def calc_statistics(line: str):
        data = json.loads(line.strip())
        reference = File.parse_obj(data['references'][0])
        n_reference_tokens = len(tokenizer.tokenize(reference.content, verbose=False))
        n_file_tokens = []
        n_excluded = 0
        dir_tree = Directory.parse_obj(data['repo'])
        dir_tree, _ = remove_readme(dir_tree)
        dir_tree, _ = remove_setuppy(dir_tree)
        n_files = 0
        for file in extract_all_files(dir_tree):
            if file.excluded or match_python_file(file.name) is None:
                n_excluded += 1
                continue
            n_files += 1
            n_file_tokens.append(len(tokenizer.tokenize(file.content, verbose=False)))
        return {
            'n_reference_tokens': n_reference_tokens,
            'n_file_tokens': n_file_tokens,
            'n_repo_tokens': sum(n_file_tokens),
            'n_excluded': n_excluded,
            'n_files': n_files,
            'name': data['meta']['full_name']
        }

    num_lines = count_lines(input)
    with open(input) as fin:
        with tqdm_joblib(tqdm.tqdm(total=num_lines)):
            statistics = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(calc_statistics)(line) for line in fin)

    summary = {
        'n_reference_tokens': summarize_statistics(statistics, 'n_reference_tokens'),
        'n_repo_tokens': summarize_statistics(statistics, 'n_repo_tokens'),
        'n_excluded': summarize_statistics(statistics, 'n_excluded'),
        'n_files': summarize_statistics(statistics, 'n_files')
    }
    print(json.dumps(summary, indent=2))
    if output is not None:
        with open(output, 'w') as fout:
            json.dump({
                'summary': summary,
                'full': statistics
            }, fout, indent=2)


if __name__ == '__main__':
    main()
