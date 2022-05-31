import glob
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import click
import tqdm

# FIXME: it is importing items from different modules. Move them to a "common"
# module for better visibility
from larch.context_creator.model import Directory, remove_readme, File, remove_setuppy
from larch.entrypoint_extractor.features.common import extract_python_files, aggregate_files
from larch.utils.language_detection import detect_language
from larch.utils.markdown import extract_raw_text


def create_data(directory: Directory, readme: File, setuppy: Optional[File], removed_files: List[str], meta: dict):
    return {
        'repo': directory.dict(),
        'references': [readme.dict()],
        'setuppy': None if setuppy is None else setuppy.dict(),
        'removed_files': removed_files,
        'meta': meta
    }


def check_if_readme_is_valid(
        user: str, repo_name: str, readme: Optional[File], min_readme_size: int) -> bool:
    if readme is None:
        click.echo(f'No readme found for {user}/{repo_name}. Skipping...', err=True)
        return False
    if readme.excluded:
        click.echo(f'Readme excluded for {user}/{repo_name}. Skipping...', err=True)
        return False
    if len(readme.content) < min_readme_size:
        click.echo(
            f'Too short readme ({len(readme.content)}) for {user}/{repo_name}. '
            'Skipping...', err=True)
        return False
    if os.path.splitext(readme.name)[1].lower() != '.md':
        click.echo(
            f'Readme isn\'t markdown ({readme.name}) for {user}/{repo_name}. '
            'Skipping...', err=True)
        return False
    readme_text = extract_raw_text(readme.content, max_blocks=10)
    lang = detect_language(readme_text[:1000])
    if lang != 'en':
        click.echo(f'Readme isn\'t English ({lang}) for {user}/{repo_name}. '
                   'Skipping...', err=True)
        return False
    return True


def check_if_files_are_valid(
        user: str, repo_name: str, directory: Directory, max_num_files: int) -> bool:
    files = aggregate_files(directory)
    if len(files) > max_num_files:
        click.echo(
            f'Too many files ({len(files)}) for {user}/{repo_name}. Skipping...',
            err=True)
        return False
    if len(files) == 0:
        click.echo(
            f'No file found for {user}/{repo_name}. Skipping...', err=True)
        return False
    python_files = extract_python_files(files)
    if len(python_files) == 0:
        click.echo(
            f'Python file does not exist for {user}/{repo_name}. Skipping...',
            err=True)
        return False
    return True


@click.command()
@click.option('--min-readme-size', type=int, default=500)
@click.option('--max-num-files', type=int, default=1000)
@click.option('--temp-dir', type=click.Path(exists=True, file_okay=False), default=None)
@click.option('--resume', is_flag=True)
@click.argument('meta-root', type=click.Path(exists=True, file_okay=False))
@click.argument('repo-root', type=click.Path(exists=True, file_okay=False))
@click.argument('out', type=click.Path())
def main(min_readme_size: int, max_num_files: int, temp_dir: Optional[str],
         resume: bool, meta_root: str, repo_root: str, out: str):
    repo_zip_paths = sorted(glob.glob(os.path.join(glob.escape(repo_root), '*', '*', 'repo.zip')))

    # we apply deterministic shuffle here because we don't apply intra-split
    # shuffle in split_dataset.py. This ensures that, if we sample first n data
    # from any of the split, we are actually taking random repos.
    random.seed(42)
    random.shuffle(repo_zip_paths)

    initial_idx = 0
    num_all_repos = len(repo_zip_paths)
    if resume and os.path.exists(out):
        with open(out) as fin:
            for line in fin:
                if len(line.strip()) > 0:
                    initial_idx += 1
        repo_zip_paths = repo_zip_paths[initial_idx:]
    elif os.path.exists(out):
        raise click.exceptions.BadArgumentUsage(
            f'Output path "{out}" must not exist unless --resume is set.')
    with open(out, 'a') as fout:
        # this is (probably) an IO bound operation so do not parallelize
        for repo_zip_path in tqdm.tqdm(repo_zip_paths, initial=initial_idx, total=num_all_repos):
            user, repo_name = Path(repo_zip_path).parts[-3:-1]
            meta_path = os.path.join(meta_root, 'results', user, repo_name, 'meta.json')
            if not os.path.exists(meta_path):
                click.echo(f'Meta file "{meta_path}" cannot be found. Skipping...', err=True)
                continue
            with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdirname:
                try:
                    with zipfile.ZipFile(repo_zip_path) as zip_file:
                        zip_file.extractall(tmpdirname)
                        dir_tree = Directory.from_directory(os.path.join(tmpdirname, 'repo'))
                except zipfile.BadZipFile as e:
                    click.echo(f'Loading repo zip "{repo_zip_path}" failed: {e}')
            dir_tree, setuppy = remove_setuppy(dir_tree)
            _, readme = remove_readme(dir_tree)
            if not check_if_files_are_valid(
                    user, repo_name, dir_tree, max_num_files):
                continue
            if not check_if_readme_is_valid(user, repo_name, readme, min_readme_size):
                continue
            with open(meta_path) as fin:
                meta = json.load(fin)
            with open(os.path.join(repo_root, user, repo_name, 'removed_files.json')) as fin:
                removed_files = json.load(fin)
            if setuppy is not None:
                removed_files.append('setup.py')
            data = create_data(dir_tree, readme, setuppy, removed_files, meta)
            fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()
