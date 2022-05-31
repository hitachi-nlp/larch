import click
import xgboost as xgb

import larch


@click.group()
def main():
    pass


@main.command()
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '--output', '-o', required=True, type=click.Path(exists=False))
@click.option(
    '--seed', '-s', default=123, type=int)
def train(input: str, output: str, seed: int):
    dataset = larch.entrypoint_extractor.load_data(input)
    features = [
        file_features
        for repo_features in dataset.values()
        for file_features in repo_features.values()
    ]
    model = larch.entrypoint_extractor.train(features, seed=seed)
    model.save_model(output)


@main.command()
@click.option(
    '--model', '-m', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '--input',  '-i', required=True,
    type=click.Path(exists=True, file_okay=False, readable=True))
def predict(model: str, input: str):
    dir_tree = larch.context_creator.Directory.from_directory(input)
    path_to_features, _ = larch.entrypoint_extractor.load_single_data(
        'repo', dir_tree, None, None, None, timeout=120)
    if len(path_to_features) == 0:
        print('No candidate (python files) for entrypoint is found.')
        return
    model_ = xgb.XGBRanker()
    model_.load_model(model)
    best_path, _ = larch.entrypoint_extractor.predict(model_, path_to_features)
    print('/'.join(best_path))


@main.command('predict-batch')
@click.option(
    '--model', '-m', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '--input', '-i', required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True))
def predict_batch(model: str, input: str):
    model_ = xgb.XGBRanker()
    model_.load_model(model)
    dataset = larch.entrypoint_extractor.load_data(input)
    for repo_name, path_to_features in dataset.items():
        if len(path_to_features) == 0:
            print(f'{repo_name}: No candidate found')
            continue
        best_path, probs = larch.entrypoint_extractor.predict(model_, path_to_features)
        print(f'{repo_name}:', '/'.join(best_path))


if __name__ == '__main__':
    main()
