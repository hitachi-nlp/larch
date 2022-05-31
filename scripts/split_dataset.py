import datetime
import json
import random

import click


@click.command()
@click.option(
    "--date", type=click.DateTime(), default=datetime.datetime(2020, 6, 12),
    help='Make any repo created after this date to be the test split. '
         'The default is the date of GPT-3 publication.')
@click.option("--dev-ratio", type=float, default=0.3)
@click.option("--random-seed", type=int, default=123)
@click.argument(
    'input', type=click.Path(exists=True, file_okay=True, readable=True))
@click.argument('train', type=click.Path(exists=False))
@click.argument('dev', type=click.Path(exists=False))
@click.argument('test', type=click.Path(exists=False))
def main(date: datetime.datetime, dev_ratio: float, random_seed: int, input: str, train: str, dev: str, test: str):
    train_data = set()
    dev_test_data = []
    with open(input) as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            if len(line) == 0:
                continue
            creation_date = datetime.datetime.fromisoformat(
                json.loads(line)['meta']['created_at'].rstrip('Z'))
            if creation_date > date:
                dev_test_data.append(i)
            else:
                train_data.add(i)

    if len(dev_test_data) < 2:
        raise RuntimeError(
            'There must be at least two data after --date to split them into '
            f'test and dev, but only {len(dev_test_data)} was found.')

    dev_test_data = sorted(dev_test_data)
    random.seed(random_seed)
    random.shuffle(dev_test_data)
    n_dev = int(len(dev_test_data) * dev_ratio + 1)
    dev_data = set(dev_test_data[:n_dev])
    test_data = set(dev_test_data[n_dev:])

    n_train, n_dev, n_test = 0, 0, 0
    with open(input) as fin, open(train, 'w') as fout_train, open(dev, 'w') as fout_dev, open(test, 'w') as fout_test:
        for i, line in enumerate(fin):
            line = line.strip()
            if i in train_data:
                fout_train.write(line + '\n')
                n_train += 1
            elif i in dev_data:
                fout_dev.write(line + '\n')
                n_dev += 1
            elif i in test_data:
                fout_test.write(line + '\n')
                n_test += 1

    click.echo(f'train: {n_train}, dev: {n_dev}, test: {n_test}')


if __name__ == '__main__':
    main()
