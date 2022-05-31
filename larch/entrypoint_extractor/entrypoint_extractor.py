import itertools
import json
import random
from functools import partial
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from joblib import Parallel, delayed
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

from larch.context_creator.model import Directory, remove_readme, File, remove_setuppy
from larch.utils import tqdm_joblib, count_lines
from .features import (
    PathSpec,
    aggregate_files,
    extract_content_features, ContentFeatures,
    extract_filename_features, FilenameFeatures,
    extract_import_features, ImportFeatures,
    extract_inheritance_features, InheritanceFeatures,
    extract_oracle_features, OracleFeatures
)

_BLACKLIST_FILENAME = {
    '.github',
    '.circleci',
    'docs',
    'test',
    'tests',
    '.travis.yml',
    '.gitignore',
    'build',
    'dist',
    '_version.py',
    '__pycache__',
    'eggs',
    '.eggs',
    '.cov',
    'setup.py'
}


def is_blacklisted(path: PathSpec) -> bool:
    return len(set(path) & _BLACKLIST_FILENAME) > 0


class FileFeatures(object):
    def __init__(
            self,
            repo_name: str,
            path: PathSpec,
            content_features: Optional[ContentFeatures] = None,
            filename_features: Optional[FilenameFeatures] = None,
            import_features: Optional[ImportFeatures] = None,
            inheritance_features: Optional[InheritanceFeatures] = None,
            oracle_features: Optional[OracleFeatures] = None
    ):
        self.repo_name: str = repo_name
        self.path: PathSpec = path
        self.content_features: Optional[ContentFeatures] = content_features
        self.filename_features: Optional[FilenameFeatures] = filename_features
        self.import_features: Optional[ImportFeatures] = import_features
        self.inheritance_features: Optional[InheritanceFeatures] = inheritance_features
        self.oracle_features: Optional[OracleFeatures] = oracle_features

    @staticmethod
    def _map_pseudo_label_values(val: Optional[bool]) -> int:
        if val is None:
            return -1
        assert isinstance(val, bool)
        # true -> 1 (it is entrypoint), false -> 0 (it is not entrypoint)
        return int(val)

    def get_pseudo_label_array(self) -> List[int]:
        features = list(itertools.chain(
            ([None] * ContentFeatures.get_pseudo_label_dim()
             if self.content_features is None else
             self.content_features.get_pseudo_label_array()),
            ([None] * FilenameFeatures.get_pseudo_label_dim()
             if self.filename_features is None else
             self.filename_features.get_pseudo_label_array()),
            ([None] * ImportFeatures.get_pseudo_label_dim()
             if self.import_features is None else
             self.import_features.get_pseudo_label_array()),
            ([None] * InheritanceFeatures.get_pseudo_label_dim()
             if self.inheritance_features is None else
             self.inheritance_features.get_pseudo_label_array()),
            ([None] * OracleFeatures.get_pseudo_label_dim()
             if self.oracle_features is None else
             self.oracle_features.get_pseudo_label_array())
        ))
        features = list(map(self._map_pseudo_label_values, features))
        return features

    @staticmethod
    def get_pseudo_label_names() -> List[str]:
        return list(itertools.chain(
            (f'ContentFeatures.{name}' for name in ContentFeatures.get_pseudo_label_names()),
            (f'FilenameFeatures.{name}' for name in FilenameFeatures.get_pseudo_label_names()),
            (f'ImportFeatures.{name}' for name in ImportFeatures.get_pseudo_label_names()),
            (f'InheritanceFeatures.{name}' for name in InheritanceFeatures.get_pseudo_label_names()),
            (f'OracleFeatures.{name}' for name in OracleFeatures.get_pseudo_label_names())
        ))

    def get_feature_array(self) -> List[Union[int, float, bool]]:
        features = list(itertools.chain(
            ([None] * ContentFeatures.get_feature_dim()
             if self.content_features is None else
             self.content_features.get_feature_array()),
            ([None] * FilenameFeatures.get_feature_dim()
             if self.filename_features is None else
             self.filename_features.get_feature_array()),
            ([None] * ImportFeatures.get_feature_dim()
             if self.import_features is None else
             self.import_features.get_feature_array()),
            ([None] * InheritanceFeatures.get_feature_dim()
             if self.inheritance_features is None else
             self.inheritance_features.get_feature_array()),
            ([None] * OracleFeatures.get_feature_dim()
             if self.oracle_features is None else
             self.oracle_features.get_feature_array())
        ))
        features = [np.nan if v is None else v for v in features]
        return features

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(itertools.chain(
            (f'ContentFeatures.{name}' for name in ContentFeatures.get_feature_names()),
            (f'FilenameFeatures.{name}' for name in FilenameFeatures.get_feature_names()),
            (f'ImportFeatures.{name}' for name in ImportFeatures.get_feature_names()),
            (f'InheritanceFeatures.{name}' for name in InheritanceFeatures.get_feature_names()),
            (f'OracleFeatures.{name}' for name in OracleFeatures.get_feature_names()),
        ))

    def has_features(self) -> bool:
        return (
            self.content_features is not None or
            self.filename_features is not None or
            self.import_features is not None or
            self.inheritance_features is not None or
            self.oracle_features is not None
        )


def load_single_data(
        name: str,
        dir_tree: Directory,
        readme: Optional[File],
        setuppy: Optional[File],
        repo_name: Optional[str],
        timeout: Optional[int] = None) -> Tuple[Dict[PathSpec, FileFeatures], Dict[PathSpec, File]]:
    dir_tree, readme_ = remove_readme(dir_tree)
    if readme is None:
        readme = readme_
    files = aggregate_files(dir_tree)
    features = {path: FileFeatures(name, path)
                for path, f in files.items()}
    for path, feats in extract_content_features(files).items():
        features[path].content_features = feats
    for path, feats in extract_filename_features(files, repo_name).items():
        features[path].filename_features = feats
    for path, feats in extract_import_features(files, name, timeout).items():
        features[path].import_features = feats
    for path, feats in extract_inheritance_features(files).items():
        features[path].inheritance_features = feats
    for path, feats in extract_oracle_features(files, readme, setuppy).items():
        features[path].oracle_features = feats

    # we filter blacklisted path after the feature extraction because
    # import features utilize blacklisted files as well
    features = {
        path: feats for path, feats in features.items()
        if feats.has_features() and not is_blacklisted(path)}
    return features, files


def _load_single_data_from_jsonl(
        line: str,
        timeout: Optional[int] = None) -> Tuple[Optional[str], Dict[PathSpec, FileFeatures]]:
    data = json.loads(line.strip())
    dir_tree = Directory.parse_obj(data['repo'])
    dir_tree, _ = remove_readme(dir_tree)
    dir_tree, _ = remove_setuppy(dir_tree)
    readme = File.parse_obj(data['references'][0])
    setuppy = None if data['setuppy'] is None else File.parse_obj(data['setuppy'])
    path_to_features, _ = load_single_data(
        data['meta']['full_name'],
        dir_tree,
        readme,
        setuppy,
        data['meta']['name'],
        timeout=timeout)
    return data['meta']['full_name'], path_to_features


def load_data(path: str, n_jobs: Optional[int] = -2) -> Dict[str, Dict[PathSpec, FileFeatures]]:
    num_lines = count_lines(path)
    load_func = partial(_load_single_data_from_jsonl, timeout=300)
    with open(path) as fin:
        with tqdm_joblib(tqdm.tqdm(total=num_lines)):
            features = Parallel(n_jobs=n_jobs)(
                delayed(load_func)(line) for line in fin)
    return {name: path_to_features for name, path_to_features in features
            if len(path_to_features) > 0}


def _report_nonabstain_ratio(features: np.ndarray):
    label_names = FileFeatures.get_pseudo_label_names()
    df_lf_summary = LFAnalysis(features).lf_summary()
    df_lf_summary.insert(loc=0, column='name', value=label_names)
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None):
        print(df_lf_summary)

    assert features.shape[1] == len(label_names)
    nonabstain_ratios = [
        f'{name}={ratio:.1f}%'
        for name, ratio in zip(label_names, (features != -1).mean(axis=1) * 100)
    ]
    print(f'Non-abstain ratio:', ', '.join(nonabstain_ratios))


def _to_group_sizes(group_ids: List[str]) -> List[int]:
    i = 0
    group_sizes = []
    cur_group_id = group_ids[0]
    for group_id in group_ids:
        if group_id != cur_group_id:
            group_sizes.append(i)
            cur_group_id = group_id
            i = 1
        else:
            i += 1
    group_sizes.append(i)
    assert sum(group_sizes) == len(group_ids)
    return group_sizes


def train(features: List[FileFeatures], seed: Optional[int] = 123) -> xgb.XGBRanker:
    random.seed(seed)
    np.random.seed(seed)

    features = sorted(features, key=lambda f: f.repo_name)

    print(f'Training with {len(features)} files.')
    L_train = np.vstack([
        f.get_pseudo_label_array() for f in features
    ])
    _report_nonabstain_ratio(L_train)

    print(f'Fitting LabelModel with n_epochs=500')
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=seed)
    probs = label_model.predict_proba(L=L_train)

    model = xgb.XGBRanker(
        tree_method='hist',
        booster='gbtree',
        objective='rank:pairwise',
        random_state=seed
    )
    print('Fitting XGBoost')
    X_train = [f.get_feature_array() for f in features]
    groups = _to_group_sizes([f.repo_name for f in features])
    model.fit(X_train, probs[:, 1], group=groups, verbose=True)
    print('Done training XGBoost')
    return model


def predict(model: xgb.XGBRanker, path_to_features: Dict[PathSpec, FileFeatures]) -> Tuple[PathSpec, Dict[PathSpec, float]]:
    paths = list(path_to_features.keys())
    features = [path_to_features[p] for p in paths]
    X = [f.get_feature_array() for f in features]
    probs = model.predict(X)
    best_path = paths[np.argmax(probs)]
    probs_dict = {
        path: float(prob) for path, prob in zip(paths, probs)
    }
    return best_path, probs_dict
