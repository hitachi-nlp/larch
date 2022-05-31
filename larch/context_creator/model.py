import glob
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple

from pydantic import BaseModel

from .file_filter import IGNORE_RE, is_size_acceptable, is_file_binary


class File(BaseModel):
    name: str
    excluded: bool
    content: Optional[str]

    @classmethod
    def from_path(cls, path):
        # this expects root path in OS-dependent notation
        excluded = (not is_size_acceptable(path)) or is_file_binary(path)
        if excluded:
            content = None
        else:
            with open(path, errors='ignore') as fin:
                content = fin.read()
        return cls(
            name=os.path.split(path)[1],
            excluded=excluded,
            content=content
        )


class Directory(BaseModel):
    name: str
    children: List[Union[File, 'Directory']]

    class Config:
        schema_extra = {
            "example": {
                "name": "root",
                "children": [
                    {
                        'name': 'file.txt',
                        'excluded': False,
                        'content': 'File content comes here...'
                    },
                    {
                        'name': 'empty_directory',
                        'children': []
                    }
                ]
            }
        }

    @classmethod
    def from_directory(cls, root: str) -> 'Directory':
        # this expects root path in OS-dependent notation
        children = []
        root = os.path.realpath(root)
        # FIXME: It might be faster if we used os.walk?
        for path in glob.glob(os.path.join(glob.escape(root), '*')):
            # convert to posix_path as file_filter.py expects POSIX paths
            # we also utilize relative path to root to avoid matching to
            # user dir (like /tmp/)
            posix_path = './' + Path(os.path.relpath(path, start=root)).as_posix()
            if os.path.isdir(path):
                posix_path += '/'
            if IGNORE_RE.search(posix_path) is not None:
                continue
            if os.path.isfile(path):
                children.append(File.from_path(path))
            elif not os.path.islink(path):
                # Do not follow symlink, it will simply be ignored
                children.append(cls.from_directory(path))
        return Directory(
            name=os.path.split(root)[1],
            children=children
        )


def remove_readme(directory: Directory) -> Tuple[Directory, Optional[File]]:
    """ Remove readme file from directory and returns the new directory and
    readme files. This modifies directory in place so you should pass a copy
    if you intend to keep the original input intact.

    There could be multiple files that can potentially be readme documents
    like README.md.old, README_developers.md, etc.
    We will remove all these files and pick that one that most likely be
    the actual readme (i.e., the file with shortest name).
    """
    candidate_index = set()
    for i, file in enumerate(directory.children):
        if isinstance(file, Directory):
            continue
        if 'readme' in file.name.lower():
            candidate_index.add(i)
    if len(candidate_index) == 0:
        readme = None
    else:
        readme_idx = min(candidate_index,
                         key=lambda i: len(directory.children[i].name))
        readme = directory.children[readme_idx]
    directory.children = [
        directory.children[i] for i in range(len(directory.children))
        if i not in candidate_index
    ]
    return directory, readme


def remove_setuppy(directory: Directory) -> Tuple[Directory, Optional[File]]:
    """ Remove setup.py """
    setup_file = None
    children = []
    for file in directory.children:
        if isinstance(file, File) and file.name == 'setup.py':
            # it cannot have same name
            assert setup_file is None
            setup_file = file
        else:
            children.append(file)
    if setup_file is not None:
        directory.children = children
    return directory, setup_file
