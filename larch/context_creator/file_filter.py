import os
import re


_IGNORE_RULES = [
    r'\.py[cod]$',
    r'\.so$',
    r'/\.',
    r'/__pycache__/',
    r'/env/',
    r'/venv/',
    r'/build/',
    r'/develop-eggs/',
    r'/dist/',
    r'/eggs/',
    r'/lib/',
    r'/lib64/',
    r'/sdist/',
    r'/var/',
    r'/htmlcov/',  # Tox coverage report
    r'\.coverage$',
    r'\.coverage\.[^/]+$',
    r'\.egg-info$',
    r'\.egg$',
    r'\.spec$',
    r'\.manifest$',
    r'/coverage.xml$',
    r'/nosetests.xml$',
    r'/docs/_build/',  # Sphinx
    r'\.org$',
    r'~$',
    r'/#',
    r'\.spyderproject$',
    r'/tmp/',
]


IGNORE_RE = re.compile(
    '|'.join((f'(?:{r})' for r in _IGNORE_RULES)), flags=re.IGNORECASE
)


# approx 100 kbytes
_FILE_SIZE_THRESH = 100000

def is_size_acceptable(path):
    return os.path.getsize(path) < _FILE_SIZE_THRESH


# Addopted from https://stackoverflow.com/a/7392391 under CC BY-SA 3.0
_TEXTCHARS = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})

def is_binary_string(bytes):
    return bool(bytes.translate(None, _TEXTCHARS))


def is_file_binary(path):
    with open(path, 'rb') as fin:
        return is_binary_string(fin.read(1024))
