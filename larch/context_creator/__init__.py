from larch.utils import Enum
from . import all_data
from . import dummy
from . import entrypoint
from . import file_names
from . import random_file
from .model import Directory, File


class ContextCreatorName(Enum):
    all_data = 'all_data'
    dummy = 'dummy'
    entrypoint = 'entrypoint'
    random_file = 'random_file'
    file_names = 'file_names'

    def get_module(self):
        return MODULES[self]


MODULES = {
    ContextCreatorName.all_data: all_data,
    ContextCreatorName.entrypoint: entrypoint,
    ContextCreatorName.dummy: dummy,
    ContextCreatorName.random_file: random_file,
    ContextCreatorName.file_names: file_names
}
