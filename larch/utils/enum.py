import enum


class _EnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class Enum(str, enum.Enum, metaclass=_EnumMeta):
    @classmethod
    def list(cls):
        return [e.value for e in cls]
