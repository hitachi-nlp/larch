from contextlib import contextmanager

from yaspin import yaspin


@contextmanager
def spinner(text: str):
    with yaspin(text=text, color="yellow") as spinner:
        yield
        spinner.ok("✅ ")
