from typing import Tuple


def postprocess(readme_text: str, diff_text: str) -> Tuple[str, str]:
    return readme_text.rstrip(), diff_text.rstrip()
