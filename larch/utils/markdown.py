from typing import Optional

import mistletoe
import mistletoe.ast_renderer


def extract_raw_text(markdown_text: str, max_blocks: Optional[int] = None) -> str:
    document = mistletoe.Document(markdown_text)
    ast = mistletoe.ast_renderer.get_ast(document)
    raw_texts = []
    for block in ast['children']:
        if block['type'] != 'Paragraph':
            continue
        for content in block['children']:
            if content['type'] == 'RawText':
                raw_texts.append(content['content'])
                if max_blocks is not None and len(raw_texts) == max_blocks:
                    break
        else:
            continue
        break
    return '\n'.join(raw_texts)
