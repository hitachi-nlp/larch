from langdetect import detect, DetectorFactory, LangDetectException


def detect_language(text: str) -> str:
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'none'
    return lang
