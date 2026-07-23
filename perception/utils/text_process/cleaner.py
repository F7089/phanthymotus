import importlib
import logging

from .langsegmenter import LangSegmenter

log = logging.getLogger(__name__)


def clean_text(text, language):
    language_module_map = {
        "zh": "chinese",
        "en": "english",
    }
    if language not in language_module_map:
        language = "zh"
    language_module = importlib.import_module(
        f".{language_module_map[language]}", package=__package__
    )
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    return norm_text

def text_preprocess(text: str, language: str):
    try:
        segments = LangSegmenter.getTexts(text, language)
    except Exception:
        # Segmenter failed mid-run — still normalize as a single language blob.
        return clean_text(text, language) or text

    textlist = []
    langlist = []
    for tmp in segments:
        if langlist:
            if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                textlist[-1] += tmp["text"]
                continue
        if tmp["lang"] == "en":
            langlist.append(tmp["lang"])
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            langlist.append(language)
        textlist.append(tmp["text"])

    norm_text_list = []
    for i in range(len(textlist)):
        try:
            tmp_norm_text = clean_text(textlist[i], langlist[i])
        except Exception as e:
            log.warning("[text_process] clean_text(%s) failed: %s", langlist[i], e)
            tmp_norm_text = textlist[i]
        if tmp_norm_text:
            norm_text_list.append(tmp_norm_text)
    norm_text = "".join(norm_text_list)
    return norm_text
