import pickle
import os
import re

from . import symbols

from .english_utils.abbreviations import expand_abbreviations
from .english_utils.time_norm import expand_time_english
from .english_utils.number_norm import normalize_numbers

try:
    from transformers import AutoTokenizer
except Exception:  # optional for nobert / OpenEPD path
    AutoTokenizer = None  # type: ignore

# Slim OOV neural predictor (numpy + checkpoint20.npz). Avoids g2p_en/nltk.
try:
    from .slim_g2p_oov import predict_oov as _predict_oov
except Exception:  # pragma: no cover - vendored next to english.py
    from slim_g2p_oov import predict_oov as _predict_oov  # type: ignore


def distribute_phone(n_phone, n_word):
    """Local copy of Melo's helper; avoid importing japanese.py (MeCab)."""
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
# OpenEPD-built Melo eng_dict (pickle). Set MELO_OPENEPD_DICT or place
# openepd_eng_dict.pickle next to this file.
OPENEPD_DICT_PATH = os.environ.get(
    "MELO_OPENEPD_DICT",
    os.path.join(current_file_path, "openepd_eng_dict.pickle"),
)


def _g2p_oov(word: str):
    """OOV fallback: slim g2p_en neural predictor (no NLTK/CMUdict)."""
    return _predict_oov(word)


arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def _resolve_openepd_path() -> str:
    """Prefer compact .oedb (mmap) over pickle when both exist."""
    env = os.environ.get("MELO_OPENEPD_DICT", "").strip()
    cands = []
    if env:
        cands.append(env)
        if env.endswith(".pickle"):
            cands.insert(0, env[: -len(".pickle")] + ".oedb")
    cands.extend(
        [
            os.path.join(current_file_path, "openepd_eng_dict.oedb"),
            "/models/melo-openepd-g2p-assets/openepd_eng_dict.oedb",
            OPENEPD_DICT_PATH,
            os.path.join(current_file_path, "openepd_eng_dict.pickle"),
            "/models/melo-openepd-g2p-assets/openepd_eng_dict.pickle",
        ]
    )
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return OPENEPD_DICT_PATH


def _ensure_local_oedb(pickle_path: str) -> str | None:
    """One-time pickle→.oedb next to downloaded assets (no JuiceFS upload)."""
    if not pickle_path.endswith(".pickle") or not os.path.isfile(pickle_path):
        return None
    oedb_path = pickle_path[: -len(".pickle")] + ".oedb"
    if os.path.isfile(oedb_path) and os.path.getsize(oedb_path) > 0:
        return oedb_path
    try:
        try:
            from .openepd_compact import build_compact
        except Exception:  # pragma: no cover
            from openepd_compact import build_compact  # type: ignore

        meta = build_compact(pickle_path, oedb_path)
        print("[melo english] built local OpenEPD compact:", meta)
        return oedb_path
    except Exception as e:
        print("[melo english] compact build skipped:", e)
        return None


def get_dict():
    # Prefer OpenEPD (compact mmap .oedb, else pickle).
    # If only pickle exists (already on JuiceFS/download), build .oedb locally once.
    path = _resolve_openepd_path()
    if path.endswith(".pickle"):
        built = _ensure_local_oedb(path)
        if built:
            path = built
    if os.path.exists(path):
        if path.endswith(".oedb") or path.endswith(".bin"):
            try:
                from .openepd_compact import CompactEngDict
            except Exception:  # pragma: no cover
                from openepd_compact import CompactEngDict  # type: ignore

            g2p_dict = CompactEngDict(path)
            print(
                "[melo english] using OpenEPD compact:",
                path,
                "size=",
                len(g2p_dict),
            )
            return g2p_dict
        with open(path, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
        print(
            "[melo english] using OpenEPD eng_dict:",
            path,
            "size=",
            len(g2p_dict),
        )
        return g2p_dict

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)
    print("[melo english] using CMU eng_dict size=", len(g2p_dict))
    return g2p_dict


eng_dict = get_dict()


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def text_normalize(text):
    text = text.lower()
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text

model_id = 'bert-base-uncased'
_tokenizer = None


def _get_tokenizer():
    """BERT wordpiece tokenizer — only needed for bert features.
    Nobert / offline: skip by default (MELO_SKIP_HF_TOKENIZER=1)."""
    global _tokenizer
    if _tokenizer is False:
        return None
    if _tokenizer is None:
        # Default skip: offline boxes hang retrying huggingface.co
        if os.environ.get("MELO_SKIP_HF_TOKENIZER", "1") == "1":
            _tokenizer = False
            return None
        tok_path = os.environ.get("MELO_EN_TOKENIZER", model_id)
        try:
            _tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        except Exception as e:
            print("[melo english] tokenizer unavailable, fallback whitespace split:", e)
            _tokenizer = False
            return None
    return _tokenizer


def _simple_tokenize(text):
    # Word-level tokens (no BERT wordpieces). Good enough for dict lookup.
    out = []
    for w in re.findall(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]", text):
        out.append(w)
    return out


def g2p_old(text):
    tokenized = _get_tokenizer().tokenize(text)
    # import pdb; pdb.set_trace()
    phones = []
    tones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p_oov(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    # todo: implement word2ph
    word2ph = [1 for i in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tok = _get_tokenizer()
        if tok is None:
            tokenized = _simple_tokenize(text)
        else:
            tokenized = tok.tokenize(text)
    # import pdb; pdb.set_trace()
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
            phone_len += len(phns)
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p_oov(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
                phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = [post_replace_ph(i) for i in phones]

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from text import english_bert

    return english_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    from text.english_bert import get_bert_feature
    text = "In this paper, we propose 1 DSPGAN, a N-F-T GAN-based universal vocoder."
    text = text_normalize(text)
    phones, tones, word2ph = g2p(text)
    import pdb; pdb.set_trace()
    bert = get_bert_feature(text, word2ph)
    
    print(phones, tones, word2ph, bert.shape)

    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
