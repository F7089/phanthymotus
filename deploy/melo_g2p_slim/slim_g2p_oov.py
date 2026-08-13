# -*- coding: utf-8 -*-
"""Slim g2p_en OOV neural predictor (numpy only).

Extracted from Kyubyong/g2p ``G2p.predict`` — no NLTK, no CMUdict, no POS.
Requires ``checkpoint20.npz`` next to this file (or MELO_G2P_OOV_CKPT).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

_GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
_PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]


def _find_checkpoint() -> str:
    env = os.environ.get("MELO_G2P_OOV_CKPT", "").strip()
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # Prefer vendored / data-disk paths (no g2p_en import).
    candidates = [
        os.path.join(here, "checkpoint20.npz"),
        "/models/melo-openepd-g2p-assets/vendor/melo_g2p/text/checkpoint20.npz",
        # Host data disk → usually bind-mounted into the container as /models
        "/data/fanyi/tts/g2p/checkpoint20.npz",
        os.path.expanduser("~/fanyi/tts/g2p/checkpoint20.npz"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    # Last resort: installed g2p_en package (dev / parity only)
    try:
        import g2p_en

        pkg = os.path.dirname(g2p_en.__file__)
        cand = os.path.join(pkg, "checkpoint20.npz")
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    raise FileNotFoundError(
        "checkpoint20.npz not found; place it at "
        "/data/fanyi/tts/g2p/checkpoint20.npz (or set MELO_G2P_OOV_CKPT)"
    )


class SlimG2pOov:
    """NumPy GRU seq2seq OOV predictor from g2p_en checkpoint20.npz."""

    def __init__(self, ckpt: Optional[str] = None):
        self.g2idx = {g: i for i, g in enumerate(_GRAPHEMES)}
        self.idx2p = {i: p for i, p in enumerate(_PHONEMES)}
        path = ckpt or _find_checkpoint()
        variables = np.load(path)
        self.enc_emb = variables["enc_emb"]
        self.enc_w_ih = variables["enc_w_ih"]
        self.enc_w_hh = variables["enc_w_hh"]
        self.enc_b_ih = variables["enc_b_ih"]
        self.enc_b_hh = variables["enc_b_hh"]
        self.dec_emb = variables["dec_emb"]
        self.dec_w_ih = variables["dec_w_ih"]
        self.dec_w_hh = variables["dec_w_hh"]
        self.dec_b_ih = variables["dec_b_ih"]
        self.dec_b_hh = variables["dec_b_hh"]
        self.fc_w = variables["fc_w"]
        self.fc_b = variables["fc_b"]

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh
        rz_ih, n_ih = (
            rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3],
            rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :],
        )
        rz_hh, n_hh = (
            rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3],
            rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :],
        )
        rz = self._sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)
        n = np.tanh(n_ih + r * n_hh)
        return (1 - z) * n + z * h

    def _gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self._grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
            outputs[:, t, :] = h
        return outputs

    def predict(self, word: str) -> List[str]:
        word = (word or "").lower()
        word = "".join(ch for ch in word if ch.isalpha())
        if not word:
            return []
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(ch, self.g2idx["<unk>"]) for ch in chars]
        enc = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        enc = self._gru(
            enc,
            len(word) + 1,
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32),
        )
        h = enc[:, -1, :]
        dec = np.take(self.dec_emb, [2], axis=0)  # <s>
        preds = []
        for _ in range(20):
            h = self._grucell(
                dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh
            )
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = int(logits.argmax())
            if pred == 3:  # </s>
                break
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)
        return [self.idx2p.get(i, "<unk>") for i in preds]


_PREDICTOR: Optional[SlimG2pOov] = None


def predict_oov(word: str) -> List[str]:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SlimG2pOov()
        print(
            "[melo english] slim g2p OOV predictor ready (no nltk/g2p_en import)"
        )
    return _PREDICTOR.predict(word)
