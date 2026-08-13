# -*- coding: utf-8 -*-
"""Memory-compact OpenEPD English lexicon (mmap).

Avoids unpickling ~290k Python str / list objects. File stays on JuiceFS;
code stays in git. Lookup API matches Melo eng_dict: word -> list[list[str]].
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Iterator, List, Tuple

MAGIC = b"OEDB"
VERSION = 1
_HEADER = struct.Struct("<4sIII")  # magic, ver, n_words, n_phones


class CompactEngDict:
    """Read-only eng_dict backed by a single mmap'd .oedb file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, n_words, n_phones = _HEADER.unpack_from(self._mm, 0)
        if magic != MAGIC or ver != VERSION:
            raise ValueError(f"bad OpenEPD compact header: {self.path}")
        self._n = n_words
        off = _HEADER.size

        phone_lens = list(self._mm[off : off + n_phones])
        off += n_phones
        self._phones: List[str] = []
        for ln in phone_lens:
            self._phones.append(
                self._mm[off : off + ln].decode("ascii", errors="replace")
            )
            off += ln

        (key_blob_size,) = struct.unpack_from("<I", self._mm, off)
        off += 4
        self._key_blob_off = off
        off += key_blob_size

        self._key_offs_off = off
        off += 4 * (n_words + 1)
        self._val_offs_off = off
        off += 4 * (n_words + 1)
        self._val_blob_off = off

        # Materialize sorted key views as bytes for bisect (one bytes object each
        # would defeat the purpose). Store offsets only; compare via mmap slices.
        self._key_offs = [
            struct.unpack_from("<I", self._mm, self._key_offs_off + 4 * i)[0]
            for i in range(n_words + 1)
        ]

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __len__(self) -> int:
        return self._n

    def _key_at(self, i: int) -> bytes:
        a = self._key_offs[i]
        b = self._key_offs[i + 1]
        # keys stored without trailing NUL between via offsets; blob may use \0
        chunk = self._mm[
            self._key_blob_off + a : self._key_blob_off + b
        ]
        if chunk.endswith(b"\0"):
            chunk = chunk[:-1]
        return bytes(chunk)

    def _find(self, word: str) -> int:
        key = word.encode("ascii", errors="ignore")
        lo, hi = 0, self._n
        while lo < hi:
            mid = (lo + hi) // 2
            mk = self._key_at(mid)
            if mk < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < self._n and self._key_at(lo) == key:
            return lo
        return -1

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return self._find(word) >= 0

    def __getitem__(self, word: str) -> List[List[str]]:
        i = self._find(word)
        if i < 0:
            raise KeyError(word)
        return self._decode_value(i)

    def get(self, word: str, default=None):
        i = self._find(word)
        if i < 0:
            return default
        return self._decode_value(i)

    def _decode_value(self, i: int) -> List[List[str]]:
        (a,) = struct.unpack_from("<I", self._mm, self._val_offs_off + 4 * i)
        (b,) = struct.unpack_from("<I", self._mm, self._val_offs_off + 4 * (i + 1))
        blob = self._mm[self._val_blob_off + a : self._val_blob_off + b]
        pos = 0
        n_syl = blob[pos]
        pos += 1
        out: List[List[str]] = []
        for _ in range(n_syl):
            n_ph = blob[pos]
            pos += 1
            syl = [self._phones[blob[pos + j]] for j in range(n_ph)]
            pos += n_ph
            out.append(syl)
        return out

    def keys(self) -> Iterator[str]:
        for i in range(self._n):
            yield self._key_at(i).decode("ascii", errors="replace")


def build_compact(pickle_path: str | Path, out_path: str | Path) -> dict:
    """Convert openepd_eng_dict.pickle → .oedb. Run on data-disk host."""
    import pickle

    pickle_path = Path(pickle_path)
    out_path = Path(out_path)
    with open(pickle_path, "rb") as f:
        eng = pickle.load(f)
    if not isinstance(eng, dict):
        raise TypeError(f"expected dict, got {type(eng)}")

    phone_to_id: dict[str, int] = {}
    phones: List[str] = []

    def pid(ph: str) -> int:
        if ph not in phone_to_id:
            if len(phones) >= 255:
                raise RuntimeError("too many distinct phones for u8 id")
            phone_to_id[ph] = len(phones)
            phones.append(ph)
        return phone_to_id[ph]

    items: List[Tuple[bytes, bytes]] = []
    for word, syls in eng.items():
        if not isinstance(word, str):
            continue
        key = word.upper().encode("ascii", errors="ignore")
        if not key:
            continue
        parts = bytearray()
        if not isinstance(syls, (list, tuple)):
            continue
        # clamp syllable count
        syl_list = list(syls)[:255]
        parts.append(len(syl_list))
        for syl in syl_list:
            phs = [str(p) for p in syl][:255]
            parts.append(len(phs))
            for p in phs:
                parts.append(pid(p))
        items.append((key, bytes(parts)))

    items.sort(key=lambda x: x[0])
    # dedupe keys (keep first)
    dedup: List[Tuple[bytes, bytes]] = []
    prev = None
    for k, v in items:
        if k == prev:
            continue
        dedup.append((k, v))
        prev = k
    items = dedup

    key_blob = bytearray()
    key_offs = [0]
    val_blob = bytearray()
    val_offs = [0]
    for k, v in items:
        key_blob.extend(k)
        key_blob.append(0)
        key_offs.append(len(key_blob))
        val_blob.extend(v)
        val_offs.append(len(val_blob))

    phone_lens = bytes(len(p.encode("ascii")) for p in phones)
    phone_chars = b"".join(p.encode("ascii") for p in phones)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(_HEADER.pack(MAGIC, VERSION, len(items), len(phones)))
        f.write(phone_lens)
        f.write(phone_chars)
        f.write(struct.pack("<I", len(key_blob)))
        f.write(key_blob)
        f.write(struct.pack(f"<{len(key_offs)}I", *key_offs))
        f.write(struct.pack(f"<{len(val_offs)}I", *val_offs))
        f.write(val_blob)

    return {
        "words": len(items),
        "phones": len(phones),
        "out_bytes": out_path.stat().st_size,
        "out": str(out_path),
    }


def load_openepd(path: str | Path):
    """Load pickle dict or compact .oedb (same lookup API)."""
    path = Path(path)
    if path.suffix.lower() in {".oedb", ".bin"}:
        return CompactEngDict(path)
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
