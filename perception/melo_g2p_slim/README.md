# Slim Melo English G2P (git → image).

Large assets stay on JuiceFS only:
- `checkpoint20.npz` — OOV neural weights
- `openepd_eng_dict.oedb` — compact mmap lexicon (preferred)
- `openepd_eng_dict.pickle` — legacy fallback

Build compact lexicon on the data host:
`python3 deploy/build_openepd_compact.py`
