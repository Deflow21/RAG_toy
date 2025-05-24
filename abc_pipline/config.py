# config.py
from pathlib import Path

# где всё будет лежать
DATA_DIR  = Path("data")              # ./data/raw/…
RAW_DIR   = DATA_DIR/"raw"
UNZIP_DIR = DATA_DIR/"unzipped"
JSON_DIR  = DATA_DIR/"json"
INDEX_DIR = DATA_DIR/"index"

# ----- ABC URL-ы -----
STAT_ZIPS = [
    ("https://archive.nyu.edu/rest/bitstreams/89086/retrieve", "abc_0000_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89089/retrieve", "abc_0001_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89092/retrieve", "abc_0002_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89095/retrieve", "abc_0003_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89098/retrieve", "abc_0004_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89101/retrieve", "abc_0005_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89104/retrieve", "abc_0006_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89107/retrieve", "abc_0007_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89110/retrieve", "abc_0008_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89113/retrieve", "abc_0009_stat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89116/retrieve", "abc_0010_stat_v00.7z")
]
OFS_ZIPS  = [
    ("https://archive.nyu.edu/rest/bitstreams/121765/retrieve", "abc_0000_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121766/retrieve", "abc_0001_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121767/retrieve", "abc_0002_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121768/retrieve", "abc_0003_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121769/retrieve", "abc_0004_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121770/retrieve", "abc_0005_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121771/retrieve", "abc_0006_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121772/retrieve", "abc_0007_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121773/retrieve", "abc_0008_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121774/retrieve", "abc_0009_ofs_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/121775/retrieve", "abc_0010_ofs_v00.7z")
]
FEAT_ZIPS = [
    ("https://archive.nyu.edu/rest/bitstreams/89087/retrieve",  "abc_0000_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89090/retrieve",  "abc_0001_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89093/retrieve",  "abc_0002_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89096/retrieve",  "abc_0003_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89099/retrieve",  "abc_0004_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89102/retrieve",  "abc_0005_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89105/retrieve",  "abc_0006_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89108/retrieve",  "abc_0007_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89111/retrieve",  "abc_0008_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89114/retrieve",  "abc_0009_feat_v00.7z"),
    ("https://archive.nyu.edu/rest/bitstreams/89117/retrieve",  "abc_0010_feat_v00.7z")
]

URL_BATCHES = STAT_ZIPS + OFS_ZIPS + FEAT_ZIPS

# сколько потоков использовать при скачивании
N_JOBS = 4
