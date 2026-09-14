import urllib.request
import tarfile
import os

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
DEST = "data/raw"

os.makedirs(DEST, exist_ok=True)
caminho_tar = os.path.join(DEST, "lastfm-dataset-360K.tar.gz")

urllib.request.urlretrieve(URL, caminho_tar)
with tarfile.open(caminho_tar) as tar:
    tar.extractall(DEST)