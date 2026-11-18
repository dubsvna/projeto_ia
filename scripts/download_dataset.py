'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: download e extração da base Last.fm-360K (Celma, 2010) em data/raw.
Histórico:
2026-11-10 | Arthur | Criação do script
2026-11-13 | Luis Felipe | Caminho ancorado na raiz do repositório, User-Agent (o servidor
                           responde 403 sem ele), download só se o arquivo não existir e
                           extração dos .tsv direto em data/raw (sem a subpasta do .tar.gz)
2026-11-18 | Luis Felipe | Atualização da lista de integrantes
'''
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"      # independe do diretório atual
ARQUIVOS = ("usersha1-artmbid-artname-plays.tsv", "usersha1-profile.tsv")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / a).exists() for a in ARQUIVOS):
        print("Dataset já presente em", DEST)
        return
    caminho_tar = DEST / "lastfm-dataset-360K.tar.gz"
    if not caminho_tar.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(str(caminho_tar) + ".part", "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        os.replace(str(caminho_tar) + ".part", caminho_tar)
    with tarfile.open(caminho_tar) as tar:
        for membro in tar.getmembers():
            nome = os.path.basename(membro.name)
            if nome in ARQUIVOS:                 # o .tar.gz guarda tudo em lastfm-dataset-360K/
                with tar.extractfile(membro) as origem, open(DEST / nome, "wb") as destino:
                    shutil.copyfileobj(origem, destino, length=1 << 20)
    print("Download concluído em", DEST)


if __name__ == "__main__":
    main()
