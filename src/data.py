'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Alex Cruz de Santana 10420357; Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: carregamento, limpeza, filtro k-core, subamostra, indexação e split da base Last.fm-360K.
Histórico:
2026-11-10 | Arthur | Criação do módulo
2026-11-13 | Luis Felipe | Caminhos ancorados na raiz do repositório e busca dos .tsv em subpastas de data/raw
'''
import numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

# Caminhos ancorados na raiz do repositório (pasta que contém src/), e não no diretório
# de trabalho: assim o código funciona tanto a partir da raiz quanto de notebooks/.
ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RESULTS = ROOT / 'data' / 'raw', ROOT / 'data' / 'processed', ROOT / 'results'

def find_raw(name):
    '''Localiza um arquivo bruto em data/raw ou em subpastas. O .tar.gz oficial extrai os
    .tsv dentro de lastfm-dataset-360K/, e não diretamente em data/raw.'''
    hits = [RAW / name] if (RAW / name).exists() else sorted(RAW.rglob(name))
    if not hits:
        raise FileNotFoundError(f'{name} não encontrado em {RAW} (nem em subpastas). '
                                'Execute scripts/download_dataset.py a partir de qualquer pasta.')
    return hits[0]

def load_raw():
    plays = pd.read_csv(find_raw('usersha1-artmbid-artname-plays.tsv'), sep='\t', header=None,
                        names=['user_id', 'artist_mbid', 'artist_name', 'plays'],
                        quoting=3, dtype={'user_id': str, 'artist_mbid': str, 'artist_name': str})
    prof = pd.read_csv(find_raw('usersha1-profile.tsv'), sep='\t', header=None,
                       names=['user_id', 'gender', 'age', 'country', 'signup_date'], quoting=3)
    return plays, prof

def clean(plays):
    df = plays.dropna(subset=['artist_name']).copy()
    df['artist'] = df['artist_name'].str.lower().str.strip()
    df = df[(df['plays'] > 0) & (df['artist'] != '')]
    return df.groupby(['user_id', 'artist'], as_index=False)['plays'].sum()

def kcore(df, min_user, min_item):
    while True:                       # repete até estabilizar
        n0 = len(df)
        ic = df['artist'].value_counts()
        df = df[df['artist'].isin(ic[ic >= min_item].index)]
        uc = df['user_id'].value_counts()
        df = df[df['user_id'].isin(uc[uc >= min_user].index)]
        if len(df) == n0:
            return df

def build_interactions(plays, n_users=20000, seed=42):
    df = kcore(clean(plays), min_user=20, min_item=50)
    rng = np.random.default_rng(seed)
    users = df['user_id'].unique()
    keep = rng.choice(users, size=min(n_users, len(users)), replace=False)
    df = kcore(df[df['user_id'].isin(keep)], min_user=20, min_item=10)
    return df.reset_index(drop=True)

def index_and_features(df, prof, top_countries=30):
    users = np.sort(df['user_id'].unique())
    arts = np.sort(df['artist'].unique())
    df = df.assign(u=pd.Index(users).get_indexer(df['user_id']),
                   a=pd.Index(arts).get_indexer(df['artist']))
    c = prof.drop_duplicates('user_id').set_index('user_id')['country'].reindex(users).fillna('unk')
    c = c.where(c.isin(c.value_counts().index[:top_countries]), 'other')
    return df, pd.factorize(c)[0], users, arts     # país como código inteiro por usuário

def split_per_user(df, val_frac=0.1, test_frac=0.1, seed=42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # embaralha
    rank = df.groupby('u').cumcount()
    size = df.groupby('u')['u'].transform('size')
    n_test = (size * test_frac).round().clip(lower=1).astype(int)
    n_val = (size * val_frac).round().clip(lower=1).astype(int)
    df['split'] = np.where(rank < n_test, 'test', np.where(rank < n_test + n_val, 'val', 'train'))
    return df

def to_csr(df, nU, nA, weighted=False):
    v = np.log1p(df['plays'].values).astype(np.float32) if weighted else np.ones(len(df), np.float32)
    return sp.csr_matrix((v, (df['u'].values, df['a'].values)), shape=(nU, nA))

def load_processed():
    df = pd.read_parquet(PROC / 'interactions.parquet')
    country = np.load(PROC / 'user_country.npy')
    nU, nA = int(df['u'].max()) + 1, int(df['a'].max()) + 1
    D = dict(nU=nU, nA=nA, country=country, n_country=int(country.max()) + 1)
    for s in ('train', 'val', 'test'):
        D[s] = df[df['split'] == s].reset_index(drop=True)
        D[s + '_csr'] = to_csr(D[s], nU, nA)
    D['train_w_csr'] = to_csr(D['train'], nU, nA, weighted=True)   # só o ALS usa pesos
    return D