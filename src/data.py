import numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

RAW, PROC = Path('data/raw'), Path('data/processed')

def load_raw():
    plays = pd.read_csv(RAW / 'usersha1-artmbid-artname-plays.tsv', sep='\t', header=None,
                        names=['user_id', 'artist_mbid', 'artist_name', 'plays'])
    prof = pd.read_csv(RAW / 'usersha1-profile.tsv', sep='\t', header=None,
                       names=['user_id', 'gender', 'age', 'country', 'signup_date'])
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