'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: execução dos experimentos em processos paralelos (grade de hiperparâmetros na
         validação e rodada final por semente), para viabilizar o treino em CPU.
Histórico:
2026-11-13 | Luis Felipe | Criação do módulo
2026-11-13 | Luis Felipe | Cache em disco por job, para retomar execuções interrompidas
2026-11-18 | Luis Felipe | Atualização da lista de integrantes
'''
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def _setup(threads):
    import warnings, torch
    warnings.filterwarnings('ignore')
    torch.set_num_threads(threads)


def grid_job(job):
    '''Treina o GraphSAGE (pesos aprendidos) em uma configuração e devolve o NDCG@10 de validação.'''
    _setup(job.get('threads', 2))
    from src.data import load_processed, RESULTS
    from src.walks import build_adj, random_walk_counts, walk_graph
    from src.train import train_gnn
    cache = RESULTS / 'cache' / f"grade_T{job['T']}_d{job['d']}_lr{job['lr']}_ep{job['max_epochs']}.json"
    if cache.exists():                                   # job já concluído em execução anterior
        return json.loads(cache.read_text())
    D = load_processed()
    indptr, indices = build_adj(D['train'], D['nU'], D['nA'])
    C = random_walk_counts(indptr, indices, num_walks=50, walk_len=6, seed=0)
    ei, ew = walk_graph(C, job['T'])
    m, h, secs = train_gnn(D, ei, ew, learn_weights=True, d=job['d'], lr=job['lr'],
                           max_epochs=job['max_epochs'])
    out = dict(T=job['T'], d=job['d'], lr=job['lr'], val_ndcg10=float(h['val_ndcg@10'].max()),
               melhor_epoca=int(h.loc[h['val_ndcg@10'].idxmax(), 'epoch']),
               epocas=len(h), segundos=round(secs))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def seed_job(job):
    '''Rodada final de uma semente: caminhada, ALS e as duas variantes do GraphSAGE, no teste.'''
    _setup(job.get('threads', 2))
    import pickle, time
    import numpy as np, torch
    from src.data import load_processed, RESULTS
    cache = RESULTS / 'cache' / f"final_seed{job['seed']}_T{job['T']}_d{job['d']}_lr{job['lr']}_ep{job['max_epochs']}.pkl"
    if cache.exists():                                   # semente já concluída em execução anterior
        return pickle.loads(cache.read_bytes())
    from src.walks import build_adj, random_walk_counts, walk_graph
    from src.baselines import walk_fn, als_fn
    from src.train import train_gnn, gnn_fn
    from src.metrics import evaluate
    seed, Ks = job['seed'], job['Ks']
    D = load_processed()
    test_mask = (D['train_csr'] + D['val_csr']).tocsr()      # no teste, treino e validação já são conhecidos
    target = D['test_csr']
    pop = np.asarray(D['train_csr'].sum(0)).ravel()
    indptr, indices = build_adj(D['train'], D['nU'], D['nA'])
    rows, per_user, rec_store, curvas = [], {}, {}, {}

    def run(name, fn, secs=0.0):
        s, pu, rc = evaluate(fn, test_mask, target, Ks)
        rows.append({'modelo': name, 'seed': seed, 'segundos_treino': secs, **s})
        per_user[(name, seed)] = pu; rec_store[(name, seed)] = rc

    t0 = time.perf_counter()
    C = random_walk_counts(indptr, indices, seed=seed)
    run('Caminhada (Pixie)', walk_fn(C, D['nU'], pop), time.perf_counter() - t0)
    t0 = time.perf_counter()
    fn = als_fn(D['train_w_csr'], seed=seed, **job['als'])
    run('ALS', fn, time.perf_counter() - t0)
    ei, ew = walk_graph(C, job['T'])
    for name, lw in (('GraphSAGE (pesos fixos)', False), ('GraphSAGE (pesos aprendidos)', True)):
        m, h, secs = train_gnn(D, ei, ew, learn_weights=lw, seed=seed, d=job['d'], lr=job['lr'],
                               max_epochs=job['max_epochs'])
        run(name, gnn_fn(m, ei, ew), secs)
        curvas[lw] = h
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), cache.parent / f'modelo_seed{seed}_aprendido{lw}.pt')   # para análise posterior
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((rows, per_user, rec_store, curvas)))
    return rows, per_user, rec_store, curvas


def run_parallel(fn, jobs, workers=4):
    '''Executa os jobs em processos separados ("spawn": seguro com PyTorch e igual no Windows).'''
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        return list(ex.map(fn, jobs))
