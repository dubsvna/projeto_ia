'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: baselines de popularidade, caminhada aleatória (estilo Pixie) e ALS (implicit).
Histórico:
2026-11-10 | Arthur | Criação do módulo
2026-11-18 | Luis Felipe | Atualização da lista de integrantes
'''
import numpy as np
from implicit.als import AlternatingLeastSquares

def popularity_fn(train_csr):
    pop = np.asarray(train_csr.sum(0)).ravel().astype(np.float32)
    return lambda us: np.broadcast_to(pop, (len(us), len(pop)))

def walk_fn(C, nU, pop_tiebreak):
    '''Heurística estilo Pixie: score = nº de visitas da caminhada aleatória ao artista.
    C é a matriz de contagens (ver 3.3); desempate por popularidade.'''
    Cua = C[:nU, nU:].tocsr()
    eps = 1e-6 * pop_tiebreak / pop_tiebreak.max()
    return lambda us: Cua[us].toarray() + eps

def als_fn(train_w_csr, factors=64, reg=0.01, iters=15, alpha=20, seed=0):
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                    alpha=alpha, random_state=seed, use_gpu=False)
    model.fit(train_w_csr)                      # matriz usuários x itens (implicit >= 0.5)
    U, I = model.user_factors, model.item_factors
    return lambda us: U[us] @ I.T