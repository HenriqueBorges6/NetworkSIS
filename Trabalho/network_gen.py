"""
network_gen.py - Geracao de redes

Este modulo contem funcoes para gerar diferentes topologias de rede.
"""

import numpy as np
import networkx as nx


def gerar_rede_er(n, k_medio, seed=42):
    """
    Gera uma rede Erdős-Rényi (ER) com grau médio especificado.

    Parâmetros:
    ----------
    n : int
        Número de nós
    k_medio : float
        Grau médio desejado
    seed : int, optional
        Seed para reprodutibilidade (padrão: 42)

    Retorna:
    -------
    networkx.Graph
        Rede ER gerada
    """
    # Probabilidade de conexão para atingir grau médio
    p = k_medio / (n - 1)

    print(f"Gerando rede Erdos-Renyi...")
    print(f"  n = {n} nos")
    print(f"  <k> alvo = {k_medio}")
    print(f"  p = {p:.6f}")

    G = nx.erdos_renyi_graph(n, p, seed=seed)

    # Verifica propriedades
    graus = [d for n, d in G.degree()]
    k_real = np.mean(graus)

    print(f"\nRede gerada:")
    print(f"  Nos: {G.number_of_nodes()}")
    print(f"  Arestas: {G.number_of_edges()}")
    print(f"  <k> real: {k_real:.2f}")
    print(f"  Grau min: {min(graus)}")
    print(f"  Grau max: {max(graus)}")
    print(f"  Conexa: {nx.is_connected(G)}")

    return G


def gerar_rede_scale_free(n, k_medio_alvo, gamma, max_tentativas=50, seed=42, metodo='configuration'):
    """
    Gera rede scale-free.

    metodo: 'ba' -> Barabasi-Albert (m = k_medio_alvo/2, expoente ~3)
            'configuration' -> sequência de graus com lei de potência (expoente = gamma)
    """
    import math
    np.random.seed(seed)

    print(f"Gerando rede scale-free (metodo={metodo})...")
    print(f"  n = {n}, <k> alvo = {k_medio_alvo}, gamma = {gamma}")

    if metodo == 'ba':
        m = max(1, int(round(k_medio_alvo / 2)))
        print(f"  usando BA com m = {m}")
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        return G

    # metodo == 'configuration'
    melhor_G = None
    melhor_diff = float('inf')

    k_min = max(1, int(max(2, k_medio_alvo // 4)))
    k_max = n - 1

    for tentativa in range(max_tentativas):
        # Amostragem por inversa da lei de potência contínua
        u = np.random.random(n)
        if gamma == 1:
            # caso degenerado
            ks = k_min * (k_max / k_min) ** u
        else:
            a = k_min ** (1 - gamma)
            b = k_max ** (1 - gamma)
            ks = (a + u * (b - a)) ** (1.0 / (1 - gamma))
        seq = np.maximum(1, np.floor(ks)).astype(int).tolist()

        # Ajusta média multiplicativamente para aproximar <k> alvo
        media = np.mean(seq)
        if media <= 0:
            continue
        fator = k_medio_alvo / media
        seq = np.clip(np.round(np.array(seq) * fator).astype(int), 1, n - 1).tolist()

        # Garante soma par
        if sum(seq) % 2 == 1:
            idx = np.random.randint(0, n)
            seq[idx] = min(n - 1, seq[idx] + 1)

        # Se sequência gráfica -> Havel-Hakimi para grafo simples
        if not nx.is_graphical(seq):
            # tenta pequenas perturbações antes de descartar
            # shuffle e recortar máximo/mínimo
            np.random.shuffle(seq)
            if not nx.is_graphical(seq):
                continue

        try:
            G = nx.havel_hakimi_graph(seq)
            # remove possíveis isolados criando tentativa de reconexão?
            if nx.is_connected(G):
                k_real = np.mean([d for _, d in G.degree()])
                diff = abs(k_real - k_medio_alvo)
                if diff < melhor_diff:
                    melhor_diff = diff
                    melhor_G = G
                if diff < 1.0:
                    print(f"  Convergiu na tentativa {tentativa+1} (diff={diff:.3f})")
                    return G
        except Exception:
            continue

    if melhor_G is None:
        raise ValueError("Não foi possível gerar rede scale-free conexa após várias tentativas")

    graus = [d for _, d in melhor_G.degree()]
    k_real = np.mean(graus)
    print(f"\nRede retornada (melhor encontrada): Nos={melhor_G.number_of_nodes()}, Arestas={melhor_G.number_of_edges()}, <k>={k_real:.2f}, gamma alvo={gamma}")
    return melhor_G

def verificar_expoente_lei_potencia(G, k_min_estimativa=10, min_pontos=3):
    """
    Estima o expoente γ da lei de potência P(k) ~ k^(-γ).

    Retorna apenas gamma estimado (float). Ajustes:
    - corrige 'ef' -> 'def'
    - evita log(0) filtrando CCDF=0
    - relaxa k_min se houver poucos pontos para ajuste
    """
    import numpy as np
    from scipy import stats

    graus = np.array([d for _, d in G.degree()])
    n_nodes = len(graus)
    if n_nodes == 0:
        raise ValueError("Grafo vazio")

    # histograma por grau (mais simples e robusto que value_counts)
    max_k = int(graus.max())
    hist = np.bincount(graus, minlength=max_k + 1)
    k_vals = np.arange(len(hist))[hist > 0]
    counts = hist[hist > 0]

    # CCDF: P(K >= k)
    # ccdf[i] corresponde a probabilidade para k = k_vals[i]
    cum_counts = np.cumsum(counts[::-1])[::-1]
    ccdf = cum_counts / n_nodes

    # filtro inicial (k mínimo e ccdf>0)
    mask = (k_vals >= k_min_estimativa) & (ccdf > 0)

    # se poucos pontos, relaxar critério (mantendo ccdf>0)
    if np.count_nonzero(mask) < min_pontos:
        mask = (ccdf > 0)
        if np.count_nonzero(mask) < min_pontos:
            raise ValueError("Pontos insuficientes para ajuste da cauda (ccdf>0)")

    use_k = k_vals[mask]
    use_ccdf = ccdf[mask]

    # regressão log-log
    log_k = np.log(use_k)
    log_ccdf = np.log(use_ccdf)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_ccdf)

    # para CCDF: slope = -gamma + 1  => gamma = -slope + 1
    gamma_estimado = -slope + 1

    print(f"\nAnalise de lei de potencia:")
    print(f"  pontos usados: {len(use_k)}  k_min usado: {use_k.min()}")
    print(f"  gamma estimado: {gamma_estimado:.2f}")
    print(f"  R2 do ajuste: {r_value**2:.4f}")

    return gamma_estimado
