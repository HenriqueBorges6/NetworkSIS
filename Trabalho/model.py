"""
model.py - Implementacao do Modelo SIS (Susceptible-Infected-Susceptible)

Este modulo contem as funcoes principais para simulacao de epidemias em redes.
"""

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os


def modelo_sis(G, beta, mu, inicial_infectados, max_steps=1000, imunizados=None):
    """
    Simula o modelo SIS em uma rede.

    Parâmetros:
    ----------
    G : networkx.Graph
        Grafo da rede
    beta : float
        Taxa de infecção (probabilidade de infectar por contato)
    mu : float
        Taxa de recuperação (probabilidade de se recuperar)
    inicial_infectados : array-like
        Lista de nós inicialmente infectados
    max_steps : int, optional
        Número máximo de passos da simulação (padrão: 1000)
    imunizados : set, optional
        Conjunto de nós imunizados/vacinados (padrão: None)

    Retorna:
    -------
    list
        Histórico com número de infectados em cada passo temporal

    Estados dos nós:
    ---------------
    - 0: Suscetível (pode ser infectado)
    - 1: Infectado (pode infectar vizinhos)
    - -1: Imunizado (vacinado, não pode ser infectado)
    """
    if imunizados is None:
        imunizados = set()

    # Estado dos nós: 0 = suscetível, 1 = infectado, -1 = imunizado
    estado = {node: 0 for node in G.nodes()}

    # Marca nós imunizados
    for node in imunizados:
        estado[node] = -1

    # Infecta nós iniciais (se não estiverem imunizados)
    for node in inicial_infectados:
        if estado[node] != -1:
            estado[node] = 1

    historico = []

    for step in range(max_steps):
        # Conta infectados atuais
        infectados = [node for node in G.nodes() if estado[node] == 1]
        historico.append(len(infectados))

        if len(infectados) == 0:
            # Epidemia extinta
            break

        # Cria novo estado (evita conflitos de atualização)
        novo_estado = estado.copy()

        # Processo de infecção
        for node in G.nodes():
            if estado[node] == 0:  # Suscetível
                # Conta vizinhos infectados
                vizinhos_infectados = sum(1 for vizinho in G.neighbors(node)
                                         if estado[vizinho] == 1)

                if vizinhos_infectados > 0:
                    # Probabilidade de infecção: 1 - (1-beta)^k_inf
                    # onde k_inf é o número de vizinhos infectados
                    prob_infeccao = 1 - (1 - beta) ** vizinhos_infectados

                    if np.random.random() < prob_infeccao:
                        novo_estado[node] = 1

        # Processo de recuperação
        for node in infectados:
            if np.random.random() < mu:
                novo_estado[node] = 0  # Volta para suscetível (modelo SIS)

        estado = novo_estado

    return historico


def _simular_uma_vez(G, beta, mu, nos_disponiveis, n_escolher, max_steps, imunizados, seed):
    """
    Funcao auxiliar para executar UMA simulacao (usada em paralelismo).

    Parametros:
    ----------
    seed : int
        Seed para reproducibilidade de cada simulacao
    """
    np.random.seed(seed)

    # Escolhe nos iniciais aleatoriamente
    inicial_infectados = np.random.choice(nos_disponiveis,
                                         size=n_escolher,
                                         replace=False)

    return modelo_sis(G, beta, mu, inicial_infectados, max_steps, imunizados)


def executar_multiplas_simulacoes(G, beta, mu, n_inicial_infectados=5,
                                   n_simulacoes=100, max_steps=1000,
                                   imunizados=None, verbose=True,
                                   n_jobs=-1, paralelo=True):
    """
    Executa multiplas simulacoes do modelo SIS (com suporte a paralelismo).

    Parametros:
    ----------
    G : networkx.Graph
        Grafo da rede
    beta : float
        Taxa de infeccao
    mu : float
        Taxa de recuperacao
    n_inicial_infectados : int, optional
        Numero de nos inicialmente infectados (padrao: 5)
    n_simulacoes : int, optional
        Numero de simulacoes a executar (padrao: 100)
    max_steps : int, optional
        Numero maximo de passos por simulacao (padrao: 1000)
    imunizados : set, optional
        Conjunto de nos imunizados (padrao: None)
    verbose : bool, optional
        Se True, mostra barra de progresso (padrao: True)
    n_jobs : int, optional
        Numero de processos paralelos (-1 = usa todos os CPUs) (padrao: -1)
    paralelo : str or bool, optional
        'auto': decide automaticamente (padrao)
        True: sempre usa paralelismo
        False: sempre serial

    Retorna:
    -------
    list of lists
        Lista contendo o historico de cada simulacao
    """
    # Pre-calcula nos disponiveis para infeccao inicial
    if imunizados is None:
        nos_disponiveis = list(G.nodes())
    else:
        nos_disponiveis = [n for n in G.nodes() if n not in imunizados]

    n_escolher = min(n_inicial_infectados, len(nos_disponiveis))

    # Decide se usa paralelismo
    if paralelo == 'auto':
        # Usa paralelismo apenas para problemas grandes
        usar_paralelo = (G.number_of_nodes() > 5000) or (n_simulacoes > 50)
    else:
        usar_paralelo = bool(paralelo)

    if not usar_paralelo:
        # Versao serial
        todos_historicos = []
        iterador = range(n_simulacoes)
        if verbose:
            iterador = tqdm(iterador, desc="Simulacoes")

        for i in iterador:
            historico = _simular_uma_vez(G, beta, mu, nos_disponiveis,
                                        n_escolher, max_steps, imunizados,
                                        seed=42 + i)
            todos_historicos.append(historico)

        return todos_historicos

    # Versao paralela com joblib
    n_cpus = os.cpu_count() if n_jobs == -1 else n_jobs

    if verbose:
        print(f"Executando {n_simulacoes} simulacoes usando {n_cpus} processos paralelos...")

    # batch_size maior reduz overhead
    batch_size = max(1, n_simulacoes // (n_cpus * 2))

    todos_historicos = Parallel(n_jobs=n_jobs,
                                batch_size=batch_size,
                                verbose=5 if verbose else 0)(
        delayed(_simular_uma_vez)(
            G, beta, mu, nos_disponiveis, n_escolher, max_steps, imunizados, 42 + i
        ) for i in range(n_simulacoes)
    )

    return todos_historicos


def calcular_R0(beta, k_medio, mu):
    """
    Calcula o número básico de reprodução R₀.

    R₀ = (β × <k>) / μ

    Parâmetros:
    ----------
    beta : float
        Taxa de infecção
    k_medio : float
        Grau médio da rede
    mu : float
        Taxa de recuperação

    Retorna:
    -------
    float
        Valor de R₀
    """
    return (beta * k_medio) / mu


def analisar_resultados(historicos):
    """
    Analisa resultados de múltiplas simulações.

    Parâmetros:
    ----------
    historicos : list of lists
        Lista de históricos de simulações

    Retorna:
    -------
    dict
        Dicionário com estatísticas:
        - 'media_final': média de infectados no estado final
        - 'std_final': desvio padrão no estado final
        - 'epidemias_extintas': número de epidemias extintas
        - 'fracao_extinta': fração de epidemias extintas
        - 'duracao_media': duração média das simulações
    """
    valores_finais = [h[-1] if len(h) > 0 else 0 for h in historicos]
    duracoes = [len(h) for h in historicos]

    epidemias_extintas = sum(1 for v in valores_finais if v == 0)

    return {
        'media_final': np.mean(valores_finais),
        'std_final': np.std(valores_finais),
        'epidemias_extintas': epidemias_extintas,
        'fracao_extinta': epidemias_extintas / len(historicos),
        'duracao_media': np.mean(duracoes),
        'valores_finais': valores_finais,
        'duracoes': duracoes
    }
