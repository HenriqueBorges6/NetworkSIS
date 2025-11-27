"""
utils.py - Funções auxiliares para análise e visualização

Este módulo contém funções para processamento de dados e geração de gráficos.
"""

import numpy as np
import matplotlib.pyplot as plt


def calcular_estatisticas(historicos):
    """
    Calcula estatísticas (média, desvio, percentis) dos históricos.

    Parâmetros:
    ----------
    historicos : list of lists
        Lista de históricos de simulações

    Retorna:
    -------
    tuple
        (media, std, p25, p75) - arrays NumPy com estatísticas por tempo
    """
    if not historicos:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Encontra o tamanho máximo
    max_len = max(len(h) for h in historicos)

    # Preenche históricos mais curtos com o último valor
    historicos_padded = []
    for h in historicos:
        if len(h) < max_len:
            ultimo_valor = h[-1] if len(h) > 0 else 0
            h_padded = h + [ultimo_valor] * (max_len - len(h))
        else:
            h_padded = h
        historicos_padded.append(h_padded)

    historicos_array = np.array(historicos_padded)

    media = np.mean(historicos_array, axis=0)
    std = np.std(historicos_array, axis=0)
    p25 = np.percentile(historicos_array, 25, axis=0)
    p75 = np.percentile(historicos_array, 75, axis=0)

    return media, std, p25, p75


def plotar_simulacoes(historicos, titulo, beta, mu, k_medio,
                      n_amostras=10, figsize=(12, 6), salvar=None):
    """
    Plota resultados de múltiplas simulações.

    Parâmetros:
    ----------
    historicos : list of lists
        Lista de históricos de simulações
    titulo : str
        Título do gráfico
    beta : float
        Taxa de infecção
    mu : float
        Taxa de recuperação
    k_medio : float
        Grau médio da rede
    n_amostras : int, optional
        Número de simulações individuais a plotar (padrão: 10)
    figsize : tuple, optional
        Tamanho da figura (padrão: (12, 6))
    salvar : str, optional
        Caminho para salvar a figura (padrão: None)
    """
    media, std, p25, p75 = calcular_estatisticas(historicos)

    # Calcula R0
    R0 = (beta * k_medio) / mu

    plt.figure(figsize=figsize)

    # Plota algumas simulações individuais
    for i, h in enumerate(historicos[:n_amostras]):
        plt.plot(h, alpha=0.2, color='gray', linewidth=0.5)

    # Plota média e intervalo de confiança
    steps = np.arange(len(media))
    plt.plot(steps, media, 'b-', linewidth=2, label='Média')
    plt.fill_between(steps, p25, p75, alpha=0.3, color='blue',
                     label='Percentis 25-75')

    plt.xlabel('Tempo (passos)', fontsize=12)
    plt.ylabel('Número de Infectados', fontsize=12)
    plt.title(f'{titulo}\nβ={beta}, μ={mu}, R₀={R0:.2f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if salvar:
        plt.savefig(salvar, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {salvar}")

    plt.show()


def imprimir_estatisticas(historicos, titulo, beta, mu, k_medio):
    """
    Imprime estatísticas resumidas das simulações.

    Parâmetros:
    ----------
    historicos : list of lists
        Lista de históricos de simulações
    titulo : str
        Título da análise
    beta : float
        Taxa de infecção
    mu : float
        Taxa de recuperação
    k_medio : float
        Grau médio da rede
    """
    R0 = (beta * k_medio) / mu

    valores_finais = [h[-1] if len(h) > 0 else 0 for h in historicos]
    media_final = np.mean(valores_finais)
    std_final = np.std(valores_finais)
    epidemias_extintas = sum(1 for v in valores_finais if v == 0)

    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")
    print(f"Parâmetros:")
    print(f"  β (taxa de infecção):    {beta}")
    print(f"  μ (taxa de recuperação): {mu}")
    print(f"  <k> (grau médio):        {k_medio:.2f}")
    print(f"\nR₀ = {R0:.4f}")
    print(f"Previsão teórica: {'Endêmica' if R0 > 1 else 'Extinta'}")
    print(f"\nResultados ({len(historicos)} simulações):")
    print(f"  Infectados no estado estacionário: {media_final:.2f} ± {std_final:.2f}")
    print(f"  Epidemias extintas: {epidemias_extintas}/{len(historicos)} ({100*epidemias_extintas/len(historicos):.1f}%)")
    print(f"{'='*60}\n")


def plotar_distribuicao_graus(G, titulo, log_scale=True, figsize=(12, 4), salvar=None):
    """
    Plota a distribuição de graus da rede.

    Parâmetros:
    ----------
    G : networkx.Graph
        Grafo da rede
    titulo : str
        Título do gráfico
    log_scale : bool, optional
        Se True, usa escala log-log (padrão: True)
    figsize : tuple, optional
        Tamanho da figura (padrão: (12, 4))
    salvar : str, optional
        Caminho para salvar a figura (padrão: None)
    """
    import pandas as pd

    graus = [d for n, d in G.degree()]
    k_medio = np.mean(graus)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histograma
    ax1.hist(graus, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Grau', fontsize=11)
    ax1.set_ylabel('Frequência', fontsize=11)
    ax1.set_title(f'{titulo}\n<k> = {k_medio:.2f}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(k_medio, color='red', linestyle='--', linewidth=2,
                label=f'Média = {k_medio:.1f}')
    ax1.legend()

    # Distribuição (log-log se solicitado)
    degree_count = pd.Series(graus).value_counts().sort_index()
    if log_scale:
        ax2.loglog(degree_count.index, degree_count.values, 'o', alpha=0.7)
        ax2.set_title('Distribuição P(k) (log-log)', fontsize=12)
    else:
        ax2.plot(degree_count.index, degree_count.values, 'o', alpha=0.7)
        ax2.set_title('Distribuição P(k)', fontsize=12)

    ax2.set_xlabel('Grau (k)', fontsize=11)
    ax2.set_ylabel('P(k)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar:
        plt.savefig(salvar, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {salvar}")

    plt.show()


def salvar_resultados(dados, arquivo):
    """
    Salva resultados em arquivo pickle.

    Parâmetros:
    ----------
    dados : dict
        Dicionário com dados a salvar
    arquivo : str
        Caminho do arquivo
    """
    import pickle

    with open(arquivo, 'wb') as f:
        pickle.dump(dados, f)

    print(f"Resultados salvos em: {arquivo}")


def carregar_resultados(arquivo):
    """
    Carrega resultados de arquivo pickle.

    Parâmetros:
    ----------
    arquivo : str
        Caminho do arquivo

    Retorna:
    -------
    dict
        Dados carregados
    """
    import pickle

    with open(arquivo, 'rb') as f:
        dados = pickle.load(f)

    print(f"Resultados carregados de: {arquivo}")
    return dados
