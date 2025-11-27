"""
Q3_rapido.py - Questao 3: Estrategias de Imunizacao (VERSAO RAPIDA)

Versao ultra-otimizada com:
1. Paralelizacao agressiva
2. Menos pontos (12 ao inves de 20)
3. Menos simulacoes para fracoes extremas (adaptativo)
4. MAX_STEPS reduzido (300 ao inves de 500)

IDEAL PARA TESTES RAPIDOS
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from model import modelo_sis, calcular_R0
from utils import carregar_resultados, salvar_resultados

# Configuracoes OTIMIZADAS
np.random.seed(42)
BETA = 0.01
MU = 0.1
MAX_STEPS = 300  # Reduzido de 500
N_INICIAL_INFECTADOS = 5
N_JOBS = -1

# Cria pasta para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*70)
print(" QUESTAO 3: Estrategias de Imunizacao (VERSAO RAPIDA)")
print("="*70)

# Carrega rede da Q2
try:
    dados_Q2 = carregar_resultados('results/Q2_results.pkl')
    G_SF = dados_Q2['rede']
    k_medio = dados_Q2['k_medio']
except:
    print("ERRO: Execute Q2.py primeiro para gerar a rede scale-free!")
    exit(1)

n_nodes = G_SF.number_of_nodes()
R0 = calcular_R0(BETA, k_medio, MU)

print(f"\nUsando rede scale-free da Q2:")
print(f"  Nos: {n_nodes}")
print(f"  <k>: {k_medio:.2f}")
print(f"  beta={BETA}, mu={MU}, R0={R0:.2f}")


def selecionar_imunizados(G, n_imunizar, estrategia, seed=None):
    """Seleciona nos para imunizar de acordo com a estrategia."""
    if seed is not None:
        np.random.seed(seed)

    if n_imunizar == 0:
        return set()

    if estrategia == 'aleatoria':
        return set(np.random.choice(list(G.nodes()),
                                    size=n_imunizar,
                                    replace=False))

    elif estrategia == 'hubs':
        nos_ordenados = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        return set([n for n, d in nos_ordenados[:n_imunizar]])

    elif estrategia == 'vizinhos':
        imunizados = set()
        nos_amostra = np.random.choice(list(G.nodes()),
                                      size=min(n_imunizar * 2, len(G.nodes())),
                                      replace=False)

        for node in nos_amostra:
            vizinhos = list(G.neighbors(node))
            if vizinhos and len(imunizados) < n_imunizar:
                vizinho = np.random.choice(vizinhos)
                imunizados.add(vizinho)
            if len(imunizados) >= n_imunizar:
                break
        return imunizados


def simular_uma_epidemia(G, beta, mu, n_imunizar, estrategia,
                         max_steps, n_inicial_infectados, seed):
    """Executa uma simulacao de epidemia com imunizacao."""
    imunizados = selecionar_imunizados(G, n_imunizar, estrategia, seed=seed)

    nos_disponiveis = [n for n in G.nodes() if n not in imunizados]
    if len(nos_disponiveis) < n_inicial_infectados:
        return False

    np.random.seed(seed + 1000)
    inicial_infectados = np.random.choice(nos_disponiveis,
                                         size=n_inicial_infectados,
                                         replace=False)

    historico = modelo_sis(G, beta, mu, inicial_infectados,
                          max_steps, imunizados)

    return len(historico) > 0 and historico[-1] > 0


def processar_fracao_adaptativo(G, beta, mu, fracao, estrategia,
                                max_steps, n_inicial_infectados, n_jobs):
    """
    Processa uma fracao com numero ADAPTATIVO de simulacoes.

    - Fracoes baixas (<0.1): 30 simulacoes (rapido, resultado obvio)
    - Fracoes medias (0.1-0.3): 50 simulacoes (regiao de transicao)
    - Fracoes altas (>0.3): 30 simulacoes (rapido, resultado obvio)
    """
    n_nodes = G.number_of_nodes()
    n_imunizar = int(fracao * n_nodes)

    if n_imunizar == 0:
        return 1.0

    # Determina numero de simulacoes baseado na fracao
    if fracao < 0.1 or fracao > 0.35:
        n_simulacoes = 30  # Menos simulacoes para extremos
    else:
        n_simulacoes = 50  # Mais simulacoes na regiao critica

    # Paraleliza as simulacoes
    seeds = range(n_simulacoes)
    resultados = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(simular_uma_epidemia)(
            G, beta, mu, n_imunizar, estrategia,
            max_steps, n_inicial_infectados, seed
        )
        for seed in seeds
    )

    return sum(resultados) / n_simulacoes


def encontrar_fracao_critica_rapida(G, beta, mu, estrategia,
                                    max_steps=300, n_jobs=-1):
    """Versao rapida com menos pontos e simulacoes adaptativas."""
    # Apenas 12 pontos ao inves de 20
    fracoes = np.linspace(0, 0.5, 12)

    resultados = []
    for fracao in tqdm(fracoes, desc=f"Testando {estrategia}"):
        resultado = processar_fracao_adaptativo(
            G, beta, mu, fracao, estrategia,
            max_steps, N_INICIAL_INFECTADOS, n_jobs
        )
        resultados.append(resultado)

    return fracoes, resultados


# Testa as tres estrategias
print("\n" + "="*70)
print(" Testando Estrategias de Imunizacao")
print("="*70)
print(f"Modo: RAPIDO (12 pontos, simulacoes adaptativas, {MAX_STEPS} passos)")
print()

estrategias = ['aleatoria', 'hubs', 'vizinhos']
resultados_estrategias = {}

import time
tempo_inicio = time.time()

for estrategia in estrategias:
    print(f"\nEstrategia: {estrategia}")
    tempo_est_inicio = time.time()

    fracoes, resultados = encontrar_fracao_critica_rapida(
        G_SF, BETA, MU, estrategia,
        max_steps=MAX_STEPS,
        n_jobs=N_JOBS
    )

    tempo_est = time.time() - tempo_est_inicio
    print(f"  Tempo: {tempo_est:.1f}s")

    resultados_estrategias[estrategia] = {
        'fracoes': fracoes,
        'resultados': resultados
    }

tempo_total = time.time() - tempo_inicio
print(f"\nTempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min)")

# Plota comparacao
plt.figure(figsize=(12, 6))

cores = {'aleatoria': 'blue', 'hubs': 'red', 'vizinhos': 'green'}
marcadores = {'aleatoria': 'o', 'hubs': 's', 'vizinhos': '^'}

for estrategia in estrategias:
    dados = resultados_estrategias[estrategia]
    plt.plot(dados['fracoes'], dados['resultados'],
             marcadores[estrategia] + '-',
             label=estrategia.capitalize(),
             linewidth=2,
             color=cores[estrategia],
             markersize=6)

plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5,
            label='Limiar 10%')

plt.xlabel('Fracao de Nos Imunizados', fontsize=12)
plt.ylabel('Fracao de Epidemias Persistentes', fontsize=12)
plt.title(f'Eficacia de Diferentes Estrategias de Imunizacao\n' +
          f'Rede Scale-Free (beta={BETA}, mu={MU}, R0={R0:.2f})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/Q3_comparacao_estrategias.png', dpi=300)
print(f"\nFigura salva em: results/figures/Q3_comparacao_estrategias.png")


def encontrar_fracao_critica(fracoes, resultados, limiar=0.1):
    """Encontra a fracao onde menos de 10% das epidemias persistem."""
    for f, r in zip(fracoes, resultados):
        if r < limiar:
            return f
    return fracoes[-1]


print("\n" + "="*70)
print(" Resultados: Fracao Critica de Imunizacao")
print("="*70)

fc_aleatoria = encontrar_fracao_critica(
    resultados_estrategias['aleatoria']['fracoes'],
    resultados_estrategias['aleatoria']['resultados']
)
fc_hubs = encontrar_fracao_critica(
    resultados_estrategias['hubs']['fracoes'],
    resultados_estrategias['hubs']['resultados']
)
fc_vizinhos = encontrar_fracao_critica(
    resultados_estrategias['vizinhos']['fracoes'],
    resultados_estrategias['vizinhos']['resultados']
)

print(f"\nFracao critica (limiar 10% de epidemias persistentes):")
print(f"  Estrategia Aleatoria: {fc_aleatoria:.2%} ({int(fc_aleatoria*n_nodes)} nos)")
print(f"  Estrategia Hubs:      {fc_hubs:.2%} ({int(fc_hubs*n_nodes)} nos)")
print(f"  Estrategia Vizinhos:  {fc_vizinhos:.2%} ({int(fc_vizinhos*n_nodes)} nos)")

print(f"\nEficiencia relativa (comparada com aleatoria):")
if fc_hubs > 0:
    print(f"  Hubs:     {fc_aleatoria/fc_hubs:.2f}x mais eficiente")
if fc_vizinhos > 0:
    print(f"  Vizinhos: {fc_aleatoria/fc_vizinhos:.2f}x mais eficiente")

# Salva resultados
salvar_resultados({
    'beta': BETA,
    'mu': MU,
    'R0': R0,
    'estrategias': resultados_estrategias,
    'fracoes_criticas': {
        'aleatoria': fc_aleatoria,
        'hubs': fc_hubs,
        'vizinhos': fc_vizinhos
    },
    'tempo_execucao': tempo_total,
    'modo': 'RAPIDO'
}, 'results/Q3_results.pkl')

print("\n" + "="*70)
print(" Analise: Relacao com Robustez de Redes")
print("="*70)
print("""
COMPARACAO DAS ESTRATEGIAS:

1. IMUNIZACAO ALEATORIA:
   - Requer a maior fracao de nos imunizados
   - Ineficiente em redes heterogeneas

2. IMUNIZACAO DE HUBS:
   - A estrategia MAIS EFICIENTE
   - Remove os nos mais conectados

3. IMUNIZACAO DE VIZINHOS (Acquaintance):
   - Eficiencia intermediaria
   - Nao requer conhecimento completo da rede

PARADOXO DAS REDES SCALE-FREE:
- ROBUSTAS a falhas aleatorias
- VULNERAVEIS a ataques direcionados

IMPLICACOES:
- Vacinacao: priorizar individuos com muitos contatos
- Super-espalhadores sao criticos para controle
""")

print("\nResultados salvos em:")
print("  - results/Q3_results.pkl")
print("  - results/figures/Q3_comparacao_estrategias.png")
