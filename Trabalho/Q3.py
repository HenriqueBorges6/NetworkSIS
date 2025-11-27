"""
Q3.py - Questão 3: Estratégias de Imunização

Descobre o número de vértices imunizados previamente necessários para
impedir a fixação do estado endêmico com diferentes estratégias de vacinação.

Usa parâmetros da Q2a: β=0.01, μ=0.1
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import modelo_sis, calcular_R0
from utils import carregar_resultados, salvar_resultados

# Configurações
np.random.seed(42)
BETA = 0.01
MU = 0.1
N_SIMULACOES_POR_FRACAO = 50
MAX_STEPS = 500
N_INICIAL_INFECTADOS = 5

# Cria pasta para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*70)
print(" QUESTÃO 3: Estratégias de Imunização")
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
print(f"  Nós: {n_nodes}")
print(f"  <k>: {k_medio:.2f}")
print(f"  β={BETA}, μ={MU}, R₀={R0:.2f}")


def encontrar_fracao_critica_imunizacao(G, beta, mu, estrategia,
                                        n_simulacoes=50, max_steps=500):
    """
    Encontra a fração crítica de imunização para cada estratégia.

    Estratégias:
    - 'aleatoria': escolhe nós aleatoriamente
    - 'hubs': escolhe nós com maior grau
    - 'vizinhos': escolhe vizinhos de nós aleatórios (acquaintance immunization)
    """
    n_nodes = G.number_of_nodes()
    fracoes = np.linspace(0, 0.5, 20)
    resultados = []

    for fracao in tqdm(fracoes, desc=f"Testando {estrategia}"):
        n_imunizar = int(fracao * n_nodes)

        if n_imunizar == 0:
            resultados.append(1.0)  # Sem imunização, epidemia persiste
            continue

        epidemias_persistentes = 0

        for _ in range(n_simulacoes):
            # Seleciona nós para imunizar de acordo com a estratégia
            if estrategia == 'aleatoria':
                imunizados = set(np.random.choice(list(G.nodes()),
                                                  size=n_imunizar,
                                                  replace=False))

            elif estrategia == 'hubs':
                # Ordena por grau e seleciona os maiores
                nos_ordenados = sorted(G.degree(), key=lambda x: x[1], reverse=True)
                imunizados = set([n for n, d in nos_ordenados[:n_imunizar]])

            elif estrategia == 'vizinhos':
                # Acquaintance immunization
                # Escolhe nós aleatórios e imuniza seus vizinhos
                imunizados = set()
                nos_amostra = np.random.choice(list(G.nodes()),
                                              size=min(n_imunizar * 2, n_nodes),
                                              replace=False)

                for node in nos_amostra:
                    vizinhos = list(G.neighbors(node))
                    if vizinhos and len(imunizados) < n_imunizar:
                        vizinho = np.random.choice(vizinhos)
                        imunizados.add(vizinho)

                    if len(imunizados) >= n_imunizar:
                        break

            # Executa simulação
            nos_disponiveis = [n for n in G.nodes() if n not in imunizados]
            if len(nos_disponiveis) < N_INICIAL_INFECTADOS:
                continue

            inicial_infectados = np.random.choice(nos_disponiveis,
                                                 size=N_INICIAL_INFECTADOS,
                                                 replace=False)
            historico = modelo_sis(G, beta, mu, inicial_infectados,
                                  max_steps, imunizados)

            # Considera persistente se houver infectados no final
            if len(historico) > 0 and historico[-1] > 0:
                epidemias_persistentes += 1

        fracao_persistente = epidemias_persistentes / n_simulacoes
        resultados.append(fracao_persistente)

    return fracoes, resultados


# Testa as três estratégias
print("\n" + "="*70)
print(" Testando Estratégias de Imunização")
print("="*70)

estrategias = ['aleatoria', 'hubs', 'vizinhos']
resultados_estrategias = {}

for estrategia in estrategias:
    print(f"\nEstrategia: {estrategia}")
    fracoes, resultados = encontrar_fracao_critica_imunizacao(
        G_SF, BETA, MU, estrategia,
        n_simulacoes=N_SIMULACOES_POR_FRACAO,
        max_steps=MAX_STEPS
    )
    resultados_estrategias[estrategia] = {
        'fracoes': fracoes,
        'resultados': resultados
    }

# Plota comparação
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

plt.xlabel('Fração de Nós Imunizados', fontsize=12)
plt.ylabel('Fração de Epidemias Persistentes', fontsize=12)
plt.title(f'Eficácia de Diferentes Estratégias de Imunização\n' +
          f'Rede Scale-Free (β={BETA}, μ={MU}, R₀={R0:.2f})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/Q3_comparacao_estrategias.png', dpi=300)
plt.show()


# Calcula fração crítica para cada estratégia
def encontrar_fracao_critica(fracoes, resultados, limiar=0.1):
    """Encontra a fração onde menos de 10% das epidemias persistem."""
    for f, r in zip(fracoes, resultados):
        if r < limiar:
            return f
    return fracoes[-1]


print("\n" + "="*70)
print(" Resultados: Fração Crítica de Imunização")
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

print(f"\nFração crítica (limiar 10% de epidemias persistentes):")
print(f"  Estratégia Aleatória: {fc_aleatoria:.2%} ({int(fc_aleatoria*n_nodes)} nós)")
print(f"  Estratégia Hubs:      {fc_hubs:.2%} ({int(fc_hubs*n_nodes)} nós)")
print(f"  Estratégia Vizinhos:  {fc_vizinhos:.2%} ({int(fc_vizinhos*n_nodes)} nós)")

print(f"\nEficiência relativa (comparada com aleatória):")
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
    }
}, 'results/Q3_results.pkl')

# Análise teórica
print("\n" + "="*70)
print(" Análise: Relação com Robustez de Redes")
print("="*70)
print("""
COMPARAÇÃO DAS ESTRATÉGIAS:

1. IMUNIZAÇÃO ALEATÓRIA:
   - Requer a maior fração de nós imunizados
   - Ineficiente em redes heterogêneas (não considera estrutura)
   - Análoga a falhas aleatórias em redes

2. IMUNIZAÇÃO DE HUBS:
   - A estratégia MAIS EFICIENTE
   - Requer fração muito menor de nós imunizados
   - Remove os nós mais conectados, fragmentando rapidamente a rede
   - Análoga a ataques direcionados

3. IMUNIZAÇÃO DE VIZINHOS (Acquaintance Immunization):
   - Eficiência intermediária, superior à aleatória
   - Vantagem prática: não requer conhecimento completo da rede
   - Baseada no "paradoxo da amizade": seus amigos têm mais amigos que você

RELAÇÃO COM ROBUSTEZ:

Estes resultados ilustram a VULNERABILIDADE DE REDES SCALE-FREE:

• ROBUSTAS a falhas aleatórias:
  - Remoção aleatória de nós requer alta fração para impactar epidemias
  - A maioria dos nós tem grau baixo, remoção não afeta muito a rede

• VULNERÁVEIS a ataques direcionados:
  - Remoção de hubs fragmenta rapidamente a rede
  - Poucos nós removidos impedem epidemias eficientemente

PARADOXO DAS REDES SCALE-FREE:
São simultaneamente robustas a perturbações aleatórias mas extremamente
vulneráveis a ataques direcionados aos hubs. O mesmo princípio se aplica
tanto à conectividade estrutural quanto à propagação de epidemias.

IMPLICAÇÕES PRÁTICAS:
- Campanhas de vacinação: priorizar indivíduos com muitos contatos
- Super-espalhadores são alvos críticos para controle epidêmico
- Proteção de infraestrutura crítica: focar nos hubs da rede
""")

print("\nResultados salvos em:")
print("  - results/Q3_results.pkl")
print("  - results/figures/Q3_comparacao_estrategias.png")
