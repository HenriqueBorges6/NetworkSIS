"""
Q2.py - Questão 2: Simulações em Rede Scale-Free

Gera uma rede livre de escala com 10000 vértices, grau médio <k> = 20
e expoente γ = 2.5. Compara com resultados da Questão 1.
"""

import numpy as np
import os
from model import executar_multiplas_simulacoes, calcular_R0
from network_gen import gerar_rede_scale_free, verificar_expoente_lei_potencia
from utils import (plotar_simulacoes, imprimir_estatisticas,
                   plotar_distribuicao_graus, salvar_resultados, carregar_resultados)

# Configurações
np.random.seed(42)
N_NODES = 10000
K_MEDIO = 20
GAMMA = 2.5
N_SIMULACOES = 100
N_INICIAL_INFECTADOS = 5
MAX_STEPS = 1000

# Cria pasta para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*70)
print(" QUESTÃO 2: Rede Scale-Free")
print("="*70)

# Gera rede Scale-Free
G_SF = gerar_rede_scale_free(N_NODES, K_MEDIO, GAMMA, seed=42)
graus = [d for n, d in G_SF.degree()]
k_medio_real = np.mean(graus)

# Verifica expoente
verificar_expoente_lei_potencia(G_SF)

# Plota distribuição de graus
plotar_distribuicao_graus(
    G_SF,
    titulo=f"Rede Scale-Free (γ={GAMMA})",
    log_scale=True,
    salvar='results/figures/Q2_distribuicao_graus.png'
)

# Parâmetros para as simulações
parametros = [
    {'beta': 0.01, 'mu': 0.1, 'nome': 'Q2a'},
    {'beta': 0.01, 'mu': 0.2, 'nome': 'Q2b'},
    {'beta': 0.01, 'mu': 0.3, 'nome': 'Q2c'},
]

resultados_Q2 = {}

for params in parametros:
    beta = params['beta']
    mu = params['mu']
    nome = params['nome']

    R0 = calcular_R0(beta, k_medio_real, mu)

    print(f"\n{'='*70}")
    print(f" {nome}: β={beta}, μ={mu}, R₀={R0:.2f}")
    print(f"{'='*70}")

    # Executa simulações
    historicos = executar_multiplas_simulacoes(
        G_SF, beta, mu,
        n_inicial_infectados=N_INICIAL_INFECTADOS,
        n_simulacoes=N_SIMULACOES,
        max_steps=MAX_STEPS
    )

    # Plota resultados
    titulo = f"{nome} - Rede Scale-Free"
    plotar_simulacoes(
        historicos, titulo, beta, mu, k_medio_real,
        salvar=f'results/figures/{nome}_simulacoes.png'
    )

    # Imprime estatísticas
    imprimir_estatisticas(historicos, titulo, beta, mu, k_medio_real)

    # Armazena resultados
    resultados_Q2[nome] = {
        'beta': beta,
        'mu': mu,
        'R0': R0,
        'historicos': historicos,
        'k_medio': k_medio_real
    }

# Salva todos os resultados
salvar_resultados({
    'rede': G_SF,
    'k_medio': k_medio_real,
    'gamma': GAMMA,
    'resultados': resultados_Q2
}, 'results/Q2_results.pkl')

print("\n" + "="*70)
print(" Análise Questão 2 - Comparação com Questão 1")
print("="*70)
print("""
DIFERENÇAS ENTRE REDE ER E SCALE-FREE:

1. DISTRIBUIÇÃO DE GRAUS:
   - Rede ER: distribuição aproximadamente Poisson (homogênea)
   - Rede Scale-Free: distribuição lei de potência (heterogênea, hubs)

2. LIMIAR EPIDÊMICO:
   - Rede ER: limiar epidêmico em R₀ = 1
   - Rede Scale-Free: limiar epidêmico efetivamente ausente (próximo de 0)
                       devido aos hubs

3. COMPORTAMENTO DA EPIDEMIA:
   - Em redes scale-free, os hubs (nós com muitas conexões) facilitam
     a propagação
   - Mesmo com R₀ < 1 (campo médio), a epidemia pode persistir devido
     à heterogeneidade
   - A presença de hubs torna a rede mais vulnerável a epidemias

4. OBSERVAÇÕES:
   - Na rede scale-free, epidemias tendem a ter maior alcance e duração
   - A variabilidade entre simulações é maior na rede scale-free
   - Os hubs atuam como "super-espalhadores"
""")

# Carrega resultados Q1 para comparação
try:
    dados_Q1 = carregar_resultados('results/Q1_results.pkl')
    print("\nCOMPARAÇÃO QUANTITATIVA:")
    print("(comparar manualmente os resultados de Q1 vs Q2)")
except:
    print("\nExecute Q1.py primeiro para gerar comparações.")

print("\nResultados salvos em:")
print("  - results/Q2_results.pkl")
print("  - results/figures/Q2_*.png")
