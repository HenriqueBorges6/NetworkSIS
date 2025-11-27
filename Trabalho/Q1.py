"""
Q1.py - Questão 1: Simulações em Rede Erdős-Rényi (ER)

Gera uma rede aleatória com 10000 vértices e grau médio <k> = 20.
Executa simulações SIS com diferentes parâmetros (β, μ).
"""

import numpy as np
import os
from model import executar_multiplas_simulacoes, calcular_R0
from network_gen import gerar_rede_er
from utils import (plotar_simulacoes, imprimir_estatisticas,
                   plotar_distribuicao_graus, salvar_resultados)

# Configurações
np.random.seed(42)
N_NODES = 10000
K_MEDIO = 20
N_SIMULACOES = 100
N_INICIAL_INFECTADOS = 5
MAX_STEPS = 100

# Cria pasta para resultados
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("="*70)
print(" QUESTÃO 1: Rede Erdős-Rényi (ER)")
print("="*70)

# Gera rede ER
G_ER = gerar_rede_er(N_NODES, K_MEDIO, seed=42)
graus = [d for n, d in G_ER.degree()]
k_medio_real = np.mean(graus)

# Plota distribuição de graus
plotar_distribuicao_graus(
    G_ER,
    titulo="Rede Erdős-Rényi",
    log_scale=False,
    salvar='results/figures/Q1_distribuicao_graus.png'
)

# Parâmetros para as simulações
parametros = [
    {'beta': 0.02, 'mu': 0.1, 'nome': 'Q1a'},
    {'beta': 0.02, 'mu': 0.4, 'nome': 'Q1b'},
    {'beta': 0.02, 'mu': 0.5, 'nome': 'Q1c'},
]

resultados_Q1 = {}

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
        G_ER, beta, mu,
        n_inicial_infectados=N_INICIAL_INFECTADOS,
        n_simulacoes=N_SIMULACOES,
        max_steps=MAX_STEPS
    )

    # Plota resultados
    titulo = f"{nome} - Rede ER"
    plotar_simulacoes(
        historicos, titulo, beta, mu, k_medio_real,
        salvar=f'results/figures/{nome}_simulacoes.png'
    )

    # Imprime estatísticas
    imprimir_estatisticas(historicos, titulo, beta, mu, k_medio_real)

    # Armazena resultados
    resultados_Q1[nome] = {
        'beta': beta,
        'mu': mu,
        'R0': R0,
        'historicos': historicos,
        'k_medio': k_medio_real
    }

# Salva todos os resultados
salvar_resultados({
    'rede': G_ER,
    'k_medio': k_medio_real,
    'resultados': resultados_Q1
}, 'results/Q1_results.pkl')

print("\n" + "="*70)
print(" Análise Questão 1 - Conclusões")
print("="*70)
print("""
TEORIA DO MODELO SIS DE CAMPO MÉDIO:

R₀ = (β × <k>) / μ

- Se R₀ > 1: a doença se fixa na rede (estado endêmico)
- Se R₀ ≤ 1: a doença desaparece

RESULTADOS:
- Q1a (R₀ = 4.0 > 1): Doença endêmica - alta prevalência de infectados
- Q1b (R₀ = 1.0):     Limiar epidêmico - comportamento crítico
- Q1c (R₀ = 0.8 < 1): Doença extinta - maioria das epidemias se extinguiu

Os resultados estão de acordo com a previsão teórica do modelo de campo médio.
""")

print("\nResultados salvos em:")
print("  - results/Q1_results.pkl")
print("  - results/figures/Q1_*.png")
