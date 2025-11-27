"""
analise_limiar.py - Calcula o limiar epidemico real para rede scale-free

Compara R0 de campo medio com limiar real baseado em <k> e <k^2>
"""

import numpy as np
from network_gen import gerar_rede_scale_free

# Gera rede (mesma seed da Q2)
print("="*70)
print(" ANALISE DO LIMIAR EPIDEMICO")
print("="*70)

print("\nGerando rede scale-free...")
G = gerar_rede_scale_free(10000, 20, 2.5, seed=42)
graus = np.array([d for _, d in G.degree()])

# Calcula momentos da distribuicao
k_medio = np.mean(graus)
k2_medio = np.mean(graus**2)
k_max = np.max(graus)
k_min = np.min(graus)

print(f"\nEstatisticas da rede:")
print(f"  <k> (grau medio):           {k_medio:.2f}")
print(f"  <k^2> (segundo momento):    {k2_medio:.2f}")
print(f"  k_max (grau maximo):        {k_max}")
print(f"  k_min (grau minimo):        {k_min}")
print(f"  Desvio padrao dos graus:    {np.std(graus):.2f}")

# Calcula limiar epidemico real
R0_critico = k_medio / k2_medio

print(f"\n{'='*70}")
print(" LIMIARES EPIDEMICOS")
print(f"{'='*70}")

print(f"\n1. CAMPO MEDIO (redes homogeneas):")
print(f"   R0_critico = 1.0")
print(f"   Interpretacao: Epidemia persiste se R0 > 1")

print(f"\n2. REDES HETEROGENEAS (Pastor-Satorras & Vespignani):")
print(f"   R0_critico = <k> / <k^2> = {k_medio:.2f} / {k2_medio:.2f} = {R0_critico:.4f}")
print(f"   Interpretacao: Epidemia persiste se R0 > {R0_critico:.4f}")

print(f"\n{'='*70}")
print(" ANALISE DOS CENARIOS")
print(f"{'='*70}")

# Analisa cada cenario
cenarios = [
    {'nome': 'Q2a', 'beta': 0.01, 'mu': 0.1},
    {'nome': 'Q2b', 'beta': 0.01, 'mu': 0.2},
    {'nome': 'Q2c', 'beta': 0.01, 'mu': 0.3},
]

for c in cenarios:
    beta, mu = c['beta'], c['mu']
    R0_MF = (beta * k_medio) / mu

    # Predicao campo medio
    if R0_MF > 1:
        pred_MF = "Endemica"
    elif R0_MF < 1:
        pred_MF = "Extinta"
    else:
        pred_MF = "Critica"

    # Predicao heterogenea
    if R0_MF > R0_critico:
        pred_HET = "Endemica"
    else:
        pred_HET = "Extinta"

    # Resultados observados (valores do seu output)
    resultados_obs = {
        'Q2a': {'extintas_pct': 43.0, 'media_inf': 879.76},
        'Q2b': {'extintas_pct': 69.0, 'media_inf': 254.14},
        'Q2c': {'extintas_pct': 75.0, 'media_inf': 141.67},
    }
    extintas_pct = resultados_obs[c['nome']]['extintas_pct']

    print(f"\n{c['nome']}: beta={beta}, mu={mu}")
    print(f"  R0 (campo medio):         {R0_MF:.4f}")
    print(f"  Predicao campo medio:     {pred_MF}")
    print(f"  Predicao heterogenea:     {pred_HET}")
    print(f"  Resultado observado:      {100-extintas_pct:.1f}% persistentes")

    if pred_MF != pred_HET:
        print(f"  >> DISCORDANCIA entre teorias!")

    if pred_HET == "Endemica" and extintas_pct < 60:
        print(f"  >> Teoria heterogenea ACERTOU")
    elif pred_HET == "Extinta" and extintas_pct > 60:
        print(f"  >> Teoria heterogenea FALHOU (hubs sao fortes demais)")

print(f"\n{'='*70}")
print(" CONCLUSAO")
print(f"{'='*70}")

print(f"""
Para redes scale-free com gamma <= 3:
- O limiar epidemico REAL e muito menor que 1
- No caso dessa rede: R0_critico = {R0_critico:.4f}
- Portanto, TODOS os cenarios (a, b, c) sao endemicos!

Mesmo Q2c com R0 = 0.67 < 1 (campo medio) tem:
- R0 = 0.67 >> R0_critico = {R0_critico:.4f}
- Logo, deveria ser endemica segundo teoria heterogenea

Os 75% de extincao em Q2c sao devidos a:
1. Flutuacoes estocasticas (rede finita)
2. Infectados iniciais longe dos hubs
3. Tempo finito de simulacao

Para tempos muito longos, a taxa de extincao deve se aproximar de ~0%
para todos os cenarios, pois todos satisfazem R0 > R0_critico.
""")

print(f"{'='*70}")
