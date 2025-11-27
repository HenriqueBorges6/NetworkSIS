# Trabalho Final - Simulação de Epidemias

Implementação do modelo SIS (Susceptible-Infected-Susceptible) em redes complexas.

## 📁 Estrutura do Projeto

```
Trabalho/
├── model.py              # Funções do modelo SIS
├── utils.py              # Funções auxiliares (plots, estatísticas)
├── network_gen.py        # Geração de redes (ER, Scale-Free)
├── Q1.py                 # Questão 1 - Rede Erdős-Rényi
├── Q2.py                 # Questão 2 - Rede Scale-Free
├── Q3.py                 # Questão 3 - Estratégias de Imunização
├── results/              # Pasta com resultados
│   ├── *.pkl            # Dados salvos
│   └── figures/         # Gráficos gerados
└── simulacao_epidemias.ipynb  # Notebook Jupyter completo
```

## 🚀 Como Executar

### Dependências

```bash
pip install numpy networkx matplotlib pandas scipy tqdm
```

### Executar as Questões

Execute os scripts na ordem:

```bash
# Questão 1: Rede ER
python Q1.py

# Questão 2: Rede Scale-Free
python Q2.py

# Questão 3: Estratégias de Imunização (requer Q2)
python Q3.py
```

### Ou usar o Notebook Jupyter

```bash
jupyter notebook simulacao_epidemias.ipynb
```

## 📊 Questões

### Questão 1: Rede Erdős-Rényi (ER)
- **Rede**: 10000 nós, grau médio <k> = 20
- **Parâmetros**:
  - a) β=0.02, μ=0.1 (R₀=4)
  - b) β=0.02, μ=0.4 (R₀=1)
  - c) β=0.02, μ=0.5 (R₀=0.8)
- **Objetivo**: Verificar limiar epidêmico R₀=1

### Questão 2: Rede Scale-Free
- **Rede**: 10000 nós, grau médio <k> = 20, γ=2.5
- **Parâmetros**:
  - a) β=0.01, μ=0.1 (R₀=2)
  - b) β=0.01, μ=0.2 (R₀=1)
  - c) β=0.01, μ=0.3 (R₀=0.67)
- **Objetivo**: Comparar com rede ER

### Questão 3: Estratégias de Imunização
- **Base**: Parâmetros Q2a (β=0.01, μ=0.1)
- **Estratégias**:
  - a) Imunização aleatória
  - b) Imunização de hubs (maior grau)
  - c) Imunização de vizinhos (acquaintance immunization)
- **Objetivo**: Encontrar fração crítica de vacinação

## 📈 Resultados

Todos os resultados são salvos em:
- **Dados**: `results/*.pkl` (formato pickle)
- **Figuras**: `results/figures/*.png`

### Carregar Resultados

```python
from utils import carregar_resultados

dados_Q1 = carregar_resultados('results/Q1_results.pkl')
dados_Q2 = carregar_resultados('results/Q2_results.pkl')
dados_Q3 = carregar_resultados('results/Q3_results.pkl')
```

## 🧮 Modelo SIS

### Equações

**Taxa de infecção**: β
**Taxa de recuperação**: μ
**Número básico de reprodução**: R₀ = (β × <k>) / μ

### Estados dos Nós
- **0**: Suscetível (S) - pode ser infectado
- **1**: Infectado (I) - pode infectar vizinhos
- **-1**: Imunizado - vacinado, não pode ser infectado

### Dinâmica
1. **Infecção**: Suscetível com k_inf vizinhos infectados
   → Probabilidade de infecção: 1 - (1-β)^k_inf

2. **Recuperação**: Infectado
   → Probabilidade μ de voltar para Suscetível

## 📚 Referências

- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
- Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks.

## 👥 Autores

Trabalho Final - Ciência de Redes 2025

## 📝 Licença

Código disponível para fins acadêmicos.
