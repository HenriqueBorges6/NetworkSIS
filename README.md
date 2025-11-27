# Trabalho Final - Simulação de Epidemias

Implementação do modelo SIS (Susceptible-Infected-Susceptible) em redes complexas.

## Como Executar

### Dependências

```bash
pip install -r requirements.txt
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
## Resultados

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
