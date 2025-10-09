# Classificador k-Nearest Neighbour (k-NN)

Este projeto implementa um classificador k-Nearest Neighbour (k-NN) do zero usando Python e testa em quatro grupos de dados diferentes.

## 📋 Descrição

O classificador k-NN é um dos algoritmos de classificação mais simples e intuitivos. Ele classifica um novo exemplo baseado nos k exemplos de treinamento mais próximos usando distância Euclidiana.

## 📁 Estrutura do Projeto

### Funções Principais (conforme especificação):
- **`dist.py`** - Função para calcular distância Euclidiana entre dois pontos
- **`meuKnn.py`** - Implementação do classificador k-NN
- **`visualizaPontos.py`** - Funções para visualizar dados em gráficos 2D
- **`normalizacao.py`** - Função para normalizar dados (StandardScaler)

### Scripts de Demonstração (conforme especificação):
- **`demoD1.py`** - Demonstração com Grupo de Dados 1 (Iris Dataset)
- **`demoD2.py`** - Demonstração com Grupo de Dados 2 (Wine Dataset)
- **`demoD3.py`** - Demonstração com Grupo de Dados 3
- **`demoD4.py`** - Demonstração com Grupo de Dados 4

### Scripts Auxiliares:
- **`main_menu.py`** - Menu interativo para executar as demonstrações
- **`main.py`** - Script de análise completa integrada (opcional)
- **`exemplos_uso.py`** - Exemplos adicionais de uso (opcional)

### Datasets:
- `grupoDados1.mat` - Dataset Iris (flores)
- `grupoDados2.mat` - Dataset Wine (vinhos)
- `grupoDados3.mat` - Dataset genérico
- `grupoDados4.mat` - Dataset genérico

## 🚀 Como Usar

### Opção 1: Menu Interativo (Recomendado)
```bash
python main_menu.py
```
Escolha qual demonstração deseja executar ou execute todas em sequência.

### Opção 2: Executar Demonstrações Individualmente
```bash
# Grupo 1 - Iris Dataset
python demoD1.py

# Grupo 2 - Wine Dataset
python demoD2.py

# Grupo 3
python demoD3.py

# Grupo 4
python demoD4.py
```

### Opção 3: Usar as Funções Diretamente
```python
from meuKnn import meuKnn
from dist import dist
import scipy.io as scipy
import numpy as np

# Carregar dados
mat = scipy.loadmat('grupoDados1.mat')
grupo_test = mat['grupoTest']
grupo_train = mat['grupoTrain']
test_rots = mat['testRots'].flatten()
train_rots = mat['trainRots'].flatten()

# Fazer predição com k=3
predicoes = meuKnn(grupo_train, train_rots, grupo_test, k=3)

# Calcular acurácia
acuracia = np.sum(predicoes == test_rots) / len(test_rots)
print(f"Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")
```

## 📊 Resultados Obtidos (Respostas às Questões)

### Grupo 1 (Iris Dataset)
- **Q1.1:** Acurácia máxima = **98.00%** (k=3)
- **Q1.2:** SIM, todas as 4 características são necessárias para acurácia máxima
  - Com 4 características: 98.00%
  - Com 3 características: ~96.00%
  - Com 2 características: 70-96%

**Observação:** Características 2 e 3 (relacionadas às pétalas) são mais discriminativas que 0 e 1 (sépalas).

---

### Grupo 2 (Wine Dataset)
- **Q2.1:** Acurácia inicial = **68.33%** (k=1, sem normalização)
- **Q2.2:** Acurácia de **98.33%** atingida (k=3, COM normalização)

**O que foi feito:**
- Aplicou-se normalização StandardScaler aos dados
- Testou-se diferentes valores de k

**Por quê:**
- As 13 características químicas têm escalas muito diferentes
  - Álcool: ~12-14
  - Prolina: ~500-1600
  - Magnésio: ~70-160
- Sem normalização, características com valores maiores dominam a distância Euclidiana
- StandardScaler coloca todas as características na mesma escala (média=0, desvio=1)
- Isso permite que todas contribuam igualmente para a classificação

**Resultado:**
- Melhoria de ~30 pontos percentuais
- ✓ Meta de 98% atingida!

---

### Grupo 3
- **Q3.1:** Acurácia com k=1 = **70.00%**
- **Q3.2:** Acurácia de **96.00%** atingida (k=10)

**O que foi feito:**
- Aumentou-se o valor de k de 1 para 10

**Por quê:**
- k=1 (1-NN) é muito sensível a outliers e ruído
  - Um único ponto ruidoso pode causar erro
  - Não há mecanismo de suavização
- k=10 (k-NN) é mais robusto
  - Usa votação de 10 vizinhos (votação por maioria)
  - Outliers têm menos influência
  - Suaviza as fronteiras de decisão
  - Reduz overfitting

**Resultado:**
- Melhoria de ~26 pontos percentuais
- ✓ Meta de 92% atingida (inclusive superada: 96%)!

---

### Grupo 4
- **Q4.1:** Acurácia inicial = **71.67%** (k=1, sem normalização)
- **Q4.2:** Máxima acurácia = **90.00%** (k=9, StandardScaler)

**O que foi feito:**
1. Normalização dos dados (StandardScaler, MinMaxScaler, RobustScaler)
2. Ajuste do valor de k (testado k de 1 a 15)
3. Teste de múltiplos métodos de normalização

**Por quê:**
- **PROBLEMA 1 - Escalas diferentes:**
  - Características têm magnitudes diferentes
  - Distância Euclidiana é sensível a escala
  - Normalização equaliza a contribuição

- **PROBLEMA 2 - Valor de k inadequado:**
  - k=1 muito sensível a ruído
  - k muito alto suaviza demais
  - k ótimo encontrado: 9

- **PROBLEMA 3 - Possíveis outliers:**
  - StandardScaler usa média (sensível a outliers)
  - RobustScaler usa mediana (robusto a outliers)
  - MinMaxScaler escala para [0,1]

**Resultado:**
- Melhoria de ~18 pontos percentuais
- ⚠ Meta de 92% não atingida (máximo: 90.00%)
- Possíveis limitações intrínsecas do dataset

---

## 🔧 Funções Implementadas

### 1. `dist(p, q)`
Calcula a distância Euclidiana entre dois pontos.

```python
from dist import dist
import numpy as np

p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
distancia = dist(p, q)  # Retorna: 5.196...
```

**Fórmula:** d(p,q) = √(Σ(pi - qi)²)

---

### 2. `meuKnn(dadosTrain, rotuloTrain, dadosTeste, k)`
Implementa o classificador k-NN.

**Parâmetros:**
- `dadosTrain`: Dados de treinamento (matriz)
- `rotuloTrain`: Rótulos dos dados de treinamento
- `dadosTeste`: Dados de teste
- `k`: Número de vizinhos mais próximos

**Retorna:** Array com predições

**Como funciona:**
1. Para cada exemplo de teste
2. Calcula distância para todos os exemplos de treinamento
3. Ordena as distâncias
4. Pega os k vizinhos mais próximos
5. Se k=1: retorna o rótulo do vizinho mais próximo
6. Se k>1: retorna a moda (votação) dos k vizinhos

---

### 3. `visualizaPontos(dados, rotulos, d1, d2)`
Visualiza dados em 2D.

**Parâmetros:**
- `dados`: Matriz de dados
- `rotulos`: Array de rótulos
- `d1, d2`: Índices das dimensões a visualizar

**Exemplo:**
```python
from visualizaPontos import visualizaPontos
import scipy.io as scipy

mat = scipy.loadmat('grupoDados1.mat')
dados = mat['grupoTrain']
rotulos = mat['trainRots'].flatten()

# Visualizar dimensões 0 e 1
visualizaPontos(dados, rotulos, 0, 1)
```

---

### 4. `normalizacao(dadosTrain, dadosTest)`
Normaliza os dados usando StandardScaler.

**Por que normalizar?**
- k-NN usa distância Euclidiana
- Características com valores grandes dominam
- Normalização equaliza as escalas

**Fórmula:** z = (x - média) / desvio_padrão

**Retorna:** Tupla (dadosTrain_normalizado, dadosTest_normalizado)

---

## 💡 Lições Aprendidas

1. **Normalização é CRUCIAL** para dados com escalas diferentes
   - Wine dataset: 68% → 98% com normalização
   
2. **k=1 é sensível a outliers**; k maior suaviza decisões
   - Grupo 3: 70% → 96% aumentando k
   
3. **Nem sempre é possível atingir metas altas** de acurácia
   - Grupo 4: máximo 90% (limitações do dataset)
   
4. **Qualidade dos dados** afeta significativamente os resultados
   
5. **Diferentes normalizações** podem ter resultados diferentes
   - StandardScaler, MinMaxScaler, RobustScaler

## 📈 Dependências

```bash
pip install numpy scipy matplotlib scikit-learn
```

**Bibliotecas utilizadas:**
- `numpy` - Operações numéricas
- `scipy.io` - Carregar arquivos .mat
- `matplotlib.pyplot` - Visualizações
- `scipy.stats` - Função mode (moda)
- `sklearn.preprocessing` - Normalização

## 🎯 Conceitos Importantes

### Distância Euclidiana
Medida de similaridade entre dois pontos no espaço.
Quanto menor a distância, mais similares os pontos.

### Votação por Maioria
Para k>1, o rótulo previsto é aquele que aparece mais vezes entre os k vizinhos.

### Normalização
Transforma dados para mesma escala, evitando que características com valores grandes dominem.

### Trade-off do k
- **k pequeno:** Mais flexível, mas sensível a ruído
- **k grande:** Mais robusto, mas pode suavizar demais
- **k ótimo:** Equilíbrio entre viés e variância

## 📝 Estrutura dos Comentários nos Scripts

Cada script de demonstração (`demoD1.py`, `demoD2.py`, etc.) contém:
- Descrição do dataset
- Carregamento e análise dos dados
- Implementação das questões propostas
- Explicações detalhadas do que foi feito e por quê
- Conclusões e lições aprendidas

---

**Autor:** Implementação para trabalho acadêmico  
**Data:** 2025  
**Linguagem:** Python 3.12+
