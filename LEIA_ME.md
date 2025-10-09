# Classificador k-Nearest Neighbour (k-NN) - Guia Rápido

## 📁 Arquivos do Projeto

### 4 Funções (conforme especificação):
1. **`dist.py`** - Distância Euclidiana
2. **`meuKnn.py`** - Classificador k-NN
3. **`visualizaPontos.py`** - Visualização de dados
4. **`normalizacao.py`** - Normalização de dados

### 4 Scripts de Demonstração (com todas as respostas):
1. **`demoD1.py`** - Grupo 1 (Iris)
2. **`demoD2.py`** - Grupo 2 (Wine)
3. **`demoD3.py`** - Grupo 3
4. **`demoD4.py`** - Grupo 4

## 🚀 Como Executar

### Opção 1: Menu Interativo
```bash
python main_menu.py
```

### Opção 2: Executar cada demonstração
```bash
python demoD1.py
python demoD2.py
python demoD3.py
python demoD4.py
```

## ✅ Respostas das Questões

### Grupo 1 (Iris)
- **Q1.1:** Acurácia máxima = **98.00%** (k=3)
- **Q1.2:** **SIM**, todas as 4 características são necessárias

### Grupo 2 (Wine)
- **Q2.1:** Acurácia = **68.33%** (sem normalização)
- **Q2.2:** Acurácia = **98.33%** (k=3, COM normalização)
  - **Solução:** Normalização (StandardScaler)
  - **Por quê:** Escalas muito diferentes entre características

### Grupo 3
- **Q3.1:** Acurácia = **70.00%** (k=1)
- **Q3.2:** Acurácia = **96.00%** (k=10)
  - **Solução:** Aumentar k de 1 para 10
  - **Por quê:** k=1 muito sensível a outliers

### Grupo 4
- **Q4.1:** Acurácia = **71.67%** (k=1)
- **Q4.2:** Acurácia = **90.00%** (k=9, normalização)
  - **Soluções:** Normalização + ajuste de k
  - **Por quê:** Múltiplos problemas (escala + ruído)

## 📝 Observação
Todos os scripts contêm comentários detalhados explicando:
- O que foi feito
- Por que foi feito
- Resultados obtidos
- Conclusões

Veja `README_COMPLETO.md` para documentação completa.
