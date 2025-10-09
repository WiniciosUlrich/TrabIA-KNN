"""
Script de demonstração para o Grupo de Dados 3
===============================================

Dataset: Grupo 3 (dataset genérico)
Objetivo: Classificação em 3 classes
Características: 2
"""

import numpy as np
import scipy.io as scipy
from meuKnn import meuKnn
from visualizaPontos import visualizaPontos

# Carregar os dados do arquivo .mat
print("=" * 60)
print("GRUPO DE DADOS 3")
print("=" * 60)

mat = scipy.loadmat('grupoDados3.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots'].flatten()
trainRots = mat['trainRots'].flatten()

print(f"\nDados carregados:")
print(f"  Treinamento: {grupoTrain.shape} (exemplos x características)")
print(f"  Teste: {grupoTest.shape} (exemplos x características)")
print(f"  Classes: {np.unique(trainRots)}")
print(f"\n  OBSERVAÇÃO: Dataset com apenas 2 características")

# ============================================================================
# Q3.1: Aplique o kNN com k=1. Qual é a acurácia?
# ============================================================================

print("\n" + "-" * 60)
print("Q3.1: APLICANDO k-NN COM k=1")
print("-" * 60)

# Testar com k=1 conforme solicitado
rotuloPrevisto_k1 = meuKnn(grupoTrain, trainRots, grupoTest, 1)
estaCorreto_k1 = rotuloPrevisto_k1 == testRots
numCorreto_k1 = np.sum(estaCorreto_k1)
totalNum = len(testRots)
acuracia_k1 = numCorreto_k1 / totalNum

print(f"Resultado com k=1:")
print(f"  Exemplos corretos: {numCorreto_k1}/{totalNum}")
print(f"  Acurácia: {acuracia_k1:.4f} ({acuracia_k1*100:.2f}%)")

print(f"\n✓ RESPOSTA Q3.1:")
print(f"  Acurácia com k=1: {acuracia_k1:.4f} ({acuracia_k1*100:.2f}%)")
print(f"  PROBLEMA: Acurácia baixa (~70%)")

# ============================================================================
# Q3.2: Ajuste para obter 92% de acurácia
# ============================================================================

print("\n" + "-" * 60)
print("Q3.2: INVESTIGANDO O PROBLEMA E AJUSTANDO k")
print("-" * 60)

print("\nDIAGNÓSTICO DO PROBLEMA:")
print("  1. k=1 é muito sensível a outliers e ruído nos dados")
print("  2. Um único ponto de treinamento ruidoso pode causar erro")
print("  3. Não há suavização das decisões")
print("\nSOLUÇÃO: Aumentar o valor de k!")
print("  - k maior usa votação de múltiplos vizinhos")
print("  - Isso suaviza as decisões e reduz sensibilidade a outliers")
print("  - Encontrar o k ótimo através de testes")

print("\n" + "-" * 60)
print("TESTANDO DIFERENTES VALORES DE k")
print("-" * 60)

# Testar com diferentes valores de k
valores_k = [1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 20]
acuracias = {}

print("Resultados:")
for k in valores_k:
    rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k)
    estaCorreto = rotuloPrevisto == testRots
    numCorreto = np.sum(estaCorreto)
    acuracia = numCorreto / totalNum
    acuracias[k] = acuracia
    
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)", end="")
    if acuracia >= 0.92:
        print(" ✓ META ATINGIDA!")
    else:
        print()

# Encontrar melhor resultado
melhor_k = max(acuracias, key=acuracias.get)
melhor_acuracia = acuracias[melhor_k]

print(f"\n✓ RESPOSTA Q3.2:")
print(f"  Melhor acurácia: {melhor_acuracia:.4f} ({melhor_acuracia*100:.2f}%)")
print(f"  Obtida com k={melhor_k}")

if melhor_acuracia >= 0.92:
    print(f"  ✓ META DE 92% ATINGIDA!")
else:
    print(f"  ⚠ Meta de 92% não atingida")

print(f"\n  MELHORIA: {(melhor_acuracia - acuracia_k1)*100:.2f} pontos percentuais")

# ============================================================================
# ANÁLISE DA MELHORIA
# ============================================================================

print("\n" + "=" * 60)
print("ANÁLISE DA MELHORIA COM DIFERENTES k")
print("=" * 60)

# Encontrar os k's que atingiram 92%
k_acima_92 = [k for k, acc in acuracias.items() if acc >= 0.92]

if k_acima_92:
    print(f"\nValores de k que atingiram ≥92%: {k_acima_92}")
    print("\nObservações:")
    print(f"  - k muito pequeno (k=1): {acuracias[1]*100:.2f}% - sensível a ruído")
    print(f"  - k médio (k={k_acima_92[0]}): {acuracias[k_acima_92[0]]*100:.2f}% - bom equilíbrio")
    if len(k_acima_92) > 1:
        print(f"  - k maior (k={k_acima_92[-1]}): {acuracias[k_acima_92[-1]]*100:.2f}% - ainda melhor")

# ============================================================================
# VISUALIZAÇÃO DOS DADOS
# ============================================================================

print("\n" + "-" * 60)
print("VISUALIZAÇÃO DOS DADOS")
print("-" * 60)
print("Gerando gráfico de dispersão...")
print("(Feche a janela do gráfico para continuar)")
print("(Como há apenas 2 características, visualizamos ambas)")

# Visualizar as 2 dimensões
visualizaPontos(grupoTrain, trainRots, 0, 1)

# ============================================================================
# CONCLUSÕES
# ============================================================================

print("\n" + "=" * 60)
print("CONCLUSÕES GRUPO 3:")
print("=" * 60)
print("O QUE FOI FEITO:")
print(f"  - Aumentou-se o valor de k de 1 para {melhor_k}")
print("\nPOR QUÊ:")
print("  - k=1 (1-NN) é muito sensível a outliers e ruído:")
print("    • Se um único ponto de treinamento for ruidoso ou mal rotulado,")
print("      ele pode causar erro na classificação")
print("    • Não há mecanismo de suavização ou consenso")
print(f"  - k={melhor_k} (k-NN) é mais robusto:")
print("    • Usa votação de múltiplos vizinhos (votação por maioria)")
print("    • Outliers têm menos influência na decisão final")
print("    • Suaviza as fronteiras de decisão entre classes")
print("    • Reduz overfitting aos dados de treinamento")
print("\nRESULTADO:")
print(f"  - Acurácia aumentou de {acuracia_k1*100:.2f}% para {melhor_acuracia*100:.2f}%")
print(f"  - Melhoria de {(melhor_acuracia - acuracia_k1)*100:.2f} pontos percentuais")
if melhor_acuracia >= 0.92:
    print("  - ✓ Meta de 92% atingida!")
print("\nPRINCÍPIO IMPORTANTE:")
print("  - k muito pequeno: overfitting, sensível a ruído")
print("  - k muito grande: underfitting, fronteiras muito suaves")
print(f"  - k ótimo ({melhor_k}): bom equilíbrio entre viés e variância")
print("=" * 60)
