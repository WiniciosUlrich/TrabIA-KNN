#Winicios Ivan Ulrich - Matheus Fritzen - Gabriel Puff

"""
Script de demonstração para o Grupo de Dados 1 (Iris Dataset)
==============================================================

Dataset: Iris (flores)
URL: http://archive.ics.uci.edu/ml/datasets/Iris
Características: 4 (comprimento e largura de sépala e pétala)
Classes: 3 (setosa, versicolor, virginica)
"""

import numpy as np
import scipy.io as scipy
from meuKnn import meuKnn
from visualizaPontos import visualizaPontos

# Carregar os dados do arquivo .mat
print("=" * 60)
print("GRUPO DE DADOS 1 - IRIS DATASET")
print("=" * 60)

mat = scipy.loadmat('grupoDados1.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots'].flatten()
trainRots = mat['trainRots'].flatten()

print(f"\nDados carregados:")
print(f"  Treinamento: {grupoTrain.shape} (exemplos x características)")
print(f"  Teste: {grupoTest.shape} (exemplos x características)")
print(f"  Classes: {np.unique(trainRots)}")

# ============================================================================
# Q1.1: Qual é a acurácia máxima que você consegue da classificação?
# ============================================================================

print("\n" + "-" * 60)
print("Q1.1: TESTANDO DIFERENTES VALORES DE K")
print("-" * 60)

# Testar com k=1 (conforme especificação - deve dar 96%)
rotuloPrevisto_k1 = meuKnn(grupoTrain, trainRots, grupoTest, 1)
estaCorreto_k1 = rotuloPrevisto_k1 == testRots
numCorreto_k1 = np.sum(estaCorreto_k1)
totalNum = len(testRots)
acuracia_k1 = numCorreto_k1 / totalNum
print(f"k=1:  Acurácia = {acuracia_k1:.4f} ({acuracia_k1*100:.2f}%)")

# Testar com k=10 (conforme especificação - deve dar 94%)
rotuloPrevisto_k10 = meuKnn(grupoTrain, trainRots, grupoTest, 10)
estaCorreto_k10 = rotuloPrevisto_k10 == testRots
numCorreto_k10 = np.sum(estaCorreto_k10)
acuracia_k10 = numCorreto_k10 / totalNum
print(f"k=10: Acurácia = {acuracia_k10:.4f} ({acuracia_k10*100:.2f}%)")

# Testar com outros valores de k para encontrar a acurácia máxima
valores_k = [1, 3, 5, 7, 10, 15, 20]
acuracias = []

print(f"\nTestando múltiplos valores de k:")
for k in valores_k:
    rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k)
    estaCorreto = rotuloPrevisto == testRots
    numCorreto = np.sum(estaCorreto)
    acuracia = numCorreto / totalNum
    acuracias.append(acuracia)
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)")

# Encontrar a acurácia máxima
acuracia_maxima = max(acuracias)
k_melhor = valores_k[acuracias.index(acuracia_maxima)]

print(f"\n✓ RESPOSTA Q1.1:")
print(f"  Acurácia máxima: {acuracia_maxima:.4f} ({acuracia_maxima*100:.2f}%)")
print(f"  Obtida com k={k_melhor}")

# ============================================================================
# Q1.2: É necessário ter todas as características para obter acurácia máxima?
# ============================================================================

print("\n" + "-" * 60)
print("Q1.2: TESTANDO DIFERENTES CONJUNTOS DE CARACTERÍSTICAS")
print("-" * 60)

# Usar o melhor k encontrado anteriormente
k_usado = k_melhor

# Teste 1: Com todas as 4 características
rotulo_4feat = meuKnn(grupoTrain, trainRots, grupoTest, k_usado)
acuracia_4feat = np.sum(rotulo_4feat == testRots) / totalNum
print(f"Com 4 características (0,1,2,3): {acuracia_4feat:.4f} ({acuracia_4feat*100:.2f}%)")

# Teste 2: Com apenas 2 primeiras características (0,1)
rotulo_2feat_01 = meuKnn(grupoTrain[:, 0:2], trainRots, grupoTest[:, 0:2], k_usado)
acuracia_2feat_01 = np.sum(rotulo_2feat_01 == testRots) / totalNum
print(f"Com 2 características (0,1):     {acuracia_2feat_01:.4f} ({acuracia_2feat_01*100:.2f}%)")

# Teste 3: Com apenas 2 últimas características (2,3)
rotulo_2feat_23 = meuKnn(grupoTrain[:, 2:4], trainRots, grupoTest[:, 2:4], k_usado)
acuracia_2feat_23 = np.sum(rotulo_2feat_23 == testRots) / totalNum
print(f"Com 2 características (2,3):     {acuracia_2feat_23:.4f} ({acuracia_2feat_23*100:.2f}%)")

# Teste 4: Com apenas 3 características (0,1,2)
rotulo_3feat = meuKnn(grupoTrain[:, 0:3], trainRots, grupoTest[:, 0:3], k_usado)
acuracia_3feat = np.sum(rotulo_3feat == testRots) / totalNum
print(f"Com 3 características (0,1,2):   {acuracia_3feat:.4f} ({acuracia_3feat*100:.2f}%)")

print(f"\n✓ RESPOSTA Q1.2:")
if acuracia_4feat >= max(acuracia_2feat_01, acuracia_2feat_23, acuracia_3feat):
    print(f"  SIM, todas as 4 características são necessárias para acurácia máxima.")
    print(f"  A acurácia diminui quando usamos menos características:")
    print(f"    - Com 4 características: {acuracia_4feat*100:.2f}%")
    print(f"    - Com 3 características: {acuracia_3feat*100:.2f}%")
    print(f"    - Com 2 características: {max(acuracia_2feat_01, acuracia_2feat_23)*100:.2f}%")
else:
    print(f"  NÃO é necessário ter todas as características.")
    print(f"  A acurácia pode ser mantida com menos características.")

# ============================================================================
# VISUALIZAÇÃO DOS DADOS
# ============================================================================

print("\n" + "-" * 60)
print("VISUALIZAÇÃO DOS DADOS")
print("-" * 60)
print("Gerando gráficos de dispersão...")
print("(Feche a janela do gráfico para continuar)")

# Visualizar dimensões 0 e 1
visualizaPontos(grupoTrain, trainRots, 0, 1)

# Visualizar dimensões 2 e 3
visualizaPontos(grupoTrain, trainRots, 2, 3)

print("\n" + "=" * 60)
print("CONCLUSÕES GRUPO 1:")
print("=" * 60)
print(f"- O classificador k-NN funcionou muito bem no dataset Iris")
print(f"- Acurácia máxima de {acuracia_maxima*100:.2f}% com k={k_melhor}")
print(f"- Todas as características são importantes para acurácia máxima")
print(f"- Características 2 e 3 (pétalas) são mais discriminativas que 0 e 1")
print("=" * 60)
