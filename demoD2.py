#Winicios Ivan Ulrich - Matheus Fritzen - Gabriel Puff

"""
Script de demonstração para o Grupo de Dados 2 (Wine Dataset)
==============================================================

Dataset: Wine (vinhos)
Objetivo: Prever a origem do vinho com base em componentes químicos
Características: 13 componentes químicos
Classes: 3 (3 diferentes origens)

Características:
1) Álcool
2) Ácido málico
3) Cinzas
4) Alcalinidade das cinzas
5) Magnésio
6) Fenóis totais
7) Flavonóides
8) Fenóis não flavonóides
9) Proantocianinas
10) Intensidade de cor
11) Tonalidade
12) OD280 / OD315 de vinhos diluídos
13) Prolina
"""

import numpy as np
import scipy.io as scipy
from meuKnn import meuKnn
from normalizacao import normalizacao

# Carregar os dados do arquivo .mat
print("=" * 60)
print("GRUPO DE DADOS 2 - WINE DATASET")
print("=" * 60)

mat = scipy.loadmat('grupoDados2.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots'].flatten()
trainRots = mat['trainRots'].flatten()

print(f"\nDados carregados:")
print(f"  Treinamento: {grupoTrain.shape} (exemplos x características)")
print(f"  Teste: {grupoTest.shape} (exemplos x características)")
print(f"  Classes: {np.unique(trainRots)}")

# Mostrar estatísticas dos dados para entender a escala 
# print(f"\nEstatísticas dos dados de treinamento:")
# print(f"  Mínimo por característica: {np.min(grupoTrain, axis=0)}")
# print(f"  Máximo por característica: {np.max(grupoTrain, axis=0)}")
# print(f"  Média por característica:  {np.mean(grupoTrain, axis=0)}")
# print(f"\n  OBSERVAÇÃO: As características têm escalas muito diferentes!")
# print(f"  - Álcool está em torno de 12-14")
# print(f"  - Prolina está em torno de 500-1600")
# print(f"  - Isso afeta o cálculo da distância Euclidiana!")

# ============================================================================
# Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia?
# ============================================================================

print("\n" + "-" * 60)
print("Q2.1: APLICANDO k-NN SEM NORMALIZAÇÃO")
print("-" * 60)

# Testar com diferentes valores de k
valores_k = [1, 3, 5, 7, 10, 15]
print("Testando diferentes valores de k (SEM normalização):")

acuracias_sem_norm = {}
for k in valores_k:
    rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k)
    estaCorreto = rotuloPrevisto == testRots
    numCorreto = np.sum(estaCorreto)
    totalNum = len(testRots)
    acuracia = numCorreto / totalNum
    acuracias_sem_norm[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)")

# Encontrar melhor resultado sem normalização
melhor_k_sem_norm = max(acuracias_sem_norm, key=acuracias_sem_norm.get)
melhor_acuracia_sem_norm = acuracias_sem_norm[melhor_k_sem_norm]

print(f"\n✓ RESPOSTA Q2.1:")
print(f"  Melhor acurácia SEM normalização: {melhor_acuracia_sem_norm:.4f} ({melhor_acuracia_sem_norm*100:.2f}%)")
print(f"  Obtida com k={melhor_k_sem_norm}")
print(f"  PROBLEMA: Acurácia muito baixa (~68-78%)")

# ============================================================================
# Q2.2: Ajuste para obter 98% de acurácia
# ============================================================================

print("\n" + "-" * 60)
print("Q2.2: INVESTIGANDO O PROBLEMA E APLICANDO NORMALIZAÇÃO")
print("-" * 60)

print("\nDIAGNÓSTICO DO PROBLEMA:")
print("  1. As características têm escalas muito diferentes")
print("  2. Características com valores maiores (ex: Prolina ~500-1600)")
print("     dominam o cálculo da distância Euclidiana")
print("  3. Características com valores menores (ex: Tonalidade ~0-3)")
print("     têm pouca influência na classificação")
print("\nSOLUÇÃO: Normalizar os dados!")
print("  - Usar StandardScaler para colocar todas as características")
print("    na mesma escala (média=0, desvio=1)")
print("  - Isso permite que todas as características contribuam")
print("    igualmente para o cálculo da distância")

# Normalizar os dados
grupoTrain_norm, grupoTest_norm = normalizacao(grupoTrain, grupoTest)

print("\n" + "-" * 60)
print("APLICANDO k-NN COM NORMALIZAÇÃO")
print("-" * 60)

print("Testando diferentes valores de k (COM normalização):")
acuracias_com_norm = {}
for k in valores_k:
    rotuloPrevisto = meuKnn(grupoTrain_norm, trainRots, grupoTest_norm, k)
    estaCorreto = rotuloPrevisto == testRots
    numCorreto = np.sum(estaCorreto)
    totalNum = len(testRots)
    acuracia = numCorreto / totalNum
    acuracias_com_norm[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)", end="")
    if acuracia >= 0.98:
        print(" ✓ META ATINGIDA!")
    else:
        print()

# Encontrar melhor resultado com normalização
melhor_k_com_norm = max(acuracias_com_norm, key=acuracias_com_norm.get)
melhor_acuracia_com_norm = acuracias_com_norm[melhor_k_com_norm]

print(f"\n✓ RESPOSTA Q2.2:")
print(f"  Melhor acurácia COM normalização: {melhor_acuracia_com_norm:.4f} ({melhor_acuracia_com_norm*100:.2f}%)")
print(f"  Obtida com k={melhor_k_com_norm}")

if melhor_acuracia_com_norm >= 0.98:
    print(f"  ✓ META DE 98% ATINGIDA!")
else:
    print(f"  ⚠ Meta de 98% não atingida, mas houve grande melhoria")

print(f"\n  MELHORIA: {(melhor_acuracia_com_norm - melhor_acuracia_sem_norm)*100:.2f} pontos percentuais")

# ============================================================================
# COMPARAÇÃO DETALHADA
# ============================================================================

print("\n" + "=" * 60)
print("COMPARAÇÃO: SEM vs COM NORMALIZAÇÃO")
print("=" * 60)
print(f"{'k':<5} {'Sem Normalização':<20} {'Com Normalização':<20} {'Diferença'}")
print("-" * 60)
for k in valores_k:
    sem = acuracias_sem_norm[k]
    com = acuracias_com_norm[k]
    diff = (com - sem) * 100
    print(f"{k:<5} {sem*100:>6.2f}%{'':<13} {com*100:>6.2f}%{'':<13} +{diff:>5.2f}%")

print("\n" + "=" * 60)
print("CONCLUSÕES GRUPO 2:")
print("=" * 60)
print("O QUE FOI FEITO:")
print("  - Aplicou-se normalização (StandardScaler) aos dados")
print("  - Testou-se diferentes valores de k")
print("\nPOR QUÊ:")
print("  - As 13 características químicas têm escalas muito diferentes")
print("  - Sem normalização, características com valores maiores (Prolina)")
print("    dominam a distância Euclidiana")
print("  - A normalização coloca todas na mesma escala (média=0, desvio=1)")
print("  - Isso permite que todas contribuam igualmente para a classificação")
print("\nRESULTADO:")
print(f"  - Acurácia aumentou de {melhor_acuracia_sem_norm*100:.2f}% para {melhor_acuracia_com_norm*100:.2f}%")
print(f"  - Melhoria de {(melhor_acuracia_com_norm - melhor_acuracia_sem_norm)*100:.2f} pontos percentuais")
if melhor_acuracia_com_norm >= 0.98:
    print("  - ✓ Meta de 98% atingida!")
print("=" * 60)
