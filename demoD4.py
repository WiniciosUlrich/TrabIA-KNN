#Winicios Ivan Ulrich - Matheus Fritzen - Gabriel Puff

"""
Script de demonstração para o Grupo de Dados 4
===============================================

Dataset: Grupo 4 (dataset genérico)
Objetivo: Classificação em 3 classes
Características: 4
OBSERVAÇÃO: Este dataset tem nomes de variáveis diferentes!
"""

import numpy as np
import scipy.io as scipy
from meuKnn import meuKnn
from normalizacao import normalizacao
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Carregar os dados do arquivo .mat
print("=" * 60)
print("GRUPO DE DADOS 4")
print("=" * 60)

mat = scipy.loadmat('grupoDados4.mat')

# ATENÇÃO: Nomes de variáveis diferentes neste dataset!
# grupoTest=testSet, grupoTrain=trainSet
# testRots=testLabs, trainRots=trainLabs
grupoTest = mat['testSet']
grupoTrain = mat['trainSet']
testRots = mat['testLabs'].flatten()
trainRots = mat['trainLabs'].flatten()

# print(f"\nDados carregados:")
# print(f"  Treinamento: {grupoTrain.shape} (exemplos x características)")
# print(f"  Teste: {grupoTest.shape} (exemplos x características)")
# print(f"  Classes: {np.unique(trainRots)}")
# print(f"  Distribuição de classes (treino): {np.bincount(trainRots)[1:]}")
# print(f"  Distribuição de classes (teste):  {np.bincount(testRots)[1:]}")

# # Analisar estatísticas dos dados
# print(f"\nEstatísticas dos dados de treinamento:")
# print(f"  Mínimo: {np.min(grupoTrain, axis=0)}")
# print(f"  Máximo: {np.max(grupoTrain, axis=0)}")
# print(f"  Média:  {np.mean(grupoTrain, axis=0)}")
# print(f"\n  OBSERVAÇÃO: Características têm escalas diferentes!")

# ============================================================================
# Q4.1: Aplique k-NN ao problema. Qual é a acurácia?
# ============================================================================

print("\n" + "-" * 60)
print("Q4.1: APLICANDO k-NN (TESTE INICIAL)")
print("-" * 60)

# Testar com diferentes valores de k (sem normalização)
valores_k = [1, 3, 5, 7, 10, 15]
print("Testando diferentes valores de k (SEM normalização):")

acuracias_sem_norm = {}
totalNum = len(testRots)

for k in valores_k:
    rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k)
    estaCorreto = rotuloPrevisto == testRots
    numCorreto = np.sum(estaCorreto)
    acuracia = numCorreto / totalNum
    acuracias_sem_norm[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)")

melhor_k_sem_norm = max(acuracias_sem_norm, key=acuracias_sem_norm.get)
melhor_acuracia_sem_norm = acuracias_sem_norm[melhor_k_sem_norm]

print(f"\n✓ RESPOSTA Q4.1:")
print(f"  Melhor acurácia inicial: {melhor_acuracia_sem_norm:.4f} ({melhor_acuracia_sem_norm*100:.2f}%)")
print(f"  Obtida com k={melhor_k_sem_norm}")
print(f"  PROBLEMA: Acurácia baixa (~71-77%)")

# ============================================================================
# Q4.2: Ajuste para obter 92% de acurácia
# ============================================================================

print("\n" + "-" * 60)
print("Q4.2: INVESTIGANDO PROBLEMAS E APLICANDO SOLUÇÕES")
print("-" * 60)

print("\nDIAGNÓSTICO DOS PROBLEMAS:")
print("  PROBLEMA 1: Escalas diferentes das características")
print("    - Como no Grupo 2, características têm magnitudes diferentes")
print("    - Afeta o cálculo da distância Euclidiana")
print("    - Solução: NORMALIZAÇÃO")
print("\n  PROBLEMA 2: Valor de k inadequado")
print("    - k muito pequeno pode ser sensível a ruído")
print("    - k muito grande pode suavizar demais as fronteiras")
print("    - Solução: TESTAR MÚLTIPLOS VALORES DE k")
print("\n  PROBLEMA 3: Possível presença de outliers")
print("    - Outliers podem afetar a normalização")
print("    - Solução: TESTAR DIFERENTES MÉTODOS DE NORMALIZAÇÃO")

print("\n" + "-" * 60)
print("SOLUÇÃO 1: NORMALIZAÇÃO COM StandardScaler")
print("-" * 60)

# Normalizar os dados com StandardScaler
grupoTrain_std, grupoTest_std = normalizacao(grupoTrain, grupoTest)

print("Testando com StandardScaler:")
acuracias_std = {}
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    rotuloPrevisto = meuKnn(grupoTrain_std, trainRots, grupoTest_std, k)
    acuracia = np.sum(rotuloPrevisto == testRots) / totalNum
    acuracias_std[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)", end="")
    if acuracia >= 0.92:
        print(" ✓ META ATINGIDA!")
    else:
        print()

melhor_k_std = max(acuracias_std, key=acuracias_std.get)
melhor_acuracia_std = acuracias_std[melhor_k_std]

print("\n" + "-" * 60)
print("SOLUÇÃO 2: NORMALIZAÇÃO COM MinMaxScaler")
print("-" * 60)

# MinMaxScaler: escala para [0, 1]
scaler_minmax = MinMaxScaler()
grupoTrain_minmax = scaler_minmax.fit_transform(grupoTrain)
grupoTest_minmax = scaler_minmax.transform(grupoTest)

print("Testando com MinMaxScaler:")
acuracias_minmax = {}
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    rotuloPrevisto = meuKnn(grupoTrain_minmax, trainRots, grupoTest_minmax, k)
    acuracia = np.sum(rotuloPrevisto == testRots) / totalNum
    acuracias_minmax[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)", end="")
    if acuracia >= 0.92:
        print(" ✓ META ATINGIDA!")
    else:
        print()

melhor_k_minmax = max(acuracias_minmax, key=acuracias_minmax.get)
melhor_acuracia_minmax = acuracias_minmax[melhor_k_minmax]

print("\n" + "-" * 60)
print("SOLUÇÃO 3: NORMALIZAÇÃO COM RobustScaler")
print("-" * 60)

# RobustScaler: usa mediana e IQR, robusto a outliers
scaler_robust = RobustScaler()
grupoTrain_robust = scaler_robust.fit_transform(grupoTrain)
grupoTest_robust = scaler_robust.transform(grupoTest)

print("Testando com RobustScaler (robusto a outliers):")
acuracias_robust = {}
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    rotuloPrevisto = meuKnn(grupoTrain_robust, trainRots, grupoTest_robust, k)
    acuracia = np.sum(rotuloPrevisto == testRots) / totalNum
    acuracias_robust[k] = acuracia
    print(f"  k={k:2d}: Acurácia = {acuracia:.4f} ({acuracia*100:.2f}%)", end="")
    if acuracia >= 0.92:
        print(" ✓ META ATINGIDA!")
    else:
        print()

melhor_k_robust = max(acuracias_robust, key=acuracias_robust.get)
melhor_acuracia_robust = acuracias_robust[melhor_k_robust]

# ============================================================================
# RESUMO DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 60)
print("RESUMO DOS RESULTADOS")
print("=" * 60)

resultados = [
    ("Sem normalização", melhor_k_sem_norm, melhor_acuracia_sem_norm),
    ("StandardScaler", melhor_k_std, melhor_acuracia_std),
    ("MinMaxScaler", melhor_k_minmax, melhor_acuracia_minmax),
    ("RobustScaler", melhor_k_robust, melhor_acuracia_robust)
]

print(f"\n{'Método':<20} {'Melhor k':<10} {'Acurácia':<15} {'Status'}")
print("-" * 60)
for metodo, k, acc in resultados:
    status = "✓ META!" if acc >= 0.92 else ""
    print(f"{metodo:<20} {k:<10} {acc*100:>6.2f}%{'':<8} {status}")

# Encontrar o melhor resultado geral
melhor_resultado = max(resultados, key=lambda x: x[2])
melhor_metodo, melhor_k_final, melhor_acuracia_final = melhor_resultado

print(f"\n✓ RESPOSTA Q4.2:")
print(f"  Melhor resultado: {melhor_acuracia_final:.4f} ({melhor_acuracia_final*100:.2f}%)")
print(f"  Método: {melhor_metodo}")
print(f"  k = {melhor_k_final}")

if melhor_acuracia_final >= 0.92:
    print(f"  ✓ META DE 92% ATINGIDA!")
else:
    print(f"  ⚠ Meta de 92% não atingida (máximo: {melhor_acuracia_final*100:.2f}%)")
    print(f"  Possíveis razões:")
    print(f"    - Dataset pode ter ruído excessivo")
    print(f"    - Sobreposição significativa entre classes")
    print(f"    - Características podem não ser suficientemente discriminativas")

print(f"\n  MELHORIA TOTAL: {(melhor_acuracia_final - melhor_acuracia_sem_norm)*100:.2f} pontos percentuais")

# ============================================================================
# CONCLUSÕES
# ============================================================================

print("\n" + "=" * 60)
print("CONCLUSÕES GRUPO 4:")
print("=" * 60)
print("O QUE FOI FEITO:")
print("  1. Normalização dos dados (StandardScaler, MinMaxScaler, RobustScaler)")
print(f"  2. Ajuste do valor de k (testado k de 1 a 15)")
print("  3. Teste de múltiplos métodos de normalização")
print("\nPOR QUÊ:")
print("  PROBLEMA 1 - Escalas diferentes:")
print("    • Características têm magnitudes diferentes")
print("    • Distância Euclidiana é sensível a escala")
print("    • Normalização equaliza a contribuição de cada característica")
print("\n  PROBLEMA 2 - Valor de k inadequado:")
print("    • k=1 sensível a ruído")
print("    • k muito alto suaviza demais as fronteiras")
print(f"    • k ótimo encontrado: {melhor_k_final}")
print("\n  PROBLEMA 3 - Possíveis outliers:")
print("    • StandardScaler sensível a outliers (usa média e desvio)")
print("    • RobustScaler usa mediana e IQR (mais robusto)")
print("    • MinMaxScaler escala para [0,1]")
print("\nRESULTADOS:")
print(f"  • Sem normalização: {melhor_acuracia_sem_norm*100:.2f}%")
print(f"  • Com {melhor_metodo}: {melhor_acuracia_final*100:.2f}%")
print(f"  • Melhoria: +{(melhor_acuracia_final - melhor_acuracia_sem_norm)*100:.2f} pontos percentuais")

if melhor_acuracia_final >= 0.92:
    print("\n  ✓ Sucesso! Meta de 92% atingida!")
else:
    print(f"\n  ⚠ Meta não atingida. Máximo obtido: {melhor_acuracia_final*100:.2f}%")
    print("  Limitações intrínsecas do dataset podem impedir acurácia maior.")

print("\nLIÇÕES APRENDIDAS:")
print("  • k-NN é muito sensível à normalização")
print("  • Diferentes métodos de normalização podem ter resultados diferentes")
print("  • Sempre testar múltiplos valores de k")
print("  • RobustScaler útil quando há suspeita de outliers")
print("  • Nem sempre é possível atingir metas muito altas de acurácia")
print("=" * 60)
