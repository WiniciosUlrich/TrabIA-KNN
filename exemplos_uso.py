"""
Exemplos de uso do classificador k-NN
=====================================

Este arquivo contém exemplos práticos de como usar as funções implementadas
para diferentes análises e visualizações.
"""

import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
from main import meuKnn, visualizaPontos, calcular_acuracia, normalizar_dados

def exemplo_basico():
    """
    Exemplo básico de uso do classificador k-NN
    """
    print("=== EXEMPLO BÁSICO ===")
    
    # Carregar dados do grupo 1 (Iris)
    mat = scipy.loadmat('grupoDados1.mat')
    grupo_test = mat['grupoTest']
    grupo_train = mat['grupoTrain']
    test_rots = mat['testRots'].flatten()
    train_rots = mat['trainRots'].flatten()
    
    # Fazer predição com k=1
    predicoes = meuKnn(grupo_train, train_rots, grupo_test, k=1)
    
    # Calcular acurácia
    acuracia = calcular_acuracia(predicoes, test_rots)
    
    print(f"Dados de treinamento: {grupo_train.shape}")
    print(f"Dados de teste: {grupo_test.shape}")
    print(f"Acurácia com k=1: {acuracia:.4f} ({acuracia*100:.2f}%)")
    
    return predicoes, acuracia

def exemplo_comparacao_k():
    """
    Compara diferentes valores de k
    """
    print("\n=== COMPARAÇÃO DE DIFERENTES VALORES DE K ===")
    
    # Carregar dados
    mat = scipy.loadmat('grupoDados1.mat')
    grupo_test = mat['grupoTest']
    grupo_train = mat['grupoTrain']
    test_rots = mat['testRots'].flatten()
    train_rots = mat['trainRots'].flatten()
    
    valores_k = [1, 3, 5, 7, 10, 15]
    resultados = {}
    
    for k in valores_k:
        predicoes = meuKnn(grupo_train, train_rots, grupo_test, k)
        acuracia = calcular_acuracia(predicoes, test_rots)
        resultados[k] = acuracia
        print(f"k={k:2d}: {acuracia:.4f} ({acuracia*100:.2f}%)")
    
    # Encontrar melhor k
    melhor_k = max(resultados, key=resultados.get)
    melhor_acuracia = resultados[melhor_k]
    print(f"\nMelhor resultado: k={melhor_k} com {melhor_acuracia:.4f} ({melhor_acuracia*100:.2f}%)")
    
    return resultados

def exemplo_normalizacao():
    """
    Demonstra o efeito da normalização
    """
    print("\n=== EFEITO DA NORMALIZAÇÃO ===")
    
    # Carregar dados do grupo 2 (vinhos) que se beneficia muito da normalização
    mat = scipy.loadmat('grupoDados2.mat')
    grupo_test = mat['grupoTest']
    grupo_train = mat['grupoTrain']
    test_rots = mat['testRots'].flatten()
    train_rots = mat['trainRots'].flatten()
    
    # Testar sem normalização
    print("Sem normalização:")
    predicoes_sem_norm = meuKnn(grupo_train, train_rots, grupo_test, k=1)
    acuracia_sem_norm = calcular_acuracia(predicoes_sem_norm, test_rots)
    print(f"  Acurácia: {acuracia_sem_norm:.4f} ({acuracia_sem_norm*100:.2f}%)")
    
    # Testar com normalização
    print("Com normalização:")
    grupo_train_norm, grupo_test_norm = normalizar_dados(grupo_train, grupo_test)
    predicoes_com_norm = meuKnn(grupo_train_norm, train_rots, grupo_test_norm, k=1)
    acuracia_com_norm = calcular_acuracia(predicoes_com_norm, test_rots)
    print(f"  Acurácia: {acuracia_com_norm:.4f} ({acuracia_com_norm*100:.2f}%)")
    
    melhoria = acuracia_com_norm - acuracia_sem_norm
    print(f"Melhoria: +{melhoria:.4f} (+{melhoria*100:.2f} pontos percentuais)")
    
    return acuracia_sem_norm, acuracia_com_norm

def exemplo_visualizacao():
    """
    Exemplo de como criar visualizações
    """
    print("\n=== EXEMPLO DE VISUALIZAÇÃO ===")
    
    # Carregar dados do grupo 1
    mat = scipy.loadmat('grupoDados1.mat')
    dados = mat['grupoTrain']
    rotulos = mat['trainRots'].flatten()
    
    print("Criando visualização das primeiras duas dimensões...")
    print("(Feche a janela do gráfico para continuar)")
    
    # Criar visualização
    plt.figure(figsize=(10, 6))
    
    # Subplot 1: Dimensões 0 e 1
    plt.subplot(1, 2, 1)
    for classe in np.unique(rotulos):
        mask = rotulos == classe
        plt.scatter(dados[mask, 0], dados[mask, 1], 
                   label=f'Classe {classe}', alpha=0.7)
    plt.xlabel('Dimensão 0')
    plt.ylabel('Dimensão 1')
    plt.title('Iris Dataset - Dimensões 0 e 1')
    plt.legend()
    
    # Subplot 2: Dimensões 2 e 3
    plt.subplot(1, 2, 2)
    for classe in np.unique(rotulos):
        mask = rotulos == classe
        plt.scatter(dados[mask, 2], dados[mask, 3], 
                   label=f'Classe {classe}', alpha=0.7)
    plt.xlabel('Dimensão 2')
    plt.ylabel('Dimensão 3')
    plt.title('Iris Dataset - Dimensões 2 e 3')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Visualização criada!")

def exemplo_analise_caracteristicas():
    """
    Analisa o impacto de diferentes características
    """
    print("\n=== ANÁLISE DE CARACTERÍSTICAS ===")
    
    # Carregar dados
    mat = scipy.loadmat('grupoDados1.mat')
    grupo_test = mat['grupoTest']
    grupo_train = mat['grupoTrain']
    test_rots = mat['testRots'].flatten()
    train_rots = mat['trainRots'].flatten()
    
    # Testar com diferentes combinações de características
    combinacoes = {
        "Todas (0,1,2,3)": [0, 1, 2, 3],
        "Primeiras 2 (0,1)": [0, 1],
        "Últimas 2 (2,3)": [2, 3],
        "Ímpares (0,2)": [0, 2],
        "Pares (1,3)": [1, 3],
        "Apenas 0": [0],
        "Apenas 1": [1],
        "Apenas 2": [2],
        "Apenas 3": [3]
    }
    
    resultados = {}
    
    for nome, indices in combinacoes.items():
        # Selecionar características
        train_subset = grupo_train[:, indices]
        test_subset = grupo_test[:, indices]
        
        # Fazer predição
        predicoes = meuKnn(train_subset, train_rots, test_subset, k=3)
        acuracia = calcular_acuracia(predicoes, test_rots)
        
        resultados[nome] = acuracia
        print(f"{nome:15s}: {acuracia:.4f} ({acuracia*100:.2f}%)")
    
    # Encontrar melhor combinação
    melhor_combinacao = max(resultados, key=resultados.get)
    melhor_acuracia = resultados[melhor_combinacao]
    print(f"\nMelhor combinação: {melhor_combinacao}")
    print(f"Acurácia: {melhor_acuracia:.4f} ({melhor_acuracia*100:.2f}%)")
    
    return resultados

def exemplo_matriz_confusao():
    """
    Cria uma matriz de confusão simples
    """
    print("\n=== MATRIZ DE CONFUSÃO ===")
    
    # Carregar dados
    mat = scipy.loadmat('grupoDados1.mat')
    grupo_test = mat['grupoTest']
    grupo_train = mat['grupoTrain']
    test_rots = mat['testRots'].flatten()
    train_rots = mat['trainRots'].flatten()
    
    # Fazer predição
    predicoes = meuKnn(grupo_train, train_rots, grupo_test, k=3)
    
    # Criar matriz de confusão manual
    classes = np.unique(test_rots)
    n_classes = len(classes)
    matriz_confusao = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, classe_real in enumerate(classes):
        for j, classe_pred in enumerate(classes):
            count = np.sum((test_rots == classe_real) & (predicoes == classe_pred))
            matriz_confusao[i, j] = count
    
    print("Matriz de Confusão:")
    print("     Predito:")
    print("     ", end="")
    for classe in classes:
        print(f"{classe:4d}", end="")
    print()
    
    for i, classe_real in enumerate(classes):
        print(f"Real {classe_real}: ", end="")
        for j in range(n_classes):
            print(f"{matriz_confusao[i, j]:4d}", end="")
        print()
    
    # Calcular precisão por classe
    print("\nPrecisão por classe:")
    for i, classe in enumerate(classes):
        tp = matriz_confusao[i, i]
        total_real = np.sum(matriz_confusao[i, :])
        precisao = tp / total_real if total_real > 0 else 0
        print(f"Classe {classe}: {precisao:.4f} ({precisao*100:.2f}%)")
    
    return matriz_confusao

if __name__ == "__main__":
    print("EXEMPLOS DE USO DO CLASSIFICADOR k-NN")
    print("=" * 50)
    
    # Executar todos os exemplos
    exemplo_basico()
    exemplo_comparacao_k()
    exemplo_normalizacao()
    exemplo_analise_caracteristicas()
    exemplo_matriz_confusao()
    
    # Pergunta se quer ver visualização
    resposta = input("\nDeseja ver a visualização gráfica? (s/n): ").lower()
    if resposta in ['s', 'sim', 'y', 'yes']:
        exemplo_visualizacao()
    
    print("\n" + "=" * 50)
    print("Exemplos concluídos!")