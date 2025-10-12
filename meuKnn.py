#Winicios Ivan Ulrich - Matheus Fritzen - Gabriel Puff

"""
Função para implementar o classificador k-Nearest Neighbour (k-NN)
"""
import numpy as np
from scipy.stats import mode
from dist import dist

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    """
    Implementa o classificador k-Nearest Neighbour (k-NN).
    
    O classificador k-NN classifica um novo exemplo com base nos k exemplos
    de treinamento mais próximos. Usa a distância Euclidiana para medir
    proximidade entre exemplos.
    
    Args:
        dadosTrain: Matriz de dados de treinamento (cada linha é um exemplo)
        rotuloTrain: Array de rótulos dos dados de treinamento
        dadosTeste: Matriz de dados de teste (cada linha é um exemplo)
        k: Número de vizinhos mais próximos a considerar
    
    Returns:
        Array com as predições para cada exemplo de teste
    """
    # Lista para armazenar as predições
    predicoes = []
    
    # Para cada exemplo de teste 
    for exemplo_teste in dadosTeste:
        # Lista para armazenar (distância, índice) de cada exemplo de treinamento
        distancias = []
        
        # Calcular a distância entre o exemplo de teste e cada exemplo de treinamento
        for i, exemplo_train in enumerate(dadosTrain):
            # Calcula a distância Euclidiana usando a função dist
            distancia = dist(exemplo_teste, exemplo_train)
            # Armazena a distância junto com o índice do exemplo
            distancias.append((distancia, i))
        
        # Ordenar as distâncias em ordem crescente
        # distancias_ordenadas[0] terá o vizinho mais próximo
        distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
        
        # Obter os índices dos k vizinhos mais próximos
        indices_k_vizinhos = [distancias_ordenadas[i][1] for i in range(k)]
        
        # Obter os rótulos correspondentes aos k vizinhos mais próximos
        rotulos_k_vizinhos = [rotuloTrain[i] for i in indices_k_vizinhos]
        
        if k == 1:
            # Para k=1, simplesmente pegar o rótulo do vizinho mais próximo
            predicao = rotulos_k_vizinhos[0]
        else:
            # Para k>1, usar a moda (valor mais frequente) dos k vizinhos
            # A moda retorna o rótulo que aparece mais vezes entre os k vizinhos
            predicao = mode(rotulos_k_vizinhos, keepdims=True)[0][0]
        
        # Adicionar a predição à lista
        predicoes.append(predicao)
    
    # Retornar as predições como um array numpy
    return np.array(predicoes)
