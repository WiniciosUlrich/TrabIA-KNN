#Winicios Ivan Ulrich - Matheus Fritzen - Gabriel Puff

"""
Função para calcular a distância Euclidiana entre dois pontos
"""
import numpy as np

def dist(p, q):
    """
    Calcula a distância Euclidiana entre dois pontos p e q.
    
    A distância Euclidiana é calculada como:
    d(p,q) = sqrt(sum((pi - qi)^2))
    
    Args:
        p: Primeiro ponto (vetor/array)
        q: Segundo ponto (vetor/array)
    
    Returns:
        float: Distância Euclidiana entre p e q
    """
    # Calcula a diferença entre os pontos
    diferenca = p - q
    
    # Eleva ao quadrado cada componente da diferença
    quadrados = diferenca ** 2
    
    # Soma todos os quadrados
    soma_quadrados = np.sum(quadrados)
    
    # Retorna a raiz quadrada da soma
    return np.sqrt(soma_quadrados)
