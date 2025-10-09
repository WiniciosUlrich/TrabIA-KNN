"""
Função para normalizar dados
"""
from sklearn.preprocessing import StandardScaler
import numpy as np

def normalizacao(dadosTrain, dadosTest):
    """
    Normaliza os dados usando StandardScaler (Z-score normalization).
    
    A normalização é importante para o k-NN porque:
    - O k-NN usa distância Euclidiana, que é sensível à escala das características
    - Características com valores grandes dominam o cálculo da distância
    - Normalização coloca todas as características na mesma escala
    
    StandardScaler transforma os dados para ter média 0 e desvio padrão 1:
    z = (x - média) / desvio_padrão
    
    Args:
        dadosTrain: Dados de treinamento (matriz onde cada linha é um exemplo)
        dadosTest: Dados de teste (matriz onde cada linha é um exemplo)
    
    Returns:
        Tupla (dadosTrain_normalizado, dadosTest_normalizado)
    """
    # Criar o normalizador
    scaler = StandardScaler()
    
    # Ajustar (fit) o scaler aos dados de treinamento e transformá-los
    # O fit calcula a média e desvio padrão de cada característica
    dadosTrain_norm = scaler.fit_transform(dadosTrain)
    
    # Transformar os dados de teste usando os mesmos parâmetros
    # (média e desvio padrão calculados do treinamento)
    # IMPORTANTE: Não usar fit nos dados de teste para evitar vazamento de dados
    dadosTest_norm = scaler.transform(dadosTest)
    
    return dadosTrain_norm, dadosTest_norm
