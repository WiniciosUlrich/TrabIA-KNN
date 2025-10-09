"""
Função para visualizar dados em gráficos 2D
"""
import matplotlib.pyplot as plt

def getDadosRotulo(dados, rotulos, rotulo, indice):
    """
    Extrai os dados de uma dimensão específica para um rótulo específico.
    
    Args:
        dados: Matriz de dados
        rotulos: Array de rótulos
        rotulo: O rótulo específico para filtrar
        indice: Índice da dimensão a extrair
    
    Returns:
        Lista com os valores da dimensão especificada para o rótulo dado
    """
    ret = []
    for idx in range(0, len(dados)):
        # Se o rótulo do exemplo atual for igual ao rótulo buscado
        if(rotulos[idx] == rotulo):
            # Adiciona o valor da dimensão especificada
            ret.append(dados[idx][indice])        
    return ret

def visualizaPontos(dados, rotulos, d1, d2):
    """
    Visualiza pontos em 2D para diferentes classes.
    
    Cria um gráfico de dispersão (scatter plot) mostrando os dados
    em duas dimensões, com cores diferentes para cada classe.
    
    Args:
        dados: Matriz de dados
        rotulos: Array de rótulos
        d1: Índice da primeira dimensão a visualizar (eixo x)
        d2: Índice da segunda dimensão a visualizar (eixo y)
    """
    # Criar figura e eixos
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plotar pontos da classe 1 em vermelho com marcador triangular
    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), 
               getDadosRotulo(dados, rotulos, 1, d2), 
               c='red', marker='^', s=100, label='Classe 1', alpha=0.7)
    
    # Plotar pontos da classe 2 em azul com marcador de cruz
    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), 
               getDadosRotulo(dados, rotulos, 2, d2), 
               c='blue', marker='+', s=100, label='Classe 2', alpha=0.7)
    
    # Plotar pontos da classe 3 em verde com marcador de ponto
    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), 
               getDadosRotulo(dados, rotulos, 3, d2), 
               c='green', marker='.', s=100, label='Classe 3', alpha=0.7)
    
    # Configurar rótulos dos eixos
    ax.set_xlabel(f'Dimensão {d1}', fontsize=12)
    ax.set_ylabel(f'Dimensão {d2}', fontsize=12)
    ax.set_title(f'Visualização dos Dados - Dimensões {d1} vs {d2}', fontsize=14)
    
    # Adicionar legenda
    ax.legend(fontsize=10)
    
    # Adicionar grade para facilitar leitura
    ax.grid(True, alpha=0.3)
    
    # Mostrar o gráfico
    plt.show()
