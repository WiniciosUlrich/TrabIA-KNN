"""
Script principal para executar todas as demonstrações do classificador k-NN
============================================================================

Este script executa todas as demonstrações em sequência.
Você também pode executar cada demonstração individualmente.
"""

import sys

def main():
    print("=" * 70)
    print("CLASSIFICADOR k-NEAREST NEIGHBOUR (k-NN)")
    print("Trabalho de Implementação")
    print("=" * 70)
    
    print("\nEste trabalho implementa um classificador k-NN e testa em 4 datasets.")
    print("\nArquivos implementados:")
    print("  • dist.py - Função de distância Euclidiana")
    print("  • meuKnn.py - Implementação do classificador k-NN")
    print("  • visualizaPontos.py - Função para visualizar dados")
    print("  • normalizacao.py - Função para normalizar dados")
    print("  • demoD1.py - Demonstração com Grupo de Dados 1 (Iris)")
    print("  • demoD2.py - Demonstração com Grupo de Dados 2 (Wine)")
    print("  • demoD3.py - Demonstração com Grupo de Dados 3")
    print("  • demoD4.py - Demonstração com Grupo de Dados 4")
    
    print("\n" + "=" * 70)
    print("MENU DE OPÇÕES")
    print("=" * 70)
    print("1. Executar demonstração do Grupo 1 (Iris)")
    print("2. Executar demonstração do Grupo 2 (Wine)")
    print("3. Executar demonstração do Grupo 3")
    print("4. Executar demonstração do Grupo 4")
    print("5. Executar TODAS as demonstrações em sequência")
    print("0. Sair")
    
    while True:
        try:
            escolha = input("\nEscolha uma opção (0-5): ").strip()
            
            if escolha == '0':
                print("\nEncerrando...")
                break
            
            elif escolha == '1':
                print("\n" + "="*70)
                print("Executando demoD1.py...")
                print("="*70)
                exec(open('demoD1.py').read())
                
            elif escolha == '2':
                print("\n" + "="*70)
                print("Executando demoD2.py...")
                print("="*70)
                exec(open('demoD2.py').read())
                
            elif escolha == '3':
                print("\n" + "="*70)
                print("Executando demoD3.py...")
                print("="*70)
                exec(open('demoD3.py').read())
                
            elif escolha == '4':
                print("\n" + "="*70)
                print("Executando demoD4.py...")
                print("="*70)
                exec(open('demoD4.py').read())
                
            elif escolha == '5':
                print("\n" + "="*70)
                print("Executando TODAS as demonstrações...")
                print("="*70)
                
                print("\n" + "="*70)
                print("1/4: Executando demoD1.py...")
                print("="*70)
                exec(open('demoD1.py').read())
                
                input("\nPressione ENTER para continuar para o Grupo 2...")
                
                print("\n" + "="*70)
                print("2/4: Executando demoD2.py...")
                print("="*70)
                exec(open('demoD2.py').read())
                
                input("\nPressione ENTER para continuar para o Grupo 3...")
                
                print("\n" + "="*70)
                print("3/4: Executando demoD3.py...")
                print("="*70)
                exec(open('demoD3.py').read())
                
                input("\nPressione ENTER para continuar para o Grupo 4...")
                
                print("\n" + "="*70)
                print("4/4: Executando demoD4.py...")
                print("="*70)
                exec(open('demoD4.py').read())
                
                print("\n" + "="*70)
                print("TODAS AS DEMONSTRAÇÕES CONCLUÍDAS!")
                print("="*70)
                
            else:
                print("Opção inválida! Escolha entre 0 e 5.")
                
        except FileNotFoundError as e:
            print(f"\nErro: Arquivo não encontrado - {e}")
            print("Certifique-se de que todos os arquivos estão no diretório correto.")
        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário.")
            break
        except Exception as e:
            print(f"\nErro ao executar: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
