"""
TESTE RÁPIDO - Verificar se tudo está funcionando
==================================================

Execute este script para verificar se todas as funções
e datasets estão corretos.
"""

import os
import sys

def verificar_arquivos():
    """Verifica se todos os arquivos necessários existem"""
    print("=" * 60)
    print("VERIFICANDO ARQUIVOS")
    print("=" * 60)
    
    arquivos_necessarios = {
        "Funções": [
            "dist.py",
            "meuKnn.py",
            "visualizaPontos.py",
            "normalizacao.py"
        ],
        "Scripts de Demonstração": [
            "demoD1.py",
            "demoD2.py",
            "demoD3.py",
            "demoD4.py"
        ],
        "Datasets": [
            "grupoDados1.mat",
            "grupoDados2.mat",
            "grupoDados3.mat",
            "grupoDados4.mat"
        ]
    }
    
    tudo_ok = True
    
    for categoria, arquivos in arquivos_necessarios.items():
        print(f"\n{categoria}:")
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                print(f"  ✓ {arquivo}")
            else:
                print(f"  ✗ {arquivo} - FALTANDO!")
                tudo_ok = False
    
    return tudo_ok

def testar_funcoes():
    """Testa se as funções funcionam corretamente"""
    print("\n" + "=" * 60)
    print("TESTANDO FUNÇÕES")
    print("=" * 60)
    
    try:
        # Testar dist
        print("\n1. Testando dist.py...")
        from dist import dist
        import numpy as np
        
        p = np.array([0, 0])
        q = np.array([3, 4])
        d = dist(p, q)
        
        if abs(d - 5.0) < 0.001:
            print(f"  ✓ dist funcionando corretamente (dist([0,0], [3,4]) = {d})")
        else:
            print(f"  ✗ dist retornou {d}, esperado 5.0")
            return False
        
        # Testar meuKnn
        print("\n2. Testando meuKnn.py...")
        from meuKnn import meuKnn
        
        # Teste simples
        train = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
        labels = np.array([1, 1, 2, 2])
        test = np.array([[0.5, 0.5], [5.5, 5.5]])
        
        pred = meuKnn(train, labels, test, k=1)
        
        if pred[0] == 1 and pred[1] == 2:
            print(f"  ✓ meuKnn funcionando corretamente")
            print(f"    Predições: {pred} (esperado: [1 2])")
        else:
            print(f"  ✗ meuKnn retornou {pred}, esperado [1 2]")
            return False
        
        # Testar normalizacao
        print("\n3. Testando normalizacao.py...")
        from normalizacao import normalizacao
        
        train_norm, test_norm = normalizacao(train, test)
        
        # Dados normalizados devem ter média ~0 e desvio ~1
        media = np.mean(train_norm)
        desvio = np.std(train_norm)
        
        if abs(media) < 0.001 and abs(desvio - 1.0) < 0.1:
            print(f"  ✓ normalizacao funcionando corretamente")
            print(f"    Média: {media:.4f}, Desvio: {desvio:.4f}")
        else:
            print(f"  ✗ Normalização incorreta (média={media:.4f}, desvio={desvio:.4f})")
            return False
        
        print("\n4. Testando visualizaPontos.py...")
        try:
            from visualizaPontos import visualizaPontos, getDadosRotulo
            print(f"  ✓ visualizaPontos importado com sucesso")
            print(f"    (visualização não testada automaticamente)")
        except Exception as e:
            print(f"  ✗ Erro ao importar: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Erro ao testar funções: {e}")
        import traceback
        traceback.print_exc()
        return False

def testar_datasets():
    """Testa se os datasets podem ser carregados"""
    print("\n" + "=" * 60)
    print("TESTANDO DATASETS")
    print("=" * 60)
    
    try:
        import scipy.io as scipy
        
        datasets = [
            ("grupoDados1.mat", ['grupoTest', 'grupoTrain', 'testRots', 'trainRots']),
            ("grupoDados2.mat", ['grupoTest', 'grupoTrain', 'testRots', 'trainRots']),
            ("grupoDados3.mat", ['grupoTest', 'grupoTrain', 'testRots', 'trainRots']),
            ("grupoDados4.mat", ['testSet', 'trainSet', 'testLabs', 'trainLabs'])
        ]
        
        for arquivo, variaveis_esperadas in datasets:
            print(f"\n{arquivo}:")
            try:
                mat = scipy.loadmat(arquivo)
                
                for var in variaveis_esperadas:
                    if var in mat:
                        dados = mat[var]
                        print(f"  ✓ {var}: shape {dados.shape}")
                    else:
                        print(f"  ✗ {var} não encontrado!")
                        return False
                        
            except Exception as e:
                print(f"  ✗ Erro ao carregar: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Erro ao testar datasets: {e}")
        return False

def teste_completo():
    """Executa teste completo simples"""
    print("\n" + "=" * 60)
    print("TESTE COMPLETO - GRUPO 1")
    print("=" * 60)
    
    try:
        from meuKnn import meuKnn
        import scipy.io as scipy
        import numpy as np
        
        # Carregar Grupo 1
        mat = scipy.loadmat('grupoDados1.mat')
        grupoTest = mat['grupoTest']
        grupoTrain = mat['grupoTrain']
        testRots = mat['testRots'].flatten()
        trainRots = mat['trainRots'].flatten()
        
        print(f"\nDados carregados:")
        print(f"  Treino: {grupoTrain.shape}")
        print(f"  Teste: {grupoTest.shape}")
        
        # Testar com k=1
        print(f"\nTestando k=1...")
        pred = meuKnn(grupoTrain, trainRots, grupoTest, 1)
        acuracia = np.sum(pred == testRots) / len(testRots)
        
        print(f"  Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")
        
        if acuracia >= 0.95:
            print(f"  ✓ Resultado esperado (~96%)")
        else:
            print(f"  ⚠ Resultado abaixo do esperado")
        
        # Testar com k=10
        print(f"\nTestando k=10...")
        pred = meuKnn(grupoTrain, trainRots, grupoTest, 10)
        acuracia = np.sum(pred == testRots) / len(testRots)
        
        print(f"  Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")
        
        if acuracia >= 0.93:
            print(f"  ✓ Resultado esperado (~94%)")
        else:
            print(f"  ⚠ Resultado abaixo do esperado")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Erro no teste completo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes"""
    print("\n" * 2)
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  TESTE RÁPIDO - CLASSIFICADOR k-NN".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    # Verificar arquivos
    if not verificar_arquivos():
        print("\n⚠ ATENÇÃO: Alguns arquivos estão faltando!")
        print("Certifique-se de que todos os arquivos estão no diretório.")
        return
    
    print("\n✓ Todos os arquivos encontrados!")
    
    # Testar funções
    if not testar_funcoes():
        print("\n✗ Falha nos testes das funções!")
        return
    
    print("\n✓ Todas as funções funcionando corretamente!")
    
    # Testar datasets
    if not testar_datasets():
        print("\n✗ Falha ao carregar datasets!")
        return
    
    print("\n✓ Todos os datasets carregados com sucesso!")
    
    # Teste completo
    if not teste_completo():
        print("\n✗ Falha no teste completo!")
        return
    
    print("\n✓ Teste completo executado com sucesso!")
    
    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print("✓ Arquivos: OK")
    print("✓ Funções: OK")
    print("✓ Datasets: OK")
    print("✓ Teste completo: OK")
    print("\n✓✓✓ TUDO FUNCIONANDO CORRETAMENTE! ✓✓✓")
    print("\nVocê pode agora executar:")
    print("  python main_menu.py")
    print("ou qualquer script de demonstração (demoD1.py, etc.)")
    print("=" * 60)

if __name__ == "__main__":
    main()
