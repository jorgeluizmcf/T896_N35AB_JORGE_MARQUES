import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Suprime mensagens desnecessárias
sys.stderr = open(os.devnull, 'w')

# Limpa o terminal
def limpar_terminal():
    os.system('clear' if os.name == 'posix' else 'cls')

# Cabeçalho e título
def exibir_cabecalho():
    ascii_art = r"""
  _______        _           _ _                 _            __      ______             _____  _____ _____ 
 |__   __|      | |         | | |               | |          /\ \    / /___ \           |  __ \|  __ \_   _|
    | |_ __ __ _| |__   __ _| | |__   ___     __| | ___     /  \ \  / /  __) |  ______  | |__) | |  | || |  
    | | '__/ _` | '_ \ / _` | | '_ \ / _ \   / _` |/ _ \   / /\ \ \/ /  |__ <  |______| |  ___/| |  | || |  
    | | | | (_| | |_) | (_| | | | | | (_) | | (_| |  __/  / ____ \  /   ___) |          | |    | |__| || |_ 
    |_|_|  \__,_|_.__/ \__,_|_|_| |_|\___/   \__,_|\___| /_/    \_\/   |____/           |_|    |_____/_____|
                                                                                                                                                                                                             
    """
    print(ascii_art)
    print("=======================================================")
    print("Alunos: Jorge Luiz Marques da Costa Filho (2127467)")
    print("        Dalton Linconl Saraiva Damasceno Lima (2225913)\n")
    print("Professora: Lyndainês Araújo dos Santos")
    print("Turma: N896-09")
    print("=======================================================")

# Carrega imagem
def carregar_imagem(caminho):
    return cv2.imread(caminho)

# Plot resultado com subplot
def exibir_resultados(imagens, titulos, titulo_geral="Resultado"):
    plt.figure(figsize=(14, 6))
    plt.suptitle(titulo_geral, fontsize=16, fontweight='bold')
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(1, len(imagens), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Função principal para restauração das imagens
def restaurar_imagem():
    return print("Gerar função aqui...")

# Menu principal
def menu():
    while True:
        limpar_terminal()
        exibir_cabecalho()
        print("\nEscolha o projeto a executar:")
        print("1️⃣  Iniciar restauração de retrato.")
        print("0️⃣  Sair\n")

        opcao = input("Digite a opção desejada: ").strip()

        if opcao == '1':
            restaurar_imagem()
            input("\nPressione ENTER para retornar ao menu...")
        elif opcao == '0':
            print("Encerrando aplicação...")
            break
        else:
            print("\n[ERRO] Opção inválida. Tente novamente.")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    menu()