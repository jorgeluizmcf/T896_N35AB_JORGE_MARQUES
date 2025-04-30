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
  _____          _           _ _                 _           ___     ______            ____  ____ ___ 
 |_   _| __ __ _| |__   __ _| | |__   ___     __| | ___     / \ \   / /___ \          |  _ \|  _ \_ _|
   | || '__/ _` | '_ \ / _` | | '_ \ / _ \   / _` |/ _ \   / _ \ \ / /  __) |  _____  | |_) | | | | | 
   | || | | (_| | |_) | (_| | | | | | (_) | | (_| |  __/  / ___ \ V /  / __/  |_____| |  __/| |_| | | 
   |_||_|  \__,_|_.__/ \__,_|_|_| |_|\___/   \__,_|\___| /_/   \_\_/  |_____|         |_|   |____/___|
                                                                                                      
    """
    print(ascii_art)
    print("==================================================")
    print("Aluno: Jorge Luiz Marques da Costa Filho (2127467)")
    print("Professora: Lyndainês Araújo dos Santos")
    print("Turma: N896-09")
    print("==================================================")

# Carrega imagem
def carregar_imagem(caminho):
    return cv2.imread(caminho)

# Plot resultado com subplot
def exibir_resultados(imagens, titulos):
    plt.figure(figsize=(14, 6))
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(1, len(imagens), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Projeto 1 - Remoção de fundo verde e composição
def projeto_1():
    limpar_terminal()
    print("================== INICIANDO PROJETO 1 ====================")

    inicio = time.time()

    img = carregar_imagem('datasets/projeto_1/img_fundo_verde_1.jpg')
    background = carregar_imagem('datasets/projeto_1/WhatsApp Image 2025-04-29 at 20.18.32.jpeg')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    result_rgb = cv2.bitwise_and(img, img, mask=mask_inv)

    b, g, r = cv2.split(result_rgb)
    alpha = mask_inv
    rgba = cv2.merge((b, g, r, alpha))
    recorte = rgba

    background_resized = cv2.resize(background, (recorte.shape[1], recorte.shape[0]))
    b, g, r, a = cv2.split(recorte)
    overlay_rgb = cv2.merge((b, g, r))
    alpha = a.astype(float) / 255

    foreground = overlay_rgb.astype(float)
    background_f = background_resized.astype(float)

    for c in range(3):
        background_f[:, :, c] = alpha * foreground[:, :, c] + (1 - alpha) * background_f[:, :, c]

    resultado_final = background_f.astype(np.uint8)

    fim = time.time()
    print(f"[LOG] Tempo de duração dos processamentos de imagem: {fim - inicio:.2f} segundos")

    exibir_resultados(
        [img, background, resultado_final],
        ["Imagem com fundo verde", "Imagem background", "Resultado Final - Pessoa sobre Background"]
    )

    print("================== FINALIZANDO PROJETO 1 ====================")


# Projeto 2 - Detecção de Círculos
def projeto_2():
    limpar_terminal()
    print("================== INICIANDO PROJETO 2 ====================")

    inicio = time.time()

    # Carregar imagem
    imagem_original = carregar_imagem('datasets/projeto_2/circulos_1.png')
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    
    # Suavização
    imagem_suavizada = cv2.medianBlur(imagem_cinza, 5)

    # Detecção de círculos com HoughCircles
    circulos = cv2.HoughCircles(
        imagem_suavizada,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=60
    )

    imagem_resultado = imagem_original.copy()
    contador = 0

    if circulos is not None:
        circulos = np.round(circulos[0, :]).astype("int")
        contador = len(circulos)

        for (x, y, r) in circulos:
            cv2.circle(imagem_resultado, (x, y), r, (0, 255, 0), 2)
            cv2.circle(imagem_resultado, (x, y), 2, (0, 0, 255), 3)

    fim = time.time()

    print(f"[LOG] Tempo de duração do processamento: {fim - inicio:.2f} segundos")
    print(f"[LOG] Quantidade de círculos detectados: {contador}")

    # Mostrar resultados
    exibir_resultados(
        [imagem_original, imagem_resultado],
        ["Imagem Original", f"Imagem com Círculos Detectados ({contador})"]
    )

    print("================== FINALIZANDO PROJETO 2 ====================")


# Menu principal
def menu():
    while True:
        limpar_terminal()
        exibir_cabecalho()
        print("\nEscolha o projeto a executar:")
        print("1️⃣  Projeto 1 - Recorte e Colagem com fundo verde")
        print("2️⃣  Projeto 2 - (em breve)")
        print("3️⃣  Projeto 3 - (em breve)")
        print("4️⃣  Projeto 4 - (em breve)")
        print("0️⃣  Sair\n")

        opcao = input("Digite a opção desejada: ").strip()

        if opcao == '1':
            projeto_1()
            input("\nPressione ENTER para retornar ao menu...")
        elif opcao == '2':
            projeto_2()
            input("\nPressione ENTER para retornar ao menu...")
        elif opcao == '3':
            print("\n[INFO] Projeto 3 ainda não implementado.")
            input("Pressione ENTER para retornar ao menu...")
        elif opcao == '4':
            print("\n[INFO] Projeto 4 ainda não implementado.")
            input("Pressione ENTER para retornar ao menu...")
        elif opcao == '0':
            print("Encerrando aplicação...")
            break
        else:
            print("\n[ERRO] Opção inválida. Tente novamente.")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    menu()
