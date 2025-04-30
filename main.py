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
    print("================== INICIANDO PROJETO 2 ====================\n")

    exemplos = [
        {
            "nome": "Exemplo 1",
            "caminho_imagem": 'datasets/projeto_2/circulos_1.png',
            # Os parâmetros foram testados e definidos de forma empírica, alterando e validando os resultados
            "parametros": {
                "dp": 1.0,
                "minDist": 50,
                "param1": 100,
                "param2": 40,
                "minRadius": 50,
                "maxRadius": 100
            }
        },
        {
            "nome": "Exemplo 2",
            "caminho_imagem": 'datasets/projeto_2/red-cherries-arranged-circular-frame-blue-background.jpg',
            # Os parâmetros foram testados e definidos de forma empírica, alterando e validando os resultados
            "parametros": {
                "dp": 1.0,
                "minDist": 50,
                "param1": 100,
                "param2": 40,
                "minRadius": 60,
                "maxRadius": 130
            }
        }
    ]

    for exemplo in exemplos:
        print(f"[LOG] {exemplo['nome']} - Iniciando processamento")
        inicio = time.time()

        imagem_original = carregar_imagem(exemplo["caminho_imagem"])
        imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
        imagem_suavizada = cv2.medianBlur(imagem_cinza, 5)

        circulos = cv2.HoughCircles(
            imagem_suavizada,
            cv2.HOUGH_GRADIENT,
            dp=exemplo["parametros"]["dp"],
            minDist=exemplo["parametros"]["minDist"],
            param1=exemplo["parametros"]["param1"],
            param2=exemplo["parametros"]["param2"],
            minRadius=exemplo["parametros"]["minRadius"],
            maxRadius=exemplo["parametros"]["maxRadius"]
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
        print(f"[LOG] {exemplo['nome']} - Tempo de processamento: {fim - inicio:.2f} segundos")
        print(f"[LOG] {exemplo['nome']} - Círculos detectados: {contador}\n")

        exibir_resultados(
            [imagem_original, imagem_resultado],
            ["Imagem Original", f"Imagem com Círculos Detectados ({contador})"],
            titulo_geral=exemplo["nome"]
        )

    print("================== FINALIZANDO PROJETO 2 ====================\n")

# Projeto 3 - Segmentação de Folha (Regiões Saudáveis e Danificadas)
def projeto_3():
    limpar_terminal()
    print("================== INICIANDO PROJETO 3 ====================\n")

    inicio = time.time()

    caminho_imagem = 'datasets/projeto_3/img_folha_4.JPG'
    imagem_original = carregar_imagem(caminho_imagem)
    imagem_hsv = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2HSV)

    # Tresholds definidos de forma empírica (tentativa e erro)
    # Segmentação da região saudável (tons de verde)
    lower_healthy = np.array([40, 10, 10])
    upper_healthy = np.array([85, 255, 255])
    mascara_saudavel = cv2.inRange(imagem_hsv, lower_healthy, upper_healthy)
    regiao_saudavel = cv2.bitwise_and(imagem_original, imagem_original, mask=mascara_saudavel)

    # Tresholds definidos de forma empírica (tentativa e erro)
    # Segmentação da região danificada (tons escuros/marrom)
    lower_danificada = np.array([5, 80, 0])
    upper_danificada = np.array([30, 255, 200])
    mascara_danificada = cv2.inRange(imagem_hsv, lower_danificada, upper_danificada)
    regiao_danificada = cv2.bitwise_and(imagem_original, imagem_original, mask=mascara_danificada)

    fim = time.time()
    print(f"[LOG] Tempo de processamento: {fim - inicio:.2f} segundos\n")

    exibir_resultados(
        [imagem_original, regiao_saudavel, regiao_danificada],
        ["Imagem Original", "Região Saudável (Verde)", "Região Danificada (Escura)"]
    )

    print("================== FINALIZANDO PROJETO 3 ====================\n")

# Projeto 4 - Segmentação de imagens médicas
def projeto_4():
    limpar_terminal()
    print("================== INICIANDO PROJETO 4 ====================\n")

    inicio = time.time()

    caminho_imagem = 'datasets/projeto_4/Tumor (103).jpg'
    imagem_original = carregar_imagem(caminho_imagem)

    # Redução de ruído com filtro gaussiano
    imagem_denoised = cv2.GaussianBlur(imagem_original, (5, 5), 0)

    # Conversão para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem_denoised, cv2.COLOR_BGR2GRAY)

    # Binarização usando Otsu
    _, imagem_bin = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operações morfológicas
    kernel = np.ones((3, 3), np.uint8)

    # Abertura (remoção de pequenos ruídos brancos)
    imagem_abertura = cv2.morphologyEx(imagem_bin, cv2.MORPH_OPEN, kernel)

    # Fechamento (preenchimento de pequenos buracos)
    imagem_fechamento = cv2.morphologyEx(imagem_abertura, cv2.MORPH_CLOSE, kernel)

    fim = time.time()
    print(f"[LOG] Tempo de processamento: {fim - inicio:.2f} segundos\n")

    # Exibir resultados
    imagens = [
        imagem_original,
        imagem_denoised,
        imagem_bin,
        imagem_abertura,
        imagem_fechamento
    ]
    titulos = [
        "Imagem Original",
        "Após Filtro Gaussiano (Redução de Ruído)",
        "Binarização com Otsu",
        "Abertura (Morfologia)",
        "Fechamento (Morfologia)"
    ]
    exibir_resultados(imagens, titulos, titulo_geral="Projeto 4 - Pré-processamento")

    print("================== FINALIZANDO PROJETO 4 ====================\n")

# Menu principal
def menu():
    while True:
        limpar_terminal()
        exibir_cabecalho()
        print("\nEscolha o projeto a executar:")
        print("1️⃣  Projeto 1 - Recorte e Colagem com fundo verde")
        print("2️⃣  Projeto 2 - Detecção de Círculos")
        print("3️⃣  Projeto 3 - Segmentação de Folha (Regiões Saudáveis e Danificadas)")
        print("4️⃣  Projeto 4 - Segmentação de imagens médicas")
        print("0️⃣  Sair\n")

        opcao = input("Digite a opção desejada: ").strip()

        if opcao == '1':
            projeto_1()
            input("\nPressione ENTER para retornar ao menu...")
        elif opcao == '2':
            projeto_2()
            input("\nPressione ENTER para retornar ao menu...")
        elif opcao == '3':
            projeto_3()
            input("Pressione ENTER para retornar ao menu...")
        elif opcao == '4':
            projeto_4()
            input("Pressione ENTER para retornar ao menu...")
        elif opcao == '0':
            print("Encerrando aplicação...")
            break
        else:
            print("\n[ERRO] Opção inválida. Tente novamente.")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    menu()