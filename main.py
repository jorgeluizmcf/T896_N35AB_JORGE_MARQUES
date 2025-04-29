import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem do PROJETO 1 em Escala de Cinza
img = cv2.imread('datasets/projeto_1/img_fundo_verde_1.jpg')
background = cv2.imread('datasets/projeto_1/WhatsApp Image 2025-04-29 at 20.18.32.jpeg')

# Converte para HSV (mais fácil para segmentar cor)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define faixa de verde em HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Cria máscara para o verde
mask = cv2.inRange(hsv, lower_green, upper_green)

# Inverte a máscara: onde está branco é o que queremos manter
mask_inv = cv2.bitwise_not(mask)

# Aplica a máscara no canal BGR
result_rgb = cv2.bitwise_and(img, img, mask=mask_inv)

# Adiciona canal alpha com base na máscara
b, g, r = cv2.split(result_rgb)
alpha = mask_inv
rgba = cv2.merge((b, g, r, alpha))

# Salva como PNG com transparência
recorte = rgba

# Carrega o background e redimensiona
background = cv2.resize(background, (recorte.shape[1], recorte.shape[0]))

# Separa canais RGBA
b, g, r, a = cv2.split(recorte)
overlay_rgb = cv2.merge((b, g, r))

# Normaliza alpha para blending
alpha = a.astype(float) / 255

# Mescla manualmente com background
foreground = overlay_rgb.astype(float)
background = background.astype(float)

for c in range(3):
    background[:, :, c] = alpha * foreground[:, :, c] + (1 - alpha) * background[:, :, c]

# Converte para uint8 e salva ou exibe
composited = background.astype(np.uint8)
#cv2.imwrite('resultado_final.png', composited)

img_show = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
plt.imshow(img_show)
plt.title("Resultado Final - Pessoa sobre Background")
plt.axis('off')
plt.show()