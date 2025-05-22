import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem
img = cv2.imread('datasets/exemplo-01.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Passo 1: Suavizar a imagem para remover pequenos detalhes
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Passo 2: Realçar as áreas rasuradas (assumindo que são mais escuras)
# Aplicar limiarização adaptativa para pegar manchas escuras
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 15, 10)

# Passo 3: Remover ruídos pequenos (morfologia)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel, iterations=1)

# Passo 4: Aplicar a máscara na imagem original (opcional)
img_inpaint = cv2.inpaint(img, mask_clean, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Mostrar resultados
plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title("Máscara de Rasuras blur")
plt.imshow(blur, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title("Máscara de Rasuras thresh")
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title("Máscara de Rasuras")
plt.imshow(mask_clean, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title("Imagem Restaurada")
plt.imshow(cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()