import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem
img = cv2.imread('datasets/exemplo-05.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Primeira Passagem ---
blur1 = cv2.GaussianBlur(img_gray, (5, 5), 0)
thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY_INV, 15, 10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_clean1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean1 = cv2.morphologyEx(mask_clean1, cv2.MORPH_DILATE, kernel, iterations=1)
img_inpaint1 = cv2.inpaint(img, mask_clean1, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# Mostrar resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("Blur")
plt.imshow(blur1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Máscara")
plt.imshow(mask_clean1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Inpaint")
plt.imshow(cv2.cvtColor(img_inpaint1, cv2.COLOR_BGR2RGB))
plt.axis('off')


plt.tight_layout()
plt.show()
