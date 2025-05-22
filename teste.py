import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem
img = cv2.imread('datasets/exemplo-01.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Primeira Passagem ---
blur1 = cv2.GaussianBlur(img_gray, (5, 5), 0)
thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY_INV, 15, 10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_clean1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean1 = cv2.morphologyEx(mask_clean1, cv2.MORPH_DILATE, kernel, iterations=1)
img_inpaint1 = cv2.inpaint(img, mask_clean1, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# --- Segunda Passagem ---
img_gray2 = cv2.cvtColor(img_inpaint1, cv2.COLOR_BGR2GRAY)
blur2 = cv2.GaussianBlur(img_gray2, (5, 5), 0)
thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY_INV, 15, 10)
mask_clean2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean2 = cv2.morphologyEx(mask_clean2, cv2.MORPH_DILATE, kernel, iterations=1)
img_inpaint2 = cv2.inpaint(img_inpaint1, mask_clean2, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

# Mostrar resultados
plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("Blur 1")
plt.imshow(blur1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Máscara 1")
plt.imshow(mask_clean1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Inpaint 1")
plt.imshow(cv2.cvtColor(img_inpaint1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 4, 5)
plt.title("Blur 2")
plt.imshow(blur2, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("Máscara 2")
plt.imshow(mask_clean2, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title("Inpaint 2")
plt.imshow(cv2.cvtColor(img_inpaint2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
