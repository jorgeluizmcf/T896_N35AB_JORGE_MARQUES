import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from enhace_image import enhance_image  # Sua função de realce morfológico

# Caminho para o diretório com as imagens
dataset_path = 'datasets/'

# Lista de imagens com extensões comuns
image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(dataset_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Erro ao carregar: {img_name}")
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Primeira Passagem ---
    blur1 = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean1 = cv2.morphologyEx(mask_clean1, cv2.MORPH_DILATE, kernel, iterations=1)
    img_inpaint1 = cv2.inpaint(img, mask_clean1, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # --- Realce ---
    img_enhanced = enhance_image(img_inpaint1, B_shape=(3, 3), n=4)

    # --- Exibir Resultados ---
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Processamento: {img_name}", fontsize=16)

    titles = ["Original", "Blur + Threshold", "Máscara", "Inpaint Inicial",
              "Realçada", "Resultado Final"]
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        blur1,
        mask_clean1,
        cv2.cvtColor(img_inpaint1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
    ]
    cmaps = ['gray' if len(im.shape) == 2 else None for im in images]

    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap=cmaps[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
