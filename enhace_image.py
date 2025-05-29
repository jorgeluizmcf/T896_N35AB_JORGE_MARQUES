import cv2
import numpy as np

def enhance_image(I, B_shape, n):
    """
    Realce de imagem usando transformadas top-hat e bottom-hat em múltiplas escalas.
    
    Parâmetros:
        I: imagem de entrada (numpy array, em escala de cinza)
        B_shape: tupla (h, w) representando a forma inicial do elemento estruturante
        n: número de escalas
        
    Retorna:
        I_en: imagem realçada
    """
    # Inicializações
    TH = []
    BH = []
    STH = []
    SBH = []

    # Para cada escala
    for i in range(1, n + 1):
        kernel_size = (B_shape[0]*i, B_shape[1]*i)
        B_i = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        # Transformadas Top-Hat e Bottom-Hat
        top_hat = cv2.subtract(I, cv2.dilate(cv2.erode(I, B_i), B_i))
        bottom_hat = cv2.subtract(cv2.erode(cv2.dilate(I, B_i), B_i), I)

        TH.append(top_hat)
        BH.append(bottom_hat)

        if i > 1:
            STH.append(cv2.subtract(TH[i-1], TH[i-2]))
            SBH.append(cv2.subtract(BH[i-1], BH[i-2]))

    # Cálculo dos máximos nas escalas
    MTH = np.maximum.reduce(TH)
    MBH = np.maximum.reduce(BH)
    MSTH = np.maximum.reduce(STH) if STH else np.zeros_like(I)
    MSBH = np.maximum.reduce(SBH) if SBH else np.zeros_like(I)

    # Realce da imagem
    I_en = cv2.add(I, cv2.subtract(cv2.add(MTH, MSTH), cv2.add(MBH, MSBH)))

    return I_en


    

def main():
    # Carregar imagem em escala de cinza
    img = cv2.imread('datasets/exemplo-04.png', cv2.IMREAD_GRAYSCALE)

    # Realçar imagem com elemento estruturante inicial (3x3) e 4 escalas
    img_enhanced = enhance_image(img, B_shape=(3, 3), n=4)

    # Exibir resultados
    cv2.imshow('Original', img)
    cv2.imshow('Realçada', img_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
