import cv2
import numpy as np
import sys

def region_growing(image, seed, tolerance):
    rows, cols = image.shape
    segmented = np.zeros_like(image)
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if (x >= 0 and x < rows and y >= 0 and y < cols):
            if abs(int(image[x, y]) - int(image[seed])) < tolerance and segmented[x, y] == 0:
                segmented[x, y] = 255
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))
    return segmented

def select_seed(event, x, y, flags, param):
    global seed
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (y, x)

def main(argv):
    #USO:
    # python main.py __nombre-con-extension__ __tolerancia__
    # Cargar la imagen
    if len(argv) < 2:
        print("USO: python main.py __nombre-con-extension__ __tolerancia__ ")
        exit(1)
    else:
        name = argv[1]
        tolerance = int(argv[2])
        image = cv2.imread(name, cv2.IMREAD_ANYCOLOR)
        image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Seleccione la semilla', image)
        global seed #definicion de variable global para la posicion de la semilla seleccionada
        seed = None #inicio de semilla en dato nulo
        # Configurar la ventana para que el usuario seleccione la semilla haciendo clic
        cv2.setMouseCallback('Seleccione la semilla', select_seed)
        while seed is None:
            cv2.waitKey(10) #esperar input de usuario desde la imagen para continuar

        # Aplicar el crecimiento de regiones
        segmented = region_growing(image_g, seed, tolerance)
        # Guardar la imagen segmentada
        cv2.imwrite('segmented_image.png', segmented)
        # Mostrar la imagen segmentada
        cv2.imshow('Imagen Segmentada', segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
