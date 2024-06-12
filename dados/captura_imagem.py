import cv2
import os

# Tamanho da imagem
largura_imagem, altura_imagem = 64, 64

# Teclas de controle
ESC = 27 
CAPTURAR = 32
dir_img_treino = 'dados/treino/'
dir_img_teste = 'dados/teste/'

# Quantidade de imagens para treino e teste
QTD_TREINO = 1200
QTD_TESTE = 300

def capturar_imagens(letra):
    """Captura imagens da webcam, salva em pastas de treino e teste."""
    
    # Cria pastas para armazenar as imagens de treino e teste, se não existirem.
    if not os.path.exists(dir_img_treino + letra):
        os.mkdir(dir_img_treino + letra)
    if not os.path.exists(dir_img_teste + letra):
        os.mkdir(dir_img_teste + letra)
    
    cam = cv2.VideoCapture(0)
    contador_total = 1
    nome_imagem_treino = 1
    nome_imagem_teste = 1
    
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Desenha o retângulo para a captura
        img = cv2.rectangle(frame, (450, 0), (0, 450), (0, 255, 0), thickness=1)
        resultado = img[2:448, 2:448]        

        cv2.imshow("Resultado", resultado)

        tecla = cv2.waitKey(1)

        if tecla == CAPTURAR:
            img_salva = cv2.resize(resultado, (largura_imagem, altura_imagem))
            if contador_total <= QTD_TREINO:
                nome_img = f"{dir_img_treino}{letra}/{nome_imagem_treino}.png"
                cv2.imwrite(nome_img, img_salva)
                print(f"{nome_img} capturado!")
                nome_imagem_treino += 1
            elif contador_total <= QTD_TREINO + QTD_TESTE:
                nome_img = f"{dir_img_teste}{letra}/{nome_imagem_teste}.png"
                cv2.imwrite(nome_img, img_salva)
                print(f"{nome_img} capturado!")
                nome_imagem_teste += 1
            contador_total += 1

            if contador_total > QTD_TREINO + QTD_TESTE:
                print('[INFO] Fim da captura')
                break

        elif tecla == ESC:
            break

    cam.release()
    cv2.destroyAllWindows()

# Solicita a letra a ser capturada
letra = input("Digite a letra para capturar imagens: ")
capturar_imagens(letra)
