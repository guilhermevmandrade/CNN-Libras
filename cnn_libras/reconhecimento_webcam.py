import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image
from letras_dicionario import letras 

# Definindo as dimensões da imagem
imagem_x, imagem_y = 64, 64

# Carregando o modelo treinado
classificador = load_model('cnn_libras/modelos/libras.keras')

def previsor():
    """
    Função para prever a letra baseada na imagem capturada.
    """
    # Carregando a imagem para teste
    imagem_teste = image.load_img('cnn_libras/capturas/img.png', target_size=(imagem_x, imagem_y))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste = np.expand_dims(imagem_teste, axis=0)
    
    # Fazendo a predição
    resultado = classificador.predict(imagem_teste)
    
    # Encontrando a classe com a maior probabilidade
    maior_prob, indice_classe = -1, -1
    for i in range(len(letras)):
        if resultado[0][i] > maior_prob:
            maior_prob = resultado[0][i]
            indice_classe = i
    
    return [resultado, letras[str(indice_classe)]]

# Iniciando a captura de vídeo
camera = cv2.VideoCapture(0)
contador_imagem = 0
texto_imagem = ['', '']

while True:
    # Capturando frame da câmera
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    # Desenhando o retângulo para ROI
    img = cv2.rectangle(frame, (450,0), (0,450), (0, 255, 0), thickness=1, lineType=8, shift=0)
    imcrop = img[2:448, 2:448]

    # Criando a imagem de saída para mostrar o texto
    output = np.ones((150, 150, 3)) * 255  # Imagem 150x150 com fundo branco
    cv2.putText(output, str(texto_imagem[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))

    # Mostrando a ROI e a predição
    cv2.imshow("ROI", imcrop)
    cv2.imshow("PREDICT", output)

    # Convertendo a ROI para escala de cinza
    img_cinza = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
    
    # Salvando a imagem em escala de cinza
    nome_imagem = "cnn_libras/capturas/img.png"
    img_salva = cv2.resize(img_cinza, (imagem_x, imagem_y))
    cv2.imwrite(nome_imagem, img_salva)
    
    # Fazendo a predição e atualizando o texto
    texto_imagem = previsor()

    # Saindo do loop ao pressionar 'ESC'
    if cv2.waitKey(1) == 27:
        break

# Liberando a câmera e destruindo as janelas
camera.release()
cv2.destroyAllWindows()
