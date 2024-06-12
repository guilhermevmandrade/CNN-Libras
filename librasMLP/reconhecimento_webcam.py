import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita as otimizações do OneDNN para o TensorFlow

import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image
from librasMLP.multilayer_perceptron import MultilayerPerceptron
from librasMLP.tratamento_dados import transformar_imagem_para_lista
from letras_dicionario import letras  # Importa o dicionário de letras

dimensao_imagem_x, dimensao_imagem_y = 64, 64  # Dimensões da imagem

def preditor():
    # Carrega a imagem para ser testada, redimensionando-a para 64x64
    imagem_teste = image.load_img('librasMLP/img.png', target_size=(dimensao_imagem_x, dimensao_imagem_y))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste = np.expand_dims(imagem_teste, axis=0)
    
    # Carrega o modelo treinado do perceptron multicamadas
    perceptron = MultilayerPerceptron.carregar_modelo('librasMLP/modelos/modelo_treinado_mlp.pk1')
    resultado = perceptron.executar(transformar_imagem_para_lista(imagem_teste))
    
    # Inicializa variáveis para armazenar o maior valor e o índice da classe correspondente
    maior, indice_classe = -1, -1

    # Encontra a classe com o maior valor de saída do perceptron
    for x in range(len(letras)):
        if resultado[0][x] > maior:
            maior = resultado[0][x]
            indice_classe = x
    
    # Retorna o resultado e a letra correspondente ao índice encontrado
    return [resultado, letras[str(indice_classe)]]

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)  # Inicia a captura de vídeo da câmera

    contador_img = 0
    texto_img = ['', '']
    while True:
        ret, frame = cam.read()  # Captura um frame da câmera
        frame = cv2.flip(frame, 1)  # Espelha o frame horizontalmente

        # Desenha um retângulo no frame
        img = cv2.rectangle(frame, (450,0), (0,450), (0, 255, 0), thickness=1, lineType=8, shift=0)
        regiao_interesse = img[2:448, 2:448]  # Recorta a região de interesse (ROI)

        # Cria uma imagem de saída de 150x150 com fundo branco
        saida_img = np.ones((150, 150, 3)) * 255  
        cv2.putText(saida_img, str(texto_img[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))  # Adiciona o texto na imagem

        cv2.imshow("ROI", regiao_interesse)  # Mostra a ROI
        cv2.imshow("PREDICT", saida_img)  # Mostra a imagem de saída

        img_cinza = cv2.cvtColor(regiao_interesse, cv2.COLOR_BGR2GRAY)  # Converte a ROI para escala de cinza
        
        nome_img = "librasMLP/img.png"  # Nome do arquivo de imagem a ser salvo
        salvar_img = cv2.resize(img_cinza, (dimensao_imagem_x, dimensao_imagem_y))  # Redimensiona a imagem para 64x64
        cv2.imwrite(nome_img, salvar_img)  # Salva a imagem redimensionada
        texto_img = preditor()  # Faz a predição da imagem salva
        print(str(texto_img[0]))  # Imprime o resultado da predição

        if cv2.waitKey(1) == 27:  # Se a tecla 'Esc' for pressionada, encerra o loop
            break

    cam.release()  # Libera a captura de vídeo
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV
