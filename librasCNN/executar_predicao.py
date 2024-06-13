import os
import numpy as np
import tensorflow as tf
from letras_dicionario import letras
from keras.api.models import load_model
from sklearn.metrics import confusion_matrix
from plotar_graficos import plotar_metricas_teste

# Definir parâmetros de configuração
tamanho_lote = 32
altura_imagem, largura_imagem = 64, 64
caminho_teste = os.path.join('dados/teste')

# Carregar o modelo pré-treinado
modelo = load_model('librasCNN/modelos/libras.keras')

# Listas para armazenar as classes verdadeiras e preditas
classe_verdadeira = []
classe_predita = []

# Iterar por todas as pastas na pasta de teste
for nome_pasta in os.listdir(caminho_teste):
    caminho_pasta = os.path.join(caminho_teste, nome_pasta)

    # Verificar se o item é uma pasta
    if not os.path.isdir(caminho_pasta):
        continue
    
    # Iterar por todas as imagens na pasta
    for nome_arquivo in os.listdir(caminho_pasta):
        caminho_imagem = os.path.join(caminho_pasta, nome_arquivo)

        # Verificar se o item é um arquivo
        if not os.path.isfile(caminho_imagem):
            continue
        
        try:
            # Carregar e processar a imagem
            img = tf.keras.utils.load_img(caminho_imagem, target_size=(altura_imagem, largura_imagem))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Fazer predição da imagem usando o modelo
            predictions = modelo.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            # Armazenar as classes verdadeira e predita
            classe_verdadeira.append(nome_pasta)
            classe_predita.append(letras[str(np.argmax(score))])
        except Exception as e:
            print(f"Erro ao processar a imagem {nome_arquivo}: {e}")

# Obter a lista de todas as classes (letras)
lista_letras = list(letras.values())

# Calcular a matriz de confusão
matriz_de_confusao = confusion_matrix(classe_verdadeira, classe_predita, labels=lista_letras)

# Plotar as métricas de teste usando a função fornecida
plotar_metricas_teste(matriz_de_confusao, lista_letras)
