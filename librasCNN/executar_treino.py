import os

from matplotlib import pyplot as plt
from convolutional_neural_network import ConvolutionalNeuralNetwork
from tratamento_dados import carregar_data_augmentation, carregar_datasets_treino
from librasCNN.plotar_graficos import plotar_metricas_treino
from keras import layers
from tensorflow import keras

epocas = 16
tamanho_lote = 32
altura_imagem, largura_imagem = 64, 64
caminho_treino = os.path.join('dados/treino')

# Carregar datasets
conjunto_treinamento, conjunto_validacao, nomes_classes = carregar_datasets_treino(caminho_treino, altura_imagem, largura_imagem, tamanho_lote)
aumento_dados = carregar_data_augmentation(altura_imagem, largura_imagem)
num_classes = len(nomes_classes)

aumento_dados = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(altura_imagem, largura_imagem, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)
   
plt.figure(figsize=(10, 10))
for images, _ in conjunto_treinamento.take(1):
    for i in range(9):
        augmented_images = aumento_dados(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.savefig('librasCNN/graficos/aumento_dados.png', bbox_inches='tight')        

# Construir e treinar o modelo
modelo = ConvolutionalNeuralNetwork(num_classes, aumento_dados, altura_imagem, largura_imagem)
classificador = modelo.treinar_modelo(conjunto_treinamento, conjunto_validacao, epocas)

# Salvar o modelo
caminho_modelo = 'librasCNN/modelos/libras.keras'
modelo.salvar_modelo(caminho_modelo)

# Plotar as métricas
plotar_metricas_treino(range(epocas), classificador)