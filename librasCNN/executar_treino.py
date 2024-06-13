import os

from convolutional_neural_network import ConvolutionalNeuralNetwork
from tratamento_dados import carregar_data_augmentation, carregar_datasets_treino
from plotar_graficos import plotar_metricas_treino

epocas = 8
tamanho_lote = 32
altura_imagem, largura_imagem = 64, 64
caminho_treino = os.path.join('dados/treino')

# Carregar datasets
conjunto_treinamento, conjunto_validacao, nomes_classes = carregar_datasets_treino(caminho_treino, altura_imagem, largura_imagem, tamanho_lote)
aumento_dados = carregar_data_augmentation(altura_imagem, largura_imagem)
num_classes = len(nomes_classes)      

# Construir e treinar o modelo
modelo = ConvolutionalNeuralNetwork(num_classes, aumento_dados, altura_imagem, largura_imagem)
classificador = modelo.treinar_modelo(conjunto_treinamento, conjunto_validacao, epocas)

# Salvar o modelo
caminho_modelo = 'librasCNN/modelos/libras.keras'
modelo.salvar_modelo(caminho_modelo)

# Plotar as métricas
plotar_metricas_treino(range(epocas), classificador)