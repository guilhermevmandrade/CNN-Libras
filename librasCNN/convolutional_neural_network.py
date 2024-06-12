import os
import tensorflow as tf
from keras import layers
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping
from keras.api.metrics import *

class ConvolutionalNeuralNetwork:
    """
    Classe para construir, treinar e salvar modelos de redes neurais convolucionais.
    """

    def __init__(self, num_classes, aumento_dados, altura_img, largura_img):
        """
        Constrói o modelo da rede neural convolucional.

        Args:
            num_classes (int): Número de classes de saída.
            aumento_dados (Sequential): Camadas de aumento de dados para aplicar durante o treinamento.

        Returns:
            Sequential: Modelo construído da rede neural convolucional.
        """
        
        inputShape = (altura_img, largura_img, 3)
        
        self.modelo = Sequential([
            aumento_dados,
            layers.Rescaling(1./255, input_shape=inputShape),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.modelo.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

    def treinar_modelo(self, conjunto_treinamento, conjunto_validacao, epocas):
        """
        Treina o modelo da rede neural convolucional.

        Args:
            conjunto_treinamento (Dataset): Conjunto de dados de treinamento.
            conjunto_validacao (Dataset): Conjunto de dados de validação.
            epocas (int): Número de épocas de treinamento.
            callbacks (list): Lista de callbacks a serem aplicados durante o treinamento.

        Returns:
            History: Histórico de treinamento do modelo.
        """
        classificador = self.modelo.fit(
            conjunto_treinamento,
            validation_data=conjunto_validacao,
            epochs=epocas,
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)]
        )
        return classificador

    def salvar_modelo(self, caminho_arquivo):
        """
        Salva o modelo treinado em um arquivo.

        Args:
            modelo (Sequential): Modelo da rede neural convolucional treinado.
            caminho_arquivo (str): Caminho do arquivo onde o modelo será salvo.
        """
        diretorio = os.path.dirname(caminho_arquivo)
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
        self.modelo.save(caminho_arquivo)
