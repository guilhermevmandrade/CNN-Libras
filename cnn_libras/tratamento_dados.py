import os
import tensorflow as tf
from keras import layers
from tensorflow import keras

def carregar_datasets_treino(caminho, altura_img, largura_img, tamanho_lote, validation_split=0.2, seed=123):
    """
    Carrega os conjuntos de dados de treinamento e validação a partir do diretório especificado.

    Args:
        caminho (str): Caminho para o diretório contendo os conjuntos de dados.
        altura_img (int): Altura das imagens.
        largura_img (int): Largura das imagens.
        tamanho_lote (int): Tamanho do lote para carregar os dados.
        validation_split (float): Proporção do conjunto de validação em relação ao conjunto de treinamento.
        seed (int): Semente para garantir a reprodutibilidade.

    Returns:
        tuple: Tupla contendo os conjuntos de dados de treinamento, validação e os nomes das classes.
    """
    conjunto_treinamento = tf.keras.utils.image_dataset_from_directory(
        caminho,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(altura_img, largura_img),
        batch_size=tamanho_lote
    )

    conjunto_validacao = tf.keras.utils.image_dataset_from_directory(
        caminho,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(altura_img, largura_img),
        batch_size=tamanho_lote
    )

    nomes_classes = conjunto_treinamento.class_names

    conjunto_treinamento = conjunto_treinamento.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    conjunto_validacao = conjunto_validacao.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return conjunto_treinamento, conjunto_validacao, nomes_classes

def carregar_dataset_teste(caminho, altura_img, largura_img, tamanho_lote):
    """
    Carrega o conjunto de dados de teste a partir do diretório especificado.

    Args:
        caminho (str): Caminho para o diretório contendo o conjunto de dados de teste.
        altura_img (int): Altura das imagens.
        largura_img (int): Largura das imagens.
        tamanho_lote (int): Tamanho do lote para carregar os dados.

    Returns:
        Dataset: Conjunto de dados de teste carregado.
    """
    conjunto_teste = tf.keras.utils.image_dataset_from_directory(
        caminho,
        image_size=(altura_img, largura_img)
    )

    return conjunto_teste

def carregar_data_augmentation(altura_img, largura_img):
    """
    Carrega as camadas de aumento de dados para aplicar durante o treinamento.

    Args:
        altura_img (int): Altura das imagens.
        largura_img (int): Largura das imagens.

    Returns:
        Sequential: Camadas de aumento de dados configuradas.
    """
    aumento_dados = keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(altura_img, largura_img, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    
    return aumento_dados
