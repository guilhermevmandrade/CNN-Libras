import os
from PIL import Image
import numpy as np
import random
from librasCNN.letras_dicionario import letras

def transformar_imagem_para_lista(caminho_imagem):
    try:
        # Abre a imagem
        img = Image.open(caminho_imagem)
        # Converte a imagem para modo escala de cinza
        img = img.convert('L')
        # Redimensiona a imagem para 64x64
        img = img.resize((64, 64))
        # Converte a imagem para um array numpy
        img_array = np.array(img)
        # Converte os valores de escala de cinza para 0 (branco) e 1 (preto)
        array_transformado = np.where(img_array > 128, 0, 1)
        # Achata o array para obter uma lista de 4096 números
        lista_achatada = array_transformado.flatten().tolist()
        return lista_achatada
    except Exception as e:
        print(f"Erro ao processar a imagem {caminho_imagem}: {e}")
        return None

def codificar_letra(indice):
    # Cria uma lista de zeros
    one_hot = [0] * len(letras)
    # Define o índice a ser configurado como 1
    one_hot[indice] = 1
    return one_hot

def processar_pastas_de_imagens(pasta_raiz):
    lista_combinada = []

    # Verifica se o caminho da pasta raiz está correto
    if not os.path.isdir(pasta_raiz):
        print(f"A pasta raiz '{pasta_raiz}' não é um diretório ou não existe.")
        return lista_combinada

    # Percorre cada pasta (cada letra) na pasta raiz
    for pasta_letra in os.listdir(pasta_raiz):
        caminho_pasta_letra = os.path.join(pasta_raiz, pasta_letra)

        # Verifica se o caminho é um diretório
        if os.path.isdir(caminho_pasta_letra):
            # Percorre cada imagem na pasta da letra
            for nome_imagem in os.listdir(caminho_pasta_letra):
                caminho_imagem = os.path.join(caminho_pasta_letra, nome_imagem)

                # Verifica se o caminho é um arquivo
                if os.path.isfile(caminho_imagem):
                    lista_imagem = transformar_imagem_para_lista(caminho_imagem)
                    if lista_imagem is not None:
                        rotulo = codificar_letra(pasta_letra)
                        # Adiciona uma tupla da lista de imagens e rótulo à lista combinada
                        lista_combinada.append((lista_imagem, rotulo))

    # Embaralha a lista combinada
    random.shuffle(lista_combinada)

    return lista_combinada
