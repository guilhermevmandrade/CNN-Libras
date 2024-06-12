import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from librasCNN.letras_dicionario import letras

# Invertendo o dicionário para facilitar a busca por índices
indices_letras = {v: int(k) for k, v in letras.items()}
lista_letras = [letras[str(i)] for i in range(len(letras))]

def calcula_matriz_confusao(y_real: list, y_pred: list):
    # Quantidade de classes do problema
    qtde_classes = len(letras)

    # Inicializa a matriz de confusão com zeros
    matriz = [[0 for _ in range(qtde_classes)] for _ in range(qtde_classes)]

    # Para cada letra da lista de real, busca a letra correspondente no vetor de predição
    for letra_real, letra_pred in zip(y_real, y_pred):
        indice_real = indices_letras[letra_real]
        indice_pred = indices_letras[letra_pred]

        # Incrementa o valor da posição da matriz de confusão
        matriz[indice_real][indice_pred] += 1

    return matriz

def plota_matriz_de_confusao(y_real: list = None, y_pred: list = None, title: str = "Matriz de confusão"):
    sns.set_theme(style="white")

    matriz = calcula_matriz_confusao(y_real, y_pred)

    df_matriz = pd.DataFrame(matriz, index=lista_letras, columns=lista_letras)

    graf = sns.heatmap(df_matriz, annot=True, fmt='d', cbar=False)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig('librasMLP/graficos/grafico.png', bbox_inches='tight')
    plt.show()

def decodifica_vetor_predicao(predicoes: list):
    letras = []
    for predicao in predicoes:
        # Busca o índice do maximo valor do vetor
        indice_max = predicao.index(max(predicao))

        # Adiciona a letra correspondente ao índice
        letras.append(letras[str(indice_max)])

    return letras
