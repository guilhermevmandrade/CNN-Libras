import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from calcula_metricas import acuracia, especificidade, f_measure, precisao, revocacao

def plotar_metricas_treino(intervalo_epocas, classificador):
    """
    Plota as métricas de treinamento e validação ao longo das épocas.

    Parâmetros:
        intervalo_epocas (list): Intervalo de épocas.
        classificador (tf.keras.callbacks.History): Objeto History retornado pelo método fit do modelo.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 10))
    plt.plot(intervalo_epocas, classificador.history["loss"], label="perda_treinamento")
    plt.plot(intervalo_epocas, classificador.history["val_loss"], label="perda_validacao")
    plt.plot(intervalo_epocas, classificador.history["accuracy"], label="acuracia_treinamento")
    plt.plot(intervalo_epocas, classificador.history["val_accuracy"], label="acuracia_validacao")
    plt.xlabel("Época")
    plt.ylabel("Perda/Acurácia")
    plt.legend()
    plt.title('Perda e Acurácia de Treinamento e Validação')
    plt.savefig('librasCNN/graficos/metricas_treino.png', bbox_inches='tight')
    plt.show()
    
def plotar_metricas_teste(matriz_de_confusao, lista_letras):
    """
    Calcula e plota as métricas de avaliação do modelo.

    Parâmetros:
        matriz_de_confusao (array): Matriz de confusão.
        lista_letras (list of str): Lista das letras/classes.
    """
    # Calcular métricas a partir da matriz de confusão
    FP = matriz_de_confusao.sum(axis=0) - np.diag(matriz_de_confusao)
    FN = matriz_de_confusao.sum(axis=1) - np.diag(matriz_de_confusao)
    TP = np.diag(matriz_de_confusao)
    TN = matriz_de_confusao.sum() - (FP + FN + TP)

    TPR = revocacao(TP, FN)
    TNR = especificidade(TN, FP)
    PPV = precisao(TP, FP)
    ACC = acuracia(TP, TN, FP, FN)
    FMEASURE = f_measure(TP, FP, FN)

    classes = ['Revocação (TPR)', 'Especificidade (TNR)', 'Precisão (PPV)', 'Acurácia (ACC)', 'F-Measure']
    metricas_letras = [TPR, TNR, PPV, ACC, FMEASURE]
    metricas_media = [TPR.mean(), TNR.mean(), PPV.mean(), ACC.mean(), FMEASURE.mean()]

    plotar_matriz_de_confusao(matriz_de_confusao, lista_letras)
    plotar_metricas_letras(classes, metricas_letras, lista_letras)
    plotar_metricas_media(classes, metricas_media)
    
def plotar_metricas_media(classes, metricas):
    """
    Plota as métricas de avaliação do modelo de teste.

    Parâmetros:
        classes (list of str): Lista com os nomes das métricas.
        metricas (list of float): Lista com os valores das métricas.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 10))
    
    # Criar o gráfico de barras
    plt.bar(classes, metricas, color='skyblue')
    
    # Configurações dos eixos e título
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    plt.title('Métricas de Avaliação do Modelo', pad=20)  # Adiciona espaço entre o título e o gráfico
    plt.ylim(0, 1)  # Definindo o limite do eixo y de 0 a 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adicionando linhas de grade horizontal
    
    # Adicionar os valores acima das barras com um deslocamento
    for i, valor in enumerate(metricas):
        plt.text(i, valor + 0.02, f'{valor:.4f}', ha='center', va='bottom', fontsize=10)

    # Ajustar automaticamente o layout
    plt.tight_layout()
    
    # Salvar e mostrar a figura
    plt.savefig('librasCNN/graficos/metricas_teste_media.png', bbox_inches='tight')
    plt.show()
    
def plotar_metricas_letras(classes, metricas, lista_letras):
    """
    Plota uma matriz de calor das métricas de avaliação do modelo para diferentes letras.

    Parâmetros:
    - classes (list of str): Lista com os nomes das métricas.
    - metricas (list of list of float): Lista de listas com os valores das métricas para cada letra.
    - lista_letras (list of str): Lista com as letras avaliadas.
    """
    # Transpor a matriz de dados para inverter os eixos
    data = np.array([[metric[i] for metric in metricas] for i in range(len(lista_letras))]).T
    
    fig, ax = plt.subplots(figsize=(10, 6))  
    cax = ax.matshow(data, cmap='coolwarm')

    plt.style.use("ggplot")
    plt.xticks(range(len(lista_letras)), lista_letras, rotation=45)  # Letras no eixo x
    plt.yticks(range(len(classes)), classes)  # Classes no eixo y
    plt.colorbar(cax)

    # Adicionar os valores nas células da matriz de calor
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.xlabel('Letras')  # Letras no eixo x
    plt.ylabel('Métricas')  # Métricas no eixo y
    plt.title('Métricas de Avaliação do Modelo')
    plt.tight_layout()  # Ajustar automaticamente o layout
    plt.savefig('librasCNN/graficos/metricas_teste_letras.png', bbox_inches='tight')
    plt.show()
    
def plotar_matriz_de_confusao(matriz_de_confusao, classes):
    """
    Plota a matriz de confusão do modelo.

    Parâmetros:
        matriz_de_confusao (array): Matriz de confusão.
        classes (list of str): Lista com os nomes das classes.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_de_confusao, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig('librasCNN/graficos/matriz_de_confusao.png', bbox_inches='tight')
    plt.show()
