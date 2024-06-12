import os
import pandas as pd
from librasMLP.multilayer_perceptron import MultilayerPerceptron
from visualizacao import decodifica_vetor_predicao, plota_matriz_de_confusao
from librasMLP.tratamento_dados import processar_pastas_de_imagens
  
perceptron = MultilayerPerceptron(num_entradas=4096, num_camada_oculta=512, num_saidas=6, num_epocas=1000, taxa_aprendizado=0.45)

dados_treino = processar_pastas_de_imagens(r'dados/treino')
lista_entradas_treino = [item[0] for item in dados_treino]
saidas_esperadas_treino = [item[1] for item in dados_treino]

perceptron.treinar(lista_entradas_treino, saidas_esperadas_treino)
perceptron.salvar_modelo(r'librasMLP/modelos/modelo_treinado_mlp.pk1')

dados_teste = processar_pastas_de_imagens(r'dados/teste')
lista_entradas_teste = [item[0] for item in dados_teste]
saidas_esperadas_teste = [item[1] for item in dados_teste]
saidas_obtidas = []

for i in range(len(lista_entradas_teste)):
    entrada = lista_entradas_teste[i]
    resultado, _ = perceptron.executar(entrada)
    saidas_obtidas.append(resultado)
    
letras_obtidas = decodifica_vetor_predicao(saidas_obtidas)
letras_esperadas = decodifica_vetor_predicao(saidas_esperadas_teste)

plota_matriz_de_confusao(letras_esperadas, letras_obtidas, "Matriz de confusão")

