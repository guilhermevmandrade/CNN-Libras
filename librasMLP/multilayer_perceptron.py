import numpy as np
import pickle

class MultilayerPerceptron:
    def __init__(self, num_entradas: int, num_camada_oculta: int, num_saidas: int, num_epocas: int, taxa_aprendizado: float):
        # Inicializa os parâmetros da rede neural: número de entradas, neurônios na camada oculta, saídas, épocas de treinamento e taxa de aprendizado
        self.num_entradas = num_entradas
        self.num_camada_oculta = num_camada_oculta
        self.num_saidas = num_saidas
        self.num_epocas = num_epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.inicializa_pesos()

    def inicializa_pesos(self):
        # Inicializa os pesos e vieses com valores aleatórios entre -0.5 e 0.5 para a camada oculta e de saída
        self.pesos_oculta = np.random.uniform(-0.5, 0.5, (self.num_entradas, self.num_camada_oculta))
        self.vieses_oculta = np.random.uniform(-0.5, 0.5, self.num_camada_oculta)
        self.pesos_saida = np.random.uniform(-0.5, 0.5, (self.num_camada_oculta, self.num_saidas))
        self.vieses_saida = np.random.uniform(-0.5, 0.5, self.num_saidas)
        self.convergencia = 0.001  # Define o critério de convergência para parada antecipada do treinamento

    def funcao_sigmoide(self, x: float):
        # Função de ativação sigmoide, usada para introduzir não-linearidade no modelo
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoide(self, sigmoide_x: float):
        # Derivada da função sigmoide, usada para calcular os gradientes durante o backpropagation
        return sigmoide_x * (1 - sigmoide_x)

    def executar(self, entradas: list):
        # Calcula a saída da camada oculta e da camada final
        soma_ponderada_oculta = self.calcula_soma_ponderada(entradas, self.pesos_oculta, self.vieses_oculta)
        saidas_oculta = [self.funcao_sigmoide(s) for s in soma_ponderada_oculta]  # Aplica a função de ativação sigmoide nas somas ponderadas da camada oculta
        
        soma_ponderada_final = self.calcula_soma_ponderada(saidas_oculta, self.pesos_saida, self.vieses_saida)
        saida_final = [self.funcao_sigmoide(s) for s in soma_ponderada_final]  # Aplica a função de ativação sigmoide nas somas ponderadas da camada final
        
        return saida_final, saidas_oculta

    def calcula_soma_ponderada(self, entradas: list, pesos: np.ndarray, vieses: np.ndarray):
        # Calcula a soma ponderada das entradas para cada neurônio em uma camada
        soma_ponderada = []
        for j in range(len(vieses)):  # Itera sobre cada neurônio na camada
            soma = vieses[j]  # Começa com o valor do viés do neurônio
            for i in range(len(entradas)):  # Adiciona a contribuição de cada entrada ponderada pelo peso correspondente
                soma += entradas[i] * pesos[i][j]
            soma_ponderada.append(soma)  # Adiciona a soma ponderada à lista de somas ponderadas
        return soma_ponderada

    def calcula_deltas_saida(self, erros_saida: list, saidas_obtidas: list):
        # Calcula os deltas para a camada de saída usando o erro e a derivada da função de ativação
        return [erro * self.derivada_sigmoide(saida) for erro, saida in zip(erros_saida, saidas_obtidas)]

    def calcula_deltas_oculta(self, deltas_saida: list, saidas_oculta: list):
        # Calcula os deltas para a camada oculta
        deltas_oculta = []
        for j in range(self.num_camada_oculta):            
            delta = sum(deltas_saida[k] * self.pesos_saida[j][k] for k in range(self.num_saidas))
            deltas_oculta.append(delta * self.derivada_sigmoide(saidas_oculta[j]))
        return deltas_oculta

    def atualiza_pesos(self, entradas: list, saidas_oculta: list, deltas_oculta: list, deltas_saida: list):
        # Atualiza os pesos e vieses da camada de saída
        for k in range(self.num_saidas):
            for j in range(self.num_camada_oculta):
                self.pesos_saida[j][k] += self.taxa_aprendizado * deltas_saida[k] * saidas_oculta[j]  # Atualiza o peso com base no delta e na saída do neurônio
            self.vieses_saida[k] += self.taxa_aprendizado * deltas_saida[k]  # Atualiza o viés com base no delta

        # Atualiza os pesos e vieses da camada oculta
        for j in range(self.num_camada_oculta):
            for i in range(len(entradas)):
                self.pesos_oculta[i][j] += self.taxa_aprendizado * deltas_oculta[j] * entradas[i]  # Atualiza o peso com base no delta e na entrada
            self.vieses_oculta[j] += self.taxa_aprendizado * deltas_oculta[j]  # Atualiza o viés com base no delta

    def treinar(self, lista_entradas: list[list], saidas_esperadas: list[list]):
        # Treina a rede neural por um número definido de épocas
        for epoca in range(self.num_epocas):
            erro_total = 0
            for entradas, saidas_esperadas_neuronio in zip(lista_entradas, saidas_esperadas):
                # Executa a rede para calcular as saídas
                saidas_obtidas, saidas_oculta = self.executar(entradas)
                
                # Calcula o erro na saída comparando a saída esperada com a obtida
                erros_saida = [esperada - obtida for esperada, obtida in zip(saidas_esperadas_neuronio, saidas_obtidas)]
                erro_total += np.sum(np.abs(erros_saida))  # Soma os erros absolutos para obter o erro total

                # Calcula os deltas para as camadas de saída e oculta
                deltas_saida = self.calcula_deltas_saida(erros_saida, saidas_obtidas)
                deltas_oculta = self.calcula_deltas_oculta(deltas_saida, saidas_oculta)

                # Atualiza os pesos e vieses da rede neural
                self.atualiza_pesos(entradas, saidas_oculta, deltas_oculta, deltas_saida)

            # Imprime o erro total para a época atual
            print(f"Epoca: {epoca} - Erro: {erro_total}\n")
            
            # Verifica se a convergência foi atingida (mudança no erro total é menor que um limiar)
            if epoca > 0 and abs(erro_total - erro_total_anterior) < self.convergencia:
                print(f"Parada antecipada na época {epoca} devido à convergência do erro (mudança menor que {self.convergencia}).")
                break

            erro_total_anterior = erro_total  # Armazena o erro total para comparação na próxima época
            
    def salvar_modelo(self, caminho_arquivo: str):
        # Salva o modelo treinado em um arquivo usando pickle
        with open(caminho_arquivo, 'wb') as arquivo:
            pickle.dump(self, arquivo)
        print(f"Modelo salvo em {caminho_arquivo}")

    def carregar_modelo(caminho_arquivo: str):
        # Carrega o modelo treinado de um arquivo usando pickle
        with open(caminho_arquivo, 'rb') as arquivo:
            modelo = pickle.load(arquivo)
        print(f"Modelo carregado de {caminho_arquivo}")
        return modelo

