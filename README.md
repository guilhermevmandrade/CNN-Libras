# Libras Convolutional Neural Network

## Descrição do Projeto

Este projeto consiste em um sistema de reconhecimento de gestos estáticos da Língua Brasileira de Sinais (Libras) utilizando uma Rede Neural Convolucional (CNN). O objetivo principal é reduzir a barreira de comunicação entre surdos e não surdos, permitindo a tradução automática de gestos para o Português.

O modelo foi treinado com um banco de dados contendo imagens reais de mãos formando sinais em Libras. Essas imagens passaram por técnicas de pré-processamento, como normalização e aumento de dados (Data Augmentation), para melhorar o desempenho da rede neural. Após o treinamento, o modelo foi avaliado utilizando métricas como acurácia, precisão e matriz de confusão.

## Tecnologias Utilizadas

O projeto foi desenvolvido utilizando Python 3.12.3 e as seguintes bibliotecas:

- **TensorFlow 2.16.1** – Framework para treinamento e inferência da rede neural.
- **Keras 3.3.3** – Interface de alto nível para construção da CNN.
- **OpenCV 4.10.0.82** – Captura e pré-processamento das imagens da câmera.
- **NumPy 1.26.4** – Manipulação de arrays e matrizes numéricas.
- **Scikit-learn 1.5.0** – Ferramentas para avaliação do modelo e matriz de confusão.

## Construção do Banco de Dados

O conjunto de dados utilizado foi criado a partir de imagens capturadas por câmera com fundo branco, representando 20 sinais do alfabeto em Libras (excluindo H, J, X e Z, pois seus gestos são dinâmicos). O banco contém 30.000 imagens, divididas da seguinte forma:

- **80%** para treinamento (sendo 20% usado para validação).
- **20%** para testes.

Cada imagem tem resolução de **64x64 pixels**. Durante o treinamento, foram aplicadas transformações aleatórias, como espelhamento, rotação e zoom, para aumentar a diversidade dos dados.

## Estrutura da Rede Neural Convolucional (CNN)

A CNN desenvolvida possui a seguinte arquitetura:

1. **Camada de entrada**: Recebe imagens RGB de 64x64 pixels.
2. **Camadas convolucionais e de pooling**:
   - **1ª camada**: 32 filtros 3x3 + ReLU + MaxPooling 2x2.
   - **2ª camada**: 64 filtros 3x3 + ReLU + MaxPooling 2x2.
   - **3ª camada**: 128 filtros 3x3 + ReLU + MaxPooling 2x2.
3. **Camada Flatten**: Transforma as matrizes em um vetor unidimensional.
4. **Camada densamente conectada**: 256 neurônios com ReLU.
5. **Camada Dropout**: 20% para evitar overfitting.
6. **Camada de saída**: 20 neurônios (um para cada classe) com ativação Softmax.

## Ambiente Virtual Python e Instalação de Pacotes

Este guia irá orientá-lo sobre o processo de configuração de um ambiente virtual Python usando o Python e a instalação dos pacotes necessários.

### Pré-requisitos

- `Python` (Desenvolvido e testado no Python 3.12.3) deve estar instalado. 
- `pip` (instalador de pacotes Python) deve estar instalado.

### Passos

1. **Criar um Ambiente Virtual**

   ```bash
   python -m venv .venv
   ```

2. **Ativar o Ambiente Virtual**

    No Windows:

    ```bash
    .\venv\Scripts\activate
    ```

    No Unix ou MacOS:

    ```bash
    source venv/bin/activate
    ```

3. **Instalar Pacotes do requirements.txt**

    ```bash
    pip install -r requirements.txt
    ```

## Executando Scripts do Projeto

Depois de configurar o ambiente virtual e instalar os pacotes necessários, você pode executar os scripts do projeto.

1. **Executar Treinamento**

   ```bash
   python cnn_libras/executar_treino.py
   ```

2. **Executar Testes**

    ```bash
    python cnn_libras/executar_predicao.py
    ```

3. **Executar Reconhecimento via Webcam**

    ```bash
    python cnn_libras/reconhecimento_webcam.py
    ```
