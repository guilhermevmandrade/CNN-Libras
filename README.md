# Libras Convolutional Neural Network

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