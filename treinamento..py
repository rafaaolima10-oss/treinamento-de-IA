#IMPORTANTE: Antes de começar, instale os pacotes contidos em requirements.txt
#
#python -m pip install -r requirements.txt

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# IMPORTANTE: Instruções para Repositório de Dados
# 1. Criar um documento no Google sheet
# 2. Inserir dados com estes cabeçalhos: faltas, nota_1, nota_2, resultado 
# 3. adicionar de 10 a 15 inhas conforme o exemplo abaixo:
#    8,2,5,4.5,reprovado
#    2,5.0,7.0,aprovado
#    ....(adicionar até 15 linhas)
# 4. Ir em "Arquivo" -> "Compartlhar" -> "Publicar na Web"
# 5. Selecionar "Valores separados por virgula (.csv)" e clicar em "Publicar"
# 6. COPIAR O LINK GERANDO E COLAR ABAIXO:
URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSv0J3C7HWF5BB4rnNUHIhXOyModQITdNC_m0_c_zuKNFBqCnr5WdNKnW366r4UqYlKiHaQawmBAFPi/pub?output=csv"

ARQUIVO_MODELO = 'modelo_desempenho.pkl'

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS}...")
    # Covertendo dados do arquivo csv em estrutura de dados (datafrane)
    data = pd.read_csv(URL_DADOS)
    print("--- Dados Carregados ---")
    print(data.head())
 
    print("--- Preparando dados para o treino ---")
    features = ["faltas", "nota_1", "nota_2"]
    target = "resultado"
    #X recebe as características e Y recebe o
    #rótulo (chamado de alvo ou target, neste caso)
    X = data[features]
    Y = data[target]
 
    #Instancia "modelo" com 3 vizinhos (n_neighbors=3)
    modelo = KNeighborsClassifier(n_neighbors=3)
    #Para treinar o modelo usa-se o método .fit() com as entradas X e Y
    modelo.fit(X, Y)
    print(f"---Modelo Treinado! Classes: {modelo.classes_}---")

    #Salvando o modelo treinado no arquivo cujo esta armazenao na 
    #constante ARQUIVO_MODELO
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"--- Modelo salvo com sucesso em '{ARQUIVO_MODELO}'---")

if __name__ == "__main__":
    treinar_modelo()