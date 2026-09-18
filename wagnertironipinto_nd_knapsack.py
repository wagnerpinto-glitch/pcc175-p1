"""
cd /home/wagner/Downloads/pcc175/fonte
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib numpy pandas python3-tk


cd /home/wagner/Downloads/pcc175/fonte && source .venv/bin/activate && code .

sudo apt update && sudo apt install -y python3-tk
/home/wagner/Downloads/pcc175/fonte/.venv/bin/python -c "import tkinter; print('Tkinter instalado com sucesso!')"

"""

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

def gera_imagem(fig, filename_prefix):
    # Criar uma pasta para as imagens no Drive, se não existir
    output_dir = './imagens'
    os.makedirs(output_dir, exist_ok=True)

    # Salva a figura
    filepath = os.path.join(output_dir, f'{filename_prefix}.png')
    fig.savefig(filepath)
    plt.close(fig) # Fechar a figura para liberar memória
    print(f"Imagem salva em: {filepath}")

"""Data da entrega : 16/09

Complete o código (marcado com None) e quando requisitado, escreva textos diretamente 
nos notebooks. 
Onde tiver None, substitua pelo seu código. 
Execute todo notebook e salve tudo em um PDF nomeado como "NomeSobrenome-P1.pdf".
"""

def no_dominated(pontos, cap=0):
   """"
   Determina os pontos não dominados de Pareto

   Input:
   pontos = conjunto de pontos [f1 f2]

   Output:
   nd_pontos = pontos não dominados [f1* f2*]

   """
   N = pontos.shape[0] # quantidade  de pontos
   nd_pontos = []
   solucao = []
   temp = []
   peso_acum = 0

   #cria um matriz_gabarito com todas as combinações possíveis de itens (0 ou 1)
   total_itens = pow(2, N)
   matriz_gabarito = np.array([[[int(bit)] for bit in f"{i:0{N}b}"] for i in range(total_itens)])

   #print(matriz_gabarito)
   item_validado = 0
   peso_acum = 0

   # faz a validação dos pesos de acordo com a capacidade
   while total_itens != item_validado:
      temp = []
      peso_acum = 0
      for k in range(N):         
         if matriz_gabarito[item_validado][k] == 1:
            #if pontos[k][1] <= cap:
            #   if peso_acum + pontos[k][1] <= cap:
                  peso_acum += pontos[k][1]
                  temp.append(pontos[k])

      #if peso_acum > 0 and peso_acum <= cap:
      solucao.append(temp)

      item_validado += 1

   # retirar itens repetidos indevidamente
   vistos = set()
   solucao_unica = []
   for combinacao in solucao:
      chave = tuple(tuple(np.asarray(item).tolist()) for item in combinacao)
      if chave not in vistos:
         vistos.add(chave)
         solucao_unica.append(combinacao)

   solucao = solucao_unica
   #print(f"Solucao: {solucao}")

   N = len(solucao)
   solucao_acum = []

   # agrupar meus valores e pesos acumulados por solução
   for item in solucao:
      N = len(item)
      valor_i = 0
      peso_i = 0   
      for i in range(N):
         valor_i += item[i][0]
         peso_i += item[i][1]

      solucao_acum.append(np.array([valor_i, peso_i]))

   #print(f"Solucao acumulada: {solucao_acum}")

   # identifica todas as soluções nd
   N = len(solucao_acum)
   dominado = [False] * N
   for i in range(N):
      for j in range(N):
         if i != j:
            if (solucao_acum[j][0] >= solucao_acum[i][0] and solucao_acum[j][1] <= solucao_acum[i][1]
               and (solucao_acum[j][0] > solucao_acum[i][0] or solucao_acum[j][1] < solucao_acum[i][1])):
               dominado[i] = True
               break

   # nd_pontos são os pontos agregados [valor, peso] de cada combinação não dominada
   nd_pontos = [solucao_acum[i] for i in range(N) if not dominado[i]]
   #print(f"ND Pontos: {nd_pontos}")

   return nd_pontos

def gera_grafico(pontos, nd_pontos, filename_prefix=None, cap=0):
   blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                           markersize=6, label='Itens da Instância')
   red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                           markersize=10, markeredgecolor='black', label='Soluções Não Dominadas (valor, peso)')

   fig = plt.figure(figsize=(8, 6))
   plt.scatter(pontos[:, 0], pontos[:, 1], color='blue', alpha=0.7)

   # Converter o nd_pontos
   nd_pontos = np.array(nd_pontos)
   if nd_pontos.size > 0 and nd_pontos.ndim == 2:
      plt.scatter(nd_pontos[:, 0], nd_pontos[:, 1], color='red', s=100, edgecolors='black', zorder=5)
   elif nd_pontos.size > 0 and nd_pontos.ndim == 1:
      # Caso retorne apenas 1 ponto isolado (1D)
      plt.scatter(nd_pontos[0], nd_pontos[1], color='red', s=100, edgecolors='black', zorder=5)

   titulo = 'Itens da Mochila e Solução Não Dominada'
   if cap:
      titulo += f' (Capacidade = {cap})'
   plt.title(titulo)
   plt.xlabel('Valor')
   plt.ylabel('Peso')
   plt.grid(True)
   plt.legend(handles=[blue_dot, red_dot])

   if filename_prefix is not None:
      gera_imagem(fig, filename_prefix)
   else:
      #print("Apresenta a imagem!")
      matplotlib.use('TkAgg')
      plt.show()


def gera_arquivo_csv(nome_arquivo):

   pontos_lista = [] 
   n_itens = 0
   capacidade = 0
   #count = 0
   nome_imagem = None
   
   with open(nome_arquivo, 'r') as f:
      for i, line in enumerate(f):
         linha = line.strip()
         #print(f"Linha {i+1}: {linha}")        

         # Ignora linhas vazias ou linhas de separação
         if not linha or 'knapPI' in linha:
            nome_imagem = linha.replace('.csv', '').strip()
            continue

         if linha.startswith('-'):
            if len(pontos_lista) > 0:
               pontos = np.array(pontos_lista)
               calcula_pontos(pontos, nome_imagem, capacidade)
               pontos_lista = []            
            n_itens = 0
            capacidade = 0
            continue

         valores = linha.split(',')
         #print(f"Valores na linha {i+1}: {valores}")

         # Verificar a quantidade de elementos a serem processados.
         if len(valores) == 1:
            valores = linha.split(' ')
            if valores[0].lower() == 'n':
               #print("Tamanho do conjunto de pontos (N) encontrado na linha:", valores[1])
               n_itens = int(valores[1])
               continue
            if valores[0].lower() == 'c':
               #print("Capacidade da mochila encontrada na linha:", valores[1])
               capacidade = int(valores[1])
               continue

         # Só processa se a linha tiver os 4 elementos esperados (ID, valor1, valor2, flag)
         if len(valores) == 4:

            try:
               num1 = int(valores[1])
            except ValueError:
               continue

            try:
               num2 = int(valores[2])
            except ValueError:
               continue

            if len(pontos_lista) < n_itens:
               pontos_lista.append([num1, num2])
            
            # Se atingir o tamanho exato da instância, calcula e limpa
            if len(pontos_lista) == n_itens:               
               pontos = np.array(pontos_lista)
               calcula_pontos(pontos, nome_imagem, capacidade)
               pontos_lista = []

def calcula_pontos(pontos, nome_imagem, cap=0):
   nd_pontos = no_dominated(pontos, cap)
   print(f"Pontos Não Dominados:\n", nd_pontos)
   gera_grafico(pontos, nd_pontos, nome_imagem, cap)
         
def main():
   nome_imagem = None
   pasta_script = os.path.dirname(os.path.abspath(__file__))
   nome_arquivo = None#os.path.join(pasta_script, 'instance', 'knapPI_16_20_1000_teste.csv')

   if nome_arquivo is not None:
      gera_arquivo_csv(nome_arquivo)
   else:
      print(f"Nome da imagem definido como: {nome_imagem}")

      #Teste da apostila
      #pontos = np.array([[8 ,5], [9, 2], [12, 1], [11, 2], [16, 2] ])#None # leia as instâncias de teste    
      #nome_imagem = "teste_apostila"

      # Teste básico
      #[valor, peso]
      pontos = np.array([[5, 3], [2, 7], [3, 1] ])
      print("\nPontos de Teste básico:\n", pontos)
      #nome_imagem = "teste_basico"
      cap = 10

      # Teste Array 2: Maior número de pontos aleatórios em um range diferente
      #np.random.seed(42) # Outra seed para reprodutibilidade
      #pontos = np.random.randint(1, 100, size=(100, 2))
      #cap = 100
      #nome_imagem = "teste_array_2"
      #print("\nPontos de Teste 2 (completos):\n", pontos)

      # execute sua função no_dominated
      calcula_pontos(pontos, nome_imagem, cap)

if __name__ == "__main__":
   main()