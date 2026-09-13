"""
cd /home/wagner/Downloads/pcc175/fonte
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib numpy pandas


cd /home/wagner/Downloads/pcc175/fonte && source .venv/bin/activate && code .

"""

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# python3 -m pip install PyQt6

#from google.colab import drive
#drive.mount('/content/drive')

def gera_imagem(fig, filename_prefix):
    # Criar uma pasta para as imagens no Drive, se não existir
    output_dir = './imagens'
    os.makedirs(output_dir, exist_ok=True)
    #print(f"Diretório de saída para imagens: {output_dir}")

    # Salva a figura
    filepath = os.path.join(output_dir, f'{filename_prefix}.png')
    fig.savefig(filepath)
    plt.close(fig) # Fechar a figura para liberar memória
    print(f"Imagem salva em: {filepath}")

"""Data da entrega : 16/09

Complete o código (marcado com None) e quando requisitado, escreva textos diretamente nos notebooks. 
Onde tiver None, substitua pelo seu código. 
Execute todo notebook e salve tudo em um PDF nomeado como "NomeSobrenome-P1.pdf".
"""

def no_dominated(pontos):
   """"
   Determina os pontos não dominados de Pareto

   Input:
   pontos = conjunto de pontos [f1 f2]

   Output:
   nd_pontos = pontos não dominados [f1* f2*]

   """

   N = pontos.shape[0] # quantidade  de pontos
   nd_pontos = []

   while N!=0:
      # Há pontos para classificar.
      #None # encontre os pontos não dominados
      for i in range(N):
         dominante = False
         for j in range(N):
            if i != j:
               if pontos[j][0] <= pontos[i][0] and pontos[j][1] <= pontos[i][1]:
                  dominante = True
                  break
         if not dominante:
            nd_pontos.append(pontos[i])

      break

   return nd_pontos # retorne os pontos não dominados

def gera_grafico(pontos, nd_pontos, filename_prefix=None):
   blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                           markersize=6, label='Pontos da Instância')
   red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                           markersize=10, markeredgecolor='black', label='Pontos Não Dominados')

   fig = plt.figure(figsize=(8, 6))
   plt.scatter(pontos[:, 0], pontos[:, 1], color='blue', alpha=0.7)

   # Converter o nd_pontos
   nd_pontos = np.array(nd_pontos)
   if nd_pontos.size > 0 and nd_pontos.ndim == 2:
      plt.scatter(nd_pontos[:, 0], nd_pontos[:, 1], color='red', s=100, edgecolors='black', zorder=5)
   elif nd_pontos.size > 0 and nd_pontos.ndim == 1:
      # Caso retorne apenas 1 ponto isolado (1D)
      plt.scatter(nd_pontos[0], nd_pontos[1], color='red', s=100, edgecolors='black', zorder=5)

   plt.title('Pontos da Instância e Pontos Não Dominados')
   plt.xlabel('f1')
   plt.ylabel('f2')
   plt.grid(True)
   plt.legend(handles=[blue_dot, red_dot])

   if filename_prefix is not None:
      gera_imagem(fig, filename_prefix)
   else:
      print("Apresenta a imagem!")
      matplotlib.use('TkAgg')
      plt.show()


def gera_arquivo_csv(nome_arquivo):

   pontos_lista = [] 
   tam = 0
   count = 0
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
               calcula_pontos(pontos, nome_imagem)
               pontos_lista = []            
            continue

         valores = linha.split(',')
         #print(f"Valores na linha {i+1}: {valores}")

         # Verificar a quantidade de elementos a serem processados.
         if len(valores) == 1:
            valores = linha.split(' ')
            if valores[0].lower() == 'n':
               #print("Tamanho do conjunto de pontos (N) encontrado na linha:", valores[1])
               tam = int(valores[1])
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

            if len(pontos_lista) < tam:
               pontos_lista.append([num1, num2])
            
            # Se atingir o tamanho exato da instância, calcula e limpa
            if len(pontos_lista) == tam:               
               pontos = np.array(pontos_lista)
               calcula_pontos(pontos, nome_imagem)
               pontos_lista = []

def calcula_pontos(pontos, nome_imagem):
   nd_pontos = no_dominated(pontos)
   print(f"Pontos Não Dominados:\n", nd_pontos)
   gera_grafico(pontos, nd_pontos, nome_imagem)
         
def main():
   nome_imagem = None
   nome_arquivo = './instance/knapPI_16_20_1000.csv'

   if nome_arquivo is not None:
      gera_arquivo_csv(nome_arquivo)
   else:
      print(f"Nome da imagem definido como: {nome_imagem}")

      #Teste da apostila
      pontos = np.array([[8 ,5], [9, 2], [12, 1], [11, 2], [16, 2] ])#None # leia as instâncias de teste    
      #nome_imagem = "teste_apostila"

      # Teste Array 1: Menor número de pontos, com alguns pontos dominados e outros não
      # Gerar pontos em uma área menor para criar mais sobreposição
      #pontos = np.array([[1, 5], [2, 4], [3, 3],[4, 2], [5, 1], [2, 5], [4, 3] ])
      #nome_imagem = "teste_array_1"

      # Teste Array 2: Maior número de pontos aleatórios em um range diferente
      #np.random.seed(42) # Outra seed para reprodutibilidade
      #pontos = np.random.randint(1, 100, size=(100, 2))
      #nome_imagem = "teste_array_2"
      #print("\nPontos de Teste 2 (primeiras 5 linhas):\n", pontos[:5])
      #print("\nPontos de Teste 2 (completos):\n", pontos)

      # execute sua função no_dominated
      #nd_pontos = no_dominated(pontos)
      #print("Pontos Não Dominados:\n", nd_pontos)
      #gera_grafico(pontos, nd_pontos, 'teste_array_2')

      # execute sua função no_dominated
      calcula_pontos(pontos, nome_imagem)
      #nd_pontos = no_dominated(pontos)
      #print("Pontos Não Dominados:\n", nd_pontos)
      #gera_grafico(pontos, nd_pontos, nome_imagem)

if __name__ == "__main__":
   main()