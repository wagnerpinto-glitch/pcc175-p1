"""
cd /home/wagner/Downloads/pcc175/fonte
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib numpy pandas fpdf2

cd /home/wagner/Downloads/pcc175/fonte && source .venv/bin/activate && code .

python -m pip install fpdf2

"""

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from fpdf import FPDF

# Configuração do nome do arquivo de saída solicitado na apostila
NOME_SOBRENOME_PDF = "WagnerPinto-P1.pdf" 

def gera_imagem(fig, filename_prefix):
    output_dir = './imagens'
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f'{filename_prefix}.png')
    fig.savefig(filepath)
    plt.close(fig) # Liberar memória do sistema
    print(f"Imagem salva em: {filepath}")
    return filepath

def no_dominated(pontos):
   """"
   Determina os pontos não dominados de Pareto para a Mochila.
   Objetivo: MAXIMIZAR Lucro (coluna 0) e MINIMIZAR Peso (coluna 1)
   """
   N = pontos.shape[0] # quantidade de pontos
   nd_pontos = []

   for i in range(N):
      dominante = False
      lucro_i = pontos[i][0]
      peso_i = pontos[i][1]

      for j in range(N):
         if i != j:
            lucro_j = pontos[j][0]
            peso_j = pontos[j][1]

            # CONCEITO EXATO DE PARETO PARA MOCHILA:
            # j domina i se tiver lucro maior/igual E peso menor/igual...
            if lucro_j >= lucro_i and peso_j <= peso_i:
               # Deve ser estritamente melhor em pelo menos um dos critérios
               if lucro_j > lucro_i or peso_j < peso_i:
                  dominante = True
                  break # i foi dominado por j, não é Pareto

      if not dominante:
         # Evita duplicar coordenadas perfeitamente idênticas na lista final
         if not any(np.array_equal(pontos[i], x) for x in nd_pontos):
            nd_pontos.append(pontos[i])

   return nd_pontos # Retorna apenas os reais pontos da Fronteira de Pareto


def gera_grafico(pontos, nd_pontos, itens_otimos, filename_prefix=None, capacidade=0, lucro_total=0):
   # Legendas explícitas e organizadas
   blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                           markersize=6, label='Itens Disponíveis')
   green_star = mlines.Line2D([], [], color='green', marker='*', linestyle='None',
                           markersize=10, label='Itens Escolhidos na Solução Ótima (x=1)')
   red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                           markersize=8, markeredgecolor='black', label='Fronteira de Pareto de Itens')

   fig, ax = plt.subplots(figsize=(8, 5.5))
   
   # 1. Desenha a base: Todos os 20 itens da instância em Azul
   ax.scatter(pontos[:, 0], pontos[:, 1], color='blue', s=50, alpha=0.8, label='Disponíveis', zorder=2)

   # 2. Desenha a Fronteira de Pareto em Vermelho (Pontos Reais Não Dominados)
   nd_pontos = np.array(nd_pontos)
   if nd_pontos.size > 0:
      if nd_pontos.ndim == 2:
         ax.scatter(nd_pontos[:, 0], nd_pontos[:, 1], color='red', s=90, edgecolors='black', facecolors='none', linewidths=1.5, zorder=3)
      elif nd_pontos.ndim == 1:
         ax.scatter(nd_pontos[0], nd_pontos[1], color='red', s=90, edgecolors='black', facecolors='none', linewidths=1.5, zorder=3)

   # 3. Desenha as Estrelas Verdes POR CIMA de tudo para garantir a visibilidade do vetor x=1
   if len(itens_otimos) > 0:
      itens_otimos = np.array(itens_otimos)
      ax.scatter(itens_otimos[:, 0], itens_otimos[:, 1], color='green', marker='*', s=150, edgecolors='darkgreen', linewidths=0.5, zorder=4)

   # Metadados e Estética do Gráfico
   plt.title(f'Instância Mochila: {filename_prefix}\nCapacidade (c): {capacidade} | Lucro Ótimo Esperado (z): {lucro_total}', fontsize=10)
   plt.xlabel('Lucro do Item (p)')
   plt.ylabel('Peso do Item (w)')
   plt.grid(True, linestyle='--', alpha=0.5)
   plt.legend(handles=[blue_dot, red_dot, green_star], loc='upper left', fontsize=9)

   if filename_prefix is not None:
      caminho_imagem = gera_imagem(fig, filename_prefix)
      return caminho_imagem
   else:
      matplotlib.use('TkAgg')
      plt.show()
      return None


def gera_arquivo_csv(nome_arquivo):
   pontos_lista = [] 
   itens_otimos = []
   dados_instancias = [] # Guardará caminhos de imagem e metadados para construir o PDF
   tam = 0
   capacidade = 0
   lucro_total = 0
   nome_imagem = "instancia"
   
   with open(nome_arquivo, 'r') as f:
      for i, line in enumerate(f):
         linha = line.strip()

         if not linha or 'knapPI' in linha:
            nome_imagem = linha.replace('.csv', '').strip()
            pontos_lista = []
            itens_otimos = []
            continue

         if linha.startswith('-'):
            if len(pontos_lista) > 0:
               #pontos_lista.append([10, 900]) 
               pontos = np.array(pontos_lista)
               img_path = calcula_pontos(pontos, itens_otimos, nome_imagem, capacidade, lucro_total)
               if img_path:
                  dados_instancias.append({
                      'nome': nome_imagem, 'c': capacidade, 'z': lucro_total, 'n': tam, 'img': img_path
                  })
               pontos_lista = []            
               itens_otimos = []
            continue

         valores = linha.split(',')

         if len(valores) == 1:
            partes = linha.split(' ')
            if len(partes) >= 2:
               chave = partes[0].lower()
               valor = partes[1]
               if chave == 'n':
                  tam = int(valor)
               elif chave == 'c':
                  capacidade = int(valor)
               elif chave == 'z':
                  lucro_total = int(valor)
            continue

         if len(valores) == 4:
            try:
               num1 = int(valores[1]) # Lucro p[i]
               num2 = int(valores[2]) # Peso w[i]
               x_opt = int(valores[3]) # Selecionado x[i]
               
               if len(pontos_lista) < tam:
                  pontos_lista.append([num1, num2])
                  if x_opt == 1:
                     itens_otimos.append([num1, num2])
               
               if len(pontos_lista) == tam:    
                  #pontos_lista.append([10, 900])            
                  pontos = np.array(pontos_lista)
                  img_path = calcula_pontos(pontos, itens_otimos, nome_imagem, capacidade, lucro_total)
                  if img_path:
                     dados_instancias.append({
                         'nome': nome_imagem, 'c': capacidade, 'z': lucro_total, 'n': tam, 'img': img_path
                     })
                  pontos_lista = []
                  itens_otimos = []
                  
            except ValueError:
               continue
               
   # Após varrer todo o arquivo CSV, cria o PDF consolidado
   if dados_instancias:
       exportar_para_pdf(dados_instancias)

def calcula_pontos(pontos, itens_otimos, nome_imagem, cap, z):
   nd_pontos = no_dominated(pontos)
   print(f"Processado: {nome_imagem} | Pareto: {len(nd_pontos)} itens")
   img_path = gera_grafico(pontos, nd_pontos, itens_otimos, nome_imagem, capacidade=cap, lucro_total=z)
   return img_path

def exportar_para_pdf(dados_instancias):
    print(f"\nIniciando compilação do relatório PDF: {NOME_SOBRENOME_PDF}...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- CAPA DO RELATÓRIO ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 40, "Relatório de Otimização Multi-Objetivo", ln=True, align="C")
    pdf.cell(0, 10, "Problema da Mochila Clássico - Análise de Pareto", ln=True, align="C")
    
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Estudante / Autor: Wagner Tironi Pinto", ln=True, align="L")
    pdf.cell(0, 10, f"Disciplina: PCC175 - Técnicas de Otimização Multiobjetivo", ln=True, align="L")
    pdf.cell(0, 10, f"Professor: Gladston Juliano Prates Moreira", ln=True, align="L")    
    pdf.cell(0, 10, f"Total de Instâncias Avaliadas: {len(dados_instancias)} conjuntos", ln=True, align="L")
    #pdf.cell(0, 10, f"Data do Processamento: 16/09", ln=True, align="L")
    
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    txt_desc = ("Descrição da Metodologia: Este documento apresenta os resultados gráficos obtidos através "
                "da triagem de Pareto aplicada sobre conjuntos de dados da Mochila. O algoritmo avalia de "
                "forma multi-critério a maximização de lucro frente à minimização de peso de cada item "
                "disponível na instância, demarcando a fronteira de não-dominância em vermelho.")
    pdf.multi_cell(0, 6, txt_desc)
    
    # --- PÁGINAS DE CONTEÚDO ---
    # Coloca 2 instâncias por página para otimizar espaço de forma organizada
    for idx, inst in enumerate(dados_instancias):
        if idx % 2 == 0:
            pdf.add_page()
            
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Instância: {inst['nome']}", ln=True)
        
        pdf.set_font("Helvetica", "", 9)
        meta_txt = f"Itens Disponíveis (n): {inst['n']}  |  Capacidade Total (c): {inst['c']}  |  Lucro Alcançado na Solução (z): {inst['z']}"
        pdf.cell(0, 5, meta_txt, ln=True)
        
        # Insere a imagem do gráfico correspondente
        if os.path.exists(inst['img']):
            # Calcula o posicionamento vertical com base no índice par/ímpar da página
            pos_y = 30 if (idx % 2 == 0) else 155
            pdf.image(inst['img'], x=15, y=pos_y, w=180)
            
        pdf.ln(115) # Espaçamento para a próxima instância ou rodapé
        
    pdf.output(NOME_SOBRENOME_PDF)
    print(f"Relatório PDF gerado: {os.path.abspath(NOME_SOBRENOME_PDF)}")

def main():
   # Processa o arquivo
   nome_arquivo = './instance/knapPI_16_100_1000.csv'

   if os.path.exists(nome_arquivo):
      gera_arquivo_csv(nome_arquivo)
   else:
      print(f"Arquivo CSV de entrada não localizado em: {nome_arquivo}")

if __name__ == "__main__":
   main()
