import torch
import numpy as np

def no_dominated_gpu(pontos):
    # Seleciona a GPU caso esteja disponível (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = len(pontos)
    total_combinacoes = 1 << N  # Total de 2^N combinações
    
    # Envia os pontos originais para a memória de vídeo (VRAM)
    pontos_tensor = torch.tensor(pontos, dtype=torch.float32, device=device)
    
    # Cria a matriz de combinações binárias (2^N, N) diretamente na GPU
    indices = torch.arange(total_combinacoes, device=device).unsqueeze(1)
    bits = torch.arange(N - 1, -1, -1, device=device)
    mascara = ((indices >> bits) & 1).float()
    
    # Multiplicação matricial paralela: (2^N, N) @ (N, 2) -> (2^N, 2)
    resultado_tensor = torch.matmul(mascara, pontos_tensor)
    
    # Transfere o resultado de volta para a CPU no formato NumPy
    return resultado_tensor.cpu().numpy()

# --- Exemplo prático ---
if __name__ == "__main__":
    # Exemplo com 20 pontos fictícios com coordenadas (X, Y)
    pontos_teste = np.random.rand(20, 2)
    
    # Processa todas as 1.048.576 combinações na GPU
    resultado = no_dominated_gpu(pontos_teste)
    
    print(f"Formato da matriz resultante: {resultado.shape}")