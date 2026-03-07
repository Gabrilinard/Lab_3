import numpy as np

def softmax(matriz):
    exponencial_matriz = np.exp(matriz - np.max(matriz, axis=-1, keepdims=True))
    return exponencial_matriz / exponencial_matriz.sum(axis=-1, keepdims=True)

def create_causal_mask(seq_len):
    triangulo_superior = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(triangulo_superior == 1, -np.inf, 0.0)
    
    return mask

seq_len = 5
Q = np.random.randn(seq_len, 64)
K = np.random.randn(seq_len, 64)

mask = create_causal_mask(seq_len)
print(mask)

pontuacoes_atencao = Q @ K.T
pontuacoes_mascaradas = pontuacoes_atencao + mask
probabilidades_atencao = softmax(pontuacoes_mascaradas)

print("Matriz de Probabilidades:")
print(np.round(probabilidades_atencao, 2))