import numpy as np

def softmax(matriz):
    exponencial_ajustada = np.exp(matriz - np.max(matriz, axis=-1, keepdims=True))
    return exponencial_ajustada / exponencial_ajustada.sum(axis=-1, keepdims=True)

def create_causal_mask(sq_len):
    triangulo_superior = np.triu(np.ones((sq_len, sq_len)), k=1)
    mascara = np.where(triangulo_superior == 1, -np.inf, 0.0)
    return mascara

print("TAREFA 1: Máscara Causal")
seq_len = 5
Q_t1 = np.random.randn(seq_len, 64)
K_t1 = np.random.randn(seq_len, 64)

mascara = create_causal_mask(seq_len)
pontuacoes_atencao = (Q_t1 @ K_t1.T) + mascara
probabilidades_atencao = softmax(pontuacoes_atencao)

print("Matriz de Probabilidades (Com Máscara):")
print(np.round(probabilidades_atencao, 2))
print("\n")


def calcular_atencao_cruzada(encoder_output, decoder_state):
    d_model = encoder_output.shape[-1]
    
    WQ, WK, WV = [np.random.randn(d_model, d_model) for _ in range(3)]
    
    Q = decoder_state @ WQ
    K = encoder_output @ WK
    V = encoder_output @ WV

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_model)
    pesos = softmax(scores)
    return pesos @ V

print("TAREFA 2: Cross-Attention")
encoder_out_t2 = np.random.randn(1, 10, 512)
decoder_state_t2 = np.random.randn(1, 4, 512)
saida_cross = calcular_atencao_cruzada(encoder_out_t2, decoder_state_t2)
print(f"Dimensão da saída: {saida_cross.shape} (Sucesso)")

def generate_next_token(current_sequence, encoder_out):
    V_SIZE = 10000
    scores = np.random.randn(V_SIZE)
    return softmax(scores)

vocabulario_map = ["<PAD>", "<START>", "<EOS>", "Gabriel", "fez", "a", "atividade", "do", "laboratório"]
sequencia_gerada = ["<START>"]
memoria_do_encoder = np.random.randn(1, 10, 512)
max_tokens = 15

print("TAREFA 3: Simulando a fala do modelo")

while len(sequencia_gerada) < max_tokens:
    distribuicao_probs = generate_next_token(sequencia_gerada, memoria_do_encoder)
    
    indice_vencedor = np.argmax(distribuicao_probs)
    
    token_gerado = vocabulario_map[indice_vencedor % len(vocabulario_map)]
    
    sequencia_gerada.append(token_gerado)
    
    print(f"Passo {len(sequencia_gerada)-1}: Gerou '{token_gerado.ljust(12)}' | Sequência: {' '.join(sequencia_gerada)}")
    
    if token_gerado == "<EOS>":
        print("Token <EOS> detectado. Parando geração.")
        break

print("FRASE FINAL GERADA:")
print(" ".join(sequencia_gerada))