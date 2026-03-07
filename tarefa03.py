import numpy as np

def normalizar_probabilidades(logits):
    exp_ajustado = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_ajustado / np.sum(exp_ajustado, axis=-1, keepdims=True)

def generate_next_token(current_sequence, encoder_out):
    TAMANHO_VOCABULARIO = 10000
    logits_ficticios = np.random.randn(TAMANHO_VOCABULARIO)
    return normalizar_probabilidades(logits_ficticios)

vocabulario_exemplo = ["<PAD>", "<START>", "<EOS>", "Gabriel", "fez", "a", "atividade", "do", "laboratório"]

contexto_da_frase = ["<START>"]
sentido_original = np.random.randn(1, 10, 512) 
max_tokens = 15

print("TAREFA 3: Simulando a fala do modelo")

while len(contexto_da_frase) < max_tokens:
    distribuicao = generate_next_token(contexto_da_frase, sentido_original)
    
    indice_escolhido = np.argmax(distribuicao)
    
    token_gerado = vocabulario_exemplo[indice_escolhido % len(vocabulario_exemplo)]
    
    contexto_da_frase.append(token_gerado)
    
    print(f"Token gerado: {token_gerado.ljust(12)} | Contexto atual: {' '.join(contexto_da_frase)}")
    
    if token_gerado == "<EOS>":
        print("\n[INFO] Token de parada <EOS> detectado.")
        break

print("-" * 50)
print(f"FRASE FINAL GERADA: {' '.join(contexto_da_frase)}")
print("-" * 50)