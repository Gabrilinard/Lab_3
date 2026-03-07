import numpy as np

def aplicar_softmax(logits):
    exponencial = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exponencial / np.sum(exponencial, axis=-1, keepdims=True)

def calcular_atencao_cruzada(encoder_output, decoder_state):
    d_k = encoder_output.shape[-1]
    
    WQ = np.random.normal(size=(d_k, d_k))
    WK   = np.random.normal(size=(d_k, d_k))
    WV = np.random.normal(size=(d_k, d_k))

    Q = decoder_state @ WQ
    K = encoder_output @ WK
    V = encoder_output @ WV

    score_de_alinhamento = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

    pesos = aplicar_softmax(score_de_alinhamento)
    
    vetor_de_sentido = pesos @ V
    
    return vetor_de_sentido

conhecimento_do_frances = np.random.standard_normal((1, 10, 512))
traducao_em_andamento = np.random.standard_normal((1, 4, 512))

resultado = calcular_atencao_cruzada(conhecimento_do_frances, traducao_em_andamento)

print(f"Dimensões da representação final: {resultado.shape}")
print("Ponte entre linguagens construída com sucesso.")