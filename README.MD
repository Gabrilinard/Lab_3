# Laboratório 3: Deep Learning e Arquitetura Transformer do Zero

Este repositório apresenta uma **implementação didática e de baixo nível** de alguns dos principais componentes de um **Transformer**, arquitetura usada em modelos modernos de IA.

O objetivo deste laboratório **não é treinar um modelo completo**, mas sim **entender profundamente a mecânica matemática por trás da arquitetura Transformer**, implementando partes essenciais **do zero usando apenas Python e NumPy**.

Neste laboratório você verá:

- Como funciona a **Máscara Causal**
- Como funciona a **Cross-Attention**
- Como funciona o **Loop de geração auto-regressiva de tokens**

---

# Clonando o repositório

Primeiro clone o repositório para sua máquina:

```bash
git clone Lab_3
```

Depois entre na pasta do projeto:

```bash
cd Lab_3
```

---

# Instalando o Python

Este projeto utiliza **Python 3**.

Caso você ainda não tenha Python instalado:

### Windows / Linux / Mac

Baixe diretamente no site oficial:

- https://www.python.org/downloads/

Após instalar, verifique no terminal:

```bash
python --version
```

ou

```bash
python3 --version
```

Se aparecer algo como:

```
Python 3.x.x
```

então a instalação foi bem-sucedida.

---

# Instalando o NumPy

O único pacote externo utilizado neste laboratório é o **NumPy**, que é uma biblioteca essencial para computação numérica em Python.

Instale com:

```bash
pip install numpy
```

ou

```bash
pip3 install numpy
```

Depois teste no Python:

```python
import numpy as np
print(np.__version__)
```

Se aparecer um número de versão, está tudo funcionando.

---

# Objetivo do Laboratório

A arquitetura **Transformer**, apresentada no artigo **"Attention is All You Need"**, revolucionou o campo de **Deep Learning aplicado à linguagem natural**.

Modelos baseados nela são usados em:

- Tradução automática
- Chatbots
- Geração de texto
- Sistemas de recomendação
- Assistentes virtuais

Neste laboratório implementamos **três mecanismos fundamentais**:

Máscara Causal  
Cross-Attention  
Loop de geração auto-regressiva  

Tudo isso **sem frameworks de deep learning**, apenas com **NumPy**, para entender **o que realmente acontece por trás das bibliotecas modernas**.

---

## Arquitetura de Dados:

Para que as multiplicações de matrizes funcionem, seguimos as dimensões padrão da arquitetura original:

Componente	          Variável	       Shape (Batch, Seq, Model)
Saída do Encoder	  encoder_out	         (1, 10, 512)
Estado do Decoder	  decoder_state	         (1, 4, 512)
Matrizes de Peso	  WQ, WK, WV	         (512, 512)
Scores de Atenção	  pesos	                 (1, 4, 10)

---

# Detalhamento Técnico das Tarefas

---

# 1.Máscara Causal (Tarefa 1) — O Filtro do Tempo

## Problema

Modelos autoregressivos como os da família **GPT** geram texto **palavra por palavra**.

Por exemplo:

```
<START> Gabriel fez a atividade
```

Quando o modelo está prevendo a palavra **"atividade"**, ele **não pode olhar para palavras futuras**, pois isso seria **vazamento de informação**.

Se o modelo pudesse ver o futuro durante o treino, ele apenas **copiaria a frase inteira** em vez de aprender a **prever a próxima palavra**.

---

## Solução

Criamos uma **máscara triangular superior** que bloqueia tokens futuros.

Exemplo de máscara para sequência de tamanho 5:

```
0   -inf  -inf  -inf  -inf
0    0    -inf  -inf  -inf
0    0     0    -inf  -inf
0    0     0     0    -inf
0    0     0     0     0
```

---

## Por que usar `-np.inf`?

O mecanismo de atenção utiliza **Softmax**:

```
softmax(x) = e^x / soma(e^x)
```

Como:

```
e^-∞ = 0
```

então os tokens futuros recebem **probabilidade zero de atenção**.

Isso faz o modelo ficar **"cego para o futuro"**.

---

# 2.Cross-Attention — A Memória do Encoder

Em arquiteturas **Encoder-Decoder** (como tradução automática), o **Decoder precisa acessar as informações processadas pelo Encoder**.

Isso é feito através do mecanismo chamado **Cross-Attention**.

---

## Ideia central

Criamos três projeções lineares:

| Matriz | Significado |
|------|------|
| Q (Query) | O que o Decoder está procurando |
| K (Key) | Onde procurar essa informação |
| V (Value) | Qual informação retornar |

---

## Fórmula da atenção

```
Attention(Q,K,V) = softmax( QKᵀ / √d ) V
```

---

## Por que dividir por √d ?

O produto escalar entre vetores grandes pode gerar **valores muito altos**, o que faria o **Softmax saturar**.

Dividir por √d mantém os valores em uma escala estável.

---

# 3.Loop de Inferência (Tarefa 3) — O Motor de Geração

Esta parte simula o funcionamento de um **modelo generativo real**.

A geração funciona assim:

1. Começa com um token inicial:

```
<START>
```

2. O modelo prevê a próxima palavra

3. A palavra gerada é adicionada à sequência

4. O processo se repete até:

- gerar `<EOS>`
- ou atingir o limite máximo de tokens

---

## Estrutura do Loop

```
<START>
↓
gera palavra
↓
<START> palavra1
↓
gera palavra
↓
<START> palavra1 palavra2
```

---

## Scores

O modelo gera um vetor chamado **scores**:

```
[10000 valores]
```

Cada posição representa **a pontuação de uma palavra no vocabulário**.

---

## Escolha da palavra

Utilizamos:

```
argmax()
```

que seleciona o **índice da maior probabilidade**.

---

# Por que a frase gerada não faz sentido?

Exemplo de saída:

```
<START> atividade do Gabriel <PAD> fez
```

Isso **não é erro**.

Existem três motivos principais:

---

## 1.Pesos Aleatórios

Os pesos foram inicializados com:

```
np.random.randn()
```

Ou seja, o modelo **não foi treinado**.

Ele possui **estrutura de rede neural**, mas **nenhum conhecimento linguístico**.

---

## 2.Sem Backpropagation

Para gerar frases coerentes seria necessário:

- bilhões de frases de treino
- otimização via **Backpropagation**
- ajuste de milhões ou bilhões de parâmetros

---

## 3.Vocabulário Simulado

Simulamos um vocabulário de **10.000 palavras**, mas usamos apenas **9 palavras reais**.

Por isso usamos:

```
indice % tamanho_vocabulario
```

Isso aumenta a chance de **repetições e combinações aleatórias**.

---

# O que este código realmente prova

Mesmo sem treino, o código demonstra que:

a infraestrutura de geração funciona  
o loop autoregressivo está correto  
o critério de parada `<EOS>` funciona  
a lógica de seleção de tokens está correta  

Ou seja, **a arquitetura está pronta para receber um modelo treinado**.

---

# Código Completo do Laboratório

```python
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
```

---

# Conclusão

Este laboratório demonstra **na prática** como funcionam componentes centrais de modelos Transformer.

Mesmo sem treinamento real, ele mostra:

- como a atenção é calculada
- como o modelo evita olhar o futuro
- como ocorre a geração de tokens

---

Este laboratório serve como uma **ponte entre teoria matemática e implementação real de modelos de linguagem modernos**.

## Nota de Integridade e Créditos
Este trabalho seguiu rigorosamente as diretrizes de integridade do laboratório:

Uso de IA: Ferramentas de IA Generativa (Gemini/ChatGPT) foram consultadas exclusivamente para suporte em sintaxe da biblioteca numpy e para auxílio na estruturação deste documento explicativo. Elas auxiliaram na resolução de dúvidas de sintaxe da biblioteca numpy e no brainstorming para a estruturação lógica em pequenas partes do código, algumas dúvidas mesmo. Mas toda a implementação foi eu tentando fazer passo a passo, garantindo que cada parte do código fosse de acordo com o que foi pedido no laboratório. A documentação foi revisada e validada por mim, garantindo que o trabalho reflita meu entendimento sobre o funcionamento do mecanismo de atenção (Self e Cross-Attention), o papel da máscara causal na preservação da autoregressão e o processo de geração de tokens, conforme as regras do contrato pedagógico.