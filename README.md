# rnn-lstm-imdb-spam-detection

**Detecção e explicação de anomalias sintéticas em avaliações de filmes do IMDB com uma LSTM, LIME e Gradio.**

> *English summary:* an LSTM classifier trained on IMDB reviews with **synthetic** anomalies (promotional spam, excessive
> repetition and review manipulation), explained word by word with LIME and served through a Gradio app. The project
> follows the six CRISP-DM phases and openly reports that the model **does not generalize** to unseen anomaly mechanisms.
> Documentation and notebook are in Portuguese.

Projeto desenvolvido para o Checkpoint 2 de Redes Neurais Recorrentes (FIAP). A base do IMDB não tem rótulos de spam;
por isso, as anomalias são **criadas pelo projeto** e o classificador é treinado de forma supervisionada para reconhecê-las.

## Tecnologias

Python 3.12 · TensorFlow/Keras (Embedding, LSTM, `TextVectorization`) · scikit-learn · pandas · NumPy · Matplotlib ·
LIME · Gradio

## Como funciona

1. **Dados:** 50.000 avaliações do IMDB, deduplicadas (424 cópias removidas) e divididas em treino, validação, teste e
   generalização sem vazamento entre partições (agrupamento de quase-duplicatas).
2. **Anomalias sintéticas** (50% de cada partição): spam promocional, repetição excessiva e manipulação de avaliações,
   inseridas em posições variadas. A partição de generalização usa **mecanismos inéditos**, ausentes do treino.
3. **Pré-processamento:** padronização própria (mantém negações, contrações e sinais como `! % $ @ /`), vocabulário de
   20.000 tokens ajustado só no treino e sequências de 256 posições.
4. **Modelo:** Embedding (64, com máscara) → LSTM (64) → Dropout (0,5) → Dense sigmoid; perda *binary crossentropy*,
   *early stopping* na perda de validação (1,3 milhão de parâmetros).
5. **Explicabilidade:** LIME sobre o texto bruto, com destaque colorido das palavras (vermelho favorece anomalia,
   azul favorece normal).
6. **Interface:** aplicativo Gradio com classificação, probabilidade da classe 1, explicação visual e aviso de limitações.

## Resultados

| Partição | Acurácia | Precision | Recall | F1 | ROC AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Validação | 0,87 | 0,94 | 0,79 | 0,86 | 0,87 |
| **Teste** (mesmos geradores do treino) | **0,88** | 0,95 | 0,80 | 0,87 | 0,88 |
| **Generalização** (mecanismos inéditos) | **0,57** | 0,78 | **0,20** | 0,32 | 0,58 |

Recall por tipo de anomalia: no teste, spam 0,99, manipulação 0,97 e repetição excessiva 0,44; na generalização, 0,13, 0,39 e 0,09.
Uma regressão logística sobre TF-IDF (baseline) chega a 0,80 de acurácia na validação.

## Limitações (leia antes de usar)

- **Anomalias sintéticas, não fraude real.** O desempenho mede o reconhecimento de padrões criados pelo projeto.
- **Baixa generalização.** O modelo memorizou o vocabulário dos modelos de mensagem do treino: o recall cai de 0,80 (teste)
  para 0,20 em mecanismos inéditos. A repetição só é bem detectada quando a unidade repetida vem do repertório do gerador
  (recall de 0,74 contra 0,14 quando vem do próprio texto).
- **Prevalência artificial:** 50% de anomalias; em dados reais a precision seria muito menor.
- Apenas os **primeiros 256 tokens** de cada texto são analisados.
- A explicação do LIME é **local e aproximada** (R² do ajuste local entre 0,34 e 0,65 nos exemplos do notebook).
- Pensado para avaliações em inglês.

## Estrutura do repositório

```
imdb-review-anomaly-detection/
├── Check2_<matrículas>_<nomes>.ipynb   # notebook completo: CRISP-DM, resultados e registro de uso de IA
├── Check2_<matrículas>_<nomes>.pdf     # o mesmo notebook exportado em PDF (com as saídas)
├── app/
│   └── app_gradio.py                   # interface Gradio + LIME (inferência, sem treino)
├── artefatos/                          # tudo que a inferência precisa
│   ├── lstm_melhor_val_loss.keras      # modelo treinado
│   ├── vocabulario.json                # vocabulário de 20.000 tokens
│   ├── padronizar.py                   # função de padronização de texto
│   ├── sequencias.npz                  # sequências e rótulos (índices de tokens; sem texto)
│   └── config_modelo.json, historico_treino.csv, metricas_validacao.json
├── requirements.txt
└── README.md
```

O notebook fica na raiz porque localiza `artefatos/` por caminho relativo.

## Como executar

```bash
git clone https://github.com/<usuario>/imdb-review-anomaly-detection.git
cd imdb-review-anomaly-detection
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Demonstração (Gradio)** — não treina nada:

```bash
python app/app_gradio.py
```

Abra `http://127.0.0.1:7860`, cole uma avaliação em inglês (ou use um exemplo) e clique em **Analisar**. A aplicação roda
apenas localmente.

**Teste rápido sem abrir a interface** (carrega o modelo, confere a métrica registrada, classifica e explica um texto):

```bash
python app/app_gradio.py --verificar
```

**Notebook:** abra o `.ipynb` na raiz, com a pasta `artefatos/` ao lado. As saídas já estão salvas. Uma execução do zero
baixa o IMDB (84 MB) e **retreina** a LSTM (cerca de 10 minutos em CPU).

## Dados e créditos

O corpus é o *Large Movie Review Dataset v1.0* (Maas et al., 2011), usado na pesquisa e **não redistribuído** aqui.
`artefatos/sequencias.npz` contém apenas índices de tokens e identificadores de arquivo, sem o texto das avaliações.
Os trechos de anomalia são sintéticos; domínios, telefones e marcas citados são fictícios.

## Reprodutibilidade

Semente 42; o treinamento foi verificado como determinístico em CPU. O notebook documenta cada decisão (deduplicação,
partições, regras de escolha de vocabulário e comprimento) e o uso de ferramentas de IA em cada etapa.
