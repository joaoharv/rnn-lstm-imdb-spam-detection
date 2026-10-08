"""Interface Gradio do Checkpoint 2 (anomalias sintéticas em avaliações de filmes).

Usa exatamente o mesmo modelo (artefatos/lstm_melhor_val_loss.keras), a mesma padronização de texto
(artefatos/padronizar.py), o mesmo vocabulário e a mesma lógica de LIME das seções 7 e 8 do notebook.
Não treina nada.

    python app/app_gradio.py              # abre a interface em http://127.0.0.1:7860
    python app/app_gradio.py --verificar  # carrega o modelo, classifica e explica um texto e sai

Os artefatos são lidos de `artefatos/` (na raiz do repositório) ou da pasta indicada em ARTEFATOS_DIR.
"""
import html
import json
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.environ.get("ARTEFATOS_DIR") or os.path.join(RAIZ, "artefatos")        # padrão: <raiz do repositório>/artefatos
SEMENTE = 42

modelo_final = keras.saving.load_model(os.path.join(PASTA, "lstm_melhor_val_loss.keras"))
_voc = json.load(open(os.path.join(PASTA, "vocabulario.json"), encoding="utf-8"))
vocabulario_final, COMPRIMENTO = _voc["vocabulario"], _voc["comprimento"]
exec(open(os.path.join(PASTA, "padronizar.py"), encoding="utf-8").read(), globals())   # define `padronizar` (seção 4.2 do notebook)
camada_final = keras.layers.TextVectorization(max_tokens=len(vocabulario_final), standardize=padronizar, split="whitespace",
                                              output_mode="int", output_sequence_length=COMPRIMENTO)
camada_final.set_vocabulary(vocabulario_final[2:])


def vetorizar_final(textos, lote=1024):
    textos = [textos] if isinstance(textos, str) else list(textos)
    return np.concatenate([camada_final(tf.constant(textos[i:i + lote])).numpy() for i in range(0, len(textos), lote)]).astype("int32")


def predict_proba(textos):
    """Lista de textos brutos -> matriz (n, 2) com [P(normal), P(anomalia sintética)], o formato esperado pelo LIME."""
    textos = [textos] if isinstance(textos, str) else list(textos)
    p1 = modelo_final.predict(vetorizar_final(textos), batch_size=512, verbose=0).ravel().astype("float64")
    return np.column_stack([1 - p1, p1])


def explicar(texto, num_samples=1000, num_features=12):
    """Explicação LIME da classe 1, com semente fixa (o mesmo texto sempre gera a mesma explicação)."""
    explicador = LimeTextExplainer(class_names=["normal", "anomalia"], random_state=SEMENTE)
    return explicador.explain_instance(texto, predict_proba, num_features=num_features, num_samples=num_samples, labels=(1,))


def destacar_html(texto, pesos):
    """Texto com as palavras coloridas: vermelho favorece 'anomalia'; azul favorece 'normal'."""
    maximo = max((abs(w) for w in pesos.values()), default=0) or 1

    def pintar(m):
        palavra, peso = m.group(0), pesos.get(m.group(0))
        if peso is None:
            return html.escape(palavra)
        intensidade = 0.20 + 0.65 * abs(peso) / maximo
        cor = f"rgba(214,39,40,{intensidade:.2f})" if peso > 0 else f"rgba(31,119,180,{intensidade:.2f})"
        return f'<span style="background:{cor};border-radius:3px;padding:0 2px" title="peso LIME {peso:+.3f}">{html.escape(palavra)}</span>'
    corpo = re.sub(r"\w+|\W+", pintar, texto)
    legenda = '<div style="font-size:12px;margin-bottom:6px"><span style="background:rgba(214,39,40,.6);padding:0 6px">favorece anomalia</span> <span style="background:rgba(31,119,180,.6);padding:0 6px">favorece normal</span> (apenas as palavras mais influentes são coloridas)</div>'
    return f'{legenda}<div style="white-space:pre-wrap;line-height:1.7;font-family:sans-serif">{corpo}</div>'


AVISO = f"""**Limitações.** (1) Apenas os **primeiros {COMPRIMENTO} tokens** do texto são analisados; o restante é ignorado. (2) O modelo foi treinado com anomalias **sintéticas** (spam promocional, repetição excessiva e manipulação de avaliações) e **não detecta fraude real**. (3) Ele reconhece bem os modelos vistos no treino, mas tem **baixa generalização** para mecanismos inéditos (recall de 0,20 na avaliação final) e falha em repetições feitas com palavras do próprio texto. (4) A explicação do LIME é **local e aproximada**: indica o efeito de cada palavra neste texto, não uma regra do modelo. (5) Use avaliações em inglês."""


def contar_tokens(texto):
    return len(padronizar(tf.constant([texto])).numpy()[0].decode("utf-8").split())


def analisar(texto):
    """Devolve (classificação, probabilidade da classe 1, explicação em HTML, tabela de palavras, aviso desta análise)."""
    texto = (texto or "").strip()
    if not texto:
        return "Insira uma avaliação para analisar.", None, "", [], ""
    n_tokens = contar_tokens(texto)
    p1 = float(predict_proba([texto])[0, 1])
    rotulo = "Anomalia sintética" if p1 >= 0.5 else "Normal"
    aviso = f"O texto tem {n_tokens} tokens: apenas os primeiros {COMPRIMENTO} foram analisados pelo modelo." if n_tokens > COMPRIMENTO else ""
    try:
        pares = explicar(texto, num_samples=800).as_list(label=1)
        visual = destacar_html(texto, dict(pares))
        tabela = [[palavra, round(peso, 4), "anomalia" if peso > 0 else "normal"] for palavra, peso in sorted(pares, key=lambda x: -abs(x[1]))]
    except Exception:                                                  # por exemplo, texto sem palavras para explicar
        visual, tabela, aviso = "", [], (aviso + " Explicação LIME indisponível para este texto.").strip()
    return rotulo, p1, visual, tabela, aviso


EXEMPLOS = [["Get premium membership for only $4.99 today, coupon cine71. Visit www.filmhub24.com for free HD streaming of every new release!"],
            ["I enjoyed the cinematography and the lead performance, although the second half felt a little slow."],
            ["This movie is great great great great great great great great great great great great, one of the best I have seen."]]


def construir_interface():
    import gradio as gr
    with gr.Blocks(title="Detector de anomalias sintéticas em avaliações") as demo:
        gr.Markdown("# Anomalias sintéticas em avaliações de filmes\nCole uma avaliação (em inglês) e clique em **Analisar**.")
        gr.Markdown(AVISO)
        entrada = gr.Textbox(label="Avaliação", lines=8, placeholder="Cole aqui o texto da avaliação...")
        botao = gr.Button("Analisar", variant="primary")
        with gr.Row():
            classificacao = gr.Textbox(label="Classificação")
            probabilidade = gr.Number(label="Probabilidade da classe 1 (anomalia sintética)")
        aviso_analise = gr.Markdown()
        explicacao = gr.HTML(label="Explicação visual (LIME)")
        tabela_palavras = gr.Dataframe(headers=["palavra", "peso LIME", "favorece"], label="Palavras mais influentes (LIME)", interactive=False)
        botao.click(analisar, inputs=entrada, outputs=[classificacao, probabilidade, explicacao, tabela_palavras, aviso_analise], api_name="analisar")
        gr.Examples(examples=EXEMPLOS, inputs=entrada, label="Exemplos")
    return demo


def verificar():
    """Teste rápido em processo independente: modelo carregado, texto classificado e explicado, sem treinar."""
    dados = np.load(os.path.join(PASTA, "sequencias.npz"))
    cfg = json.load(open(os.path.join(PASTA, "config_modelo.json"), encoding="utf-8"))
    p = modelo_final.predict(dados["X_validacao"], batch_size=512, verbose=0).ravel()
    acuracia = float(((p >= 0.5).astype(int) == dados["y_validacao"]).mean())
    print(f"modelo: {modelo_final.count_params():,} parâmetros | entrada {modelo_final.input_shape}")
    print(f"acurácia na validação: {acuracia:.4f} (registrada na Etapa 05: {cfg['val_accuracy_melhor_epoca']:.4f})")
    assert abs(acuracia - cfg["val_accuracy_melhor_epoca"]) < 1e-3, "o modelo carregado não reproduz a métrica registrada"
    for nome, texto in (("suspeito", EXEMPLOS[0][0]), ("normal", EXEMPLOS[1][0]), ("vazio", "")):
        rotulo, prob, visual, tabela, aviso = analisar(texto)
        print(f"{nome}: {rotulo} | prob. classe 1 = {prob if prob is None else round(prob, 4)} | palavras explicadas = {len(tabela)}")
    rotulo, prob, visual, tabela, _ = analisar(EXEMPLOS[0][0])
    assert prob is not None and 0 <= prob <= 1 and visual and tabela, "classificação ou explicação ausente"
    print("palavras mais influentes (suspeito):", [(w, p_) for w, p_, _ in tabela[:5]])
    print("VERIFICAÇÃO OK")


if __name__ == "__main__":
    if "--verificar" in sys.argv:
        verificar()
    else:
        construir_interface().launch(share=False)       # local; a aplicação não é publicada
