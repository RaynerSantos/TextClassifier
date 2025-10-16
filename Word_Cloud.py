# Se precisar: pip install pandas matplotlib wordcloud nltk unidecode
import re
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from unidecode import unidecode
from typing import Optional, Set
from collections import Counter

# --- (opcional) stopwords em português via NLTK ---
# import nltk
# nltk.download('stopwords')
try:
    from nltk.corpus import stopwords
    PT_STOPWORDS = set(stopwords.words('portuguese'))
except:
    PT_STOPWORDS = set()

def frequencia_palavras(
    df: pd.DataFrame,
    coluna_texto: str,
    top_n: int = 40,
    remover_acentos: bool = True,
    usar_stopwords: bool = True,
    adicionais_stopwords: Optional[Set[str]] = None,
    min_len: int = 2,              # ignora tokens muito curtos
) -> pd.DataFrame:
    """
    Retorna um DataFrame com as 'top_n' palavras mais frequentes e suas contagens.
    """

    # 1) Seleção e limpeza básica
    textos = df[coluna_texto].dropna().astype(str).tolist()

    def limpar(t: str) -> str:
        t = t.lower()
        t = re.sub(r"http\S+|www\.\S+", " ", t)   # URLs
        t = re.sub(r"[@#]\w+", " ", t)            # menções/hashtags
        t = re.sub(r"\d+", " ", t)                # números
        t = re.sub(r"[^\w\s]", " ", t)            # pontuação
        t = re.sub(r"\s+", " ", t).strip()
        if remover_acentos:
            t = unidecode(t)
        return t

    textos_limpos = [limpar(t) for t in textos]

    # 2) Monta conjunto de stopwords (opcional)
    if usar_stopwords:
        base = set(STOPWORDS)
        pt = {unidecode(w) for w in PT_STOPWORDS} if remover_acentos else PT_STOPWORDS
        stops = base | pt
        if adicionais_stopwords:
            extras = {unidecode(w) for w in adicionais_stopwords} if remover_acentos else set(adicionais_stopwords)
            stops |= extras
    else:
        stops = set()

    # 3) Tokenização simples e filtragem
    tokens = []
    for t in textos_limpos:
        for tok in t.split():
            if len(tok) >= min_len and tok not in stops:
                tokens.append(tok)

    if not tokens:
        return pd.DataFrame(columns=["palavra", "frequencia"])

    # 4) Contagem e saída
    cont = Counter(tokens).most_common(top_n)
    freq_df = pd.DataFrame(cont, columns=["palavra", "frequencia"])

    # # print simples (opcional)
    # for palavra, f in cont:
    #     print(f"{palavra}: {f}")

    return freq_df

def gerar_nuvem_palavras(
    df: pd.DataFrame,
    coluna_texto: str,
    # caminho_saida: str = "nuvem_palavras.png",
    largura: int = 1600,
    altura: int = 900,
    fundo: str = "#5C4CE2",
    remover_acentos: bool = True,
    collocations: bool = False,
    usar_stopwords: bool = True,                    # << NOVO: liga/desliga stopwords
    adicionais_stopwords: Optional[Set[str]] = None # << continua opcional
):
    """
    Gera e salva uma nuvem de palavras a partir de df[coluna_texto].

    Parâmetros:
      - usar_stopwords: se False, nenhuma stopword é aplicada.
      - adicionais_stopwords: conjunto extra de termos para filtrar (se usar_stopwords=True).
    """
    # 1) Seleção e limpeza básica
    textos = df[coluna_texto].dropna().astype(str).tolist()

    def limpar(t: str) -> str:
        t = t.lower()
        t = re.sub(r"http\S+|www\.\S+", " ", t)  # URLs
        t = re.sub(r"[@#]\w+", " ", t)           # menções/hashtags
        t = re.sub(r"\d+", " ", t)               # números
        t = re.sub(r"[^\w\s]", " ", t)           # pontuação
        t = re.sub(r"\s+", " ", t).strip()
        if remover_acentos:
            t = unidecode(t)
        return t

    textos_limpos = [limpar(t) for t in textos]
    texto_unico = " ".join(textos_limpos)

    # 2) Stopwords (podem ser desligadas)
    if usar_stopwords:
        stopwords_base = set(STOPWORDS)
        stopwords_pt = {unidecode(w) for w in PT_STOPWORDS} if remover_acentos else PT_STOPWORDS
        stops = stopwords_base | stopwords_pt
        if adicionais_stopwords:
            extras = {unidecode(w) for w in adicionais_stopwords} if remover_acentos else set(adicionais_stopwords)
            stops |= extras
    else:
        stops = set()  # sem stopwords

    # 3) Geração da nuvem
    wc = WordCloud(
        width=largura,
        height=altura,
        background_color=fundo,
        stopwords=stops,
        collocations=collocations,
        max_words=20,
    ).generate(texto_unico)

    # 4) Exibir/salvar
    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.show()
    # print(f"Nuvem salva em: {caminho_saida}")


file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
sheet_name = "Completo" # TREINO  |  Completo_semNS
TAG = 'Ter mais rede credenciada perto de onde mora / o hospital que gosta'
df = pd.read_excel(file_path, sheet_name=sheet_name)
# df = df.loc[df['NPS'] == 'Detrator']
df = df.loc[(df['Classificacao_humana_1'] == TAG) & (df['Classificacao_humana_2'].isna() ) & (df['Classificacao_humana_3'].isna())]
print(f'\nShape de df: {df.shape}')

freq_df = frequencia_palavras(
    df,
    coluna_texto="resposta",
    top_n=40,
    remover_acentos=True,
    usar_stopwords=True,
    adicionais_stopwords={"cliente","pra","vezes","trabalho","unimed"}  # opcional
)
print(f'\nFrequência das palavras mais comuns em ({TAG}):')
print(freq_df.head(40))


# file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\TESTE.xlsx"
# df = pd.read_excel(file_path)
# df = df.loc[df['Classificacao_humana_1'] == TAG]
# df = df.loc[(df['acerto_1'] == 0) & (df['acerto_2'] == 0) & (df['acerto_3'] == 0)] # filtrar só os erros

# freq_df = frequencia_palavras(
#     df,
#     coluna_texto="resposta",
#     top_n=30,
#     remover_acentos=True,
#     usar_stopwords=True,
#     adicionais_stopwords={"cliente","pra","vezes","trabalho","unimed"}  # opcional
# )
# print('Banco com TAGs marcadas incorretamente')
# print(f'\nFrequência das palavras mais comuns em ({TAG}):')
# print(freq_df.head(30))


# gerar_nuvem_palavras(
#     df,
#     coluna_texto="resposta",
#     # caminho_saida="nuvem_com_stopwords.png",
#     collocations=False,
#     usar_stopwords=True,  # padrão
#     # adicionais_stopwords={"cliente", "bem", "tanto", "dia", "nada", "toda", "vez", "desde", "ponto", "certo", "brasil", "parte",
#     #                       "horario", "agenda", "pra", "vezes", "trabalho", "varios", "porque"}  # opcional
# )