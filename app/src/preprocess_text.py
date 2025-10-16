# src/text_preprocessing.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')
    print("Downloading NLTK omw-1.4...")
    nltk.download('omw-1.4')


# Inicializar lematizador e stopwords globalmente neste módulo
# para que não sejam inicializados toda vez que a função preprocess_text for chamada.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))
# stop_words.discard("não")
# print('Stop Words:\n', stop_words)

# Função para extrair trechos de entrevistado
def extrair_entrevistado(text):
    # Remove espaços extras e confere se o texto está vazio
    if not text or not text.strip():
        return ""
    
    # Verifica se há a palavra "Entrevistado:"
    if 'Entrevistado:' in text:
        # Extrai apenas trechos do entrevistado
        trechos = re.findall(r'Entrevistado:.*?(?=Entrevistador:|$)', text, flags=re.DOTALL)
        return ' '.join(trechos).strip()
    else:
        # Se não houver marcadores, considera todo o texto como se fosse do entrevistado
        return text.strip()


def preprocess_text(text, modelo):
    """
    Função para pré-processar um texto:
    - Converte para minúsculas
    - Remove caracteres não-alfabéticos (exceto espaços)
    - Tokeniza
    - Remove stop words
    - Lematiza
    """
    
    text = text.lower() # Converter para minúsculas
    # Remove caracteres que não sejam letras (a-z) ou espaços (\s)
    # text = re.sub(r'[^a-z\s]', '', text)

    # Limpeza do texto
    # text = re.sub(r'\b(entrevistado|entrevistada|entrevistador|entrevistadora|)\b', '', text, flags=re.IGNORECASE)
    # text = re.sub(r'\b(entrevistado|entrevistador)\b|[:.,?!;]', '', text, flags=re.IGNORECASE)
    # text = re.sub(r'\b(que|para|com|após|das|até|de|ao|por|qual|quando|uma|a|o|então|foi|cielo|já|dela|ela|são)\b', '', 
    #               text, flags=re.IGNORECASE)
    tokens = word_tokenize(text, language="portuguese") # Tokenização

    # Remover stop words e lematizar
    if modelo == "Detrator":
        processed_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 
    return ' '.join(processed_tokens)
    # return ' '.join(tokens)


if __name__ == '__main__':
    print("\nTestando a função preprocess_text:")
    sample_comment = "Entrevistado: É ótima. Transaciona todos os cartões 100%. Temos menos problema com ela. Entrevistador: Quando você fala assim, temos menos problemas, é em termos de que, por exemplo? Entrevistado: De as vezes dá algum erro em questão da maquininha, por não transacionar o cartão."
    processed_comment = preprocess_text(sample_comment)
    print(f"\nOriginal: {sample_comment}")
    print(f"\nProcessado: {processed_comment}\n")


# ============================================================================================================================ # 


# Pré-processamento para as TAGs
class Pre_Processamento:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name

    # === Pré-processamento dos textos === #
    def pre_processamento(self):
        """Carrega e pré-processa os dados brutos"""
        df = pd.read_excel(self.file_path, self.sheet_name)
        
        # Consolidar tags
        df["tags_resposta"] = df[[
            "Classificacao_humana_1", "Classificacao_humana_2", "Classificacao_humana_3"
        ]].values.tolist()

        # Limpar tags
        df["tags_resposta"] = df["tags_resposta"].apply(
            # lambda x: list({str(i).strip().lower() for i in x if pd.notna(i)})
            lambda x: list({str(i).strip() for i in x if pd.notna(i)})
        )

        # Limpar texto
        df["resposta"] = df["resposta"].apply(
            # lambda x: str(x).lower().strip() if pd.notna(x) else ""
            lambda x: str(x).strip() if pd.notna(x) else ""
        )

        return df

    # === Pré-processamento para ver a frequência das TAGs === # 
    def pre_processamento_tags(self):
        """Carrega e pré-processa os dados brutos"""
        df = pd.read_excel(self.file_path, self.sheet_name)

        # Consolidar tags
        df["tags_resposta"] = df[[
            "Classificacao_humana_1", "Classificacao_humana_2", "Classificacao_humana_3"
        ]].values.tolist()

        # Limpar tags
        df["tags_resposta"] = df["tags_resposta"].apply(
            # lambda x: list({str(i).strip().lower() for i in x if pd.notna(i)})
            lambda x: list({str(i).strip() for i in x if pd.notna(i)})
        )

        tags_df = (
        df.melt(
            value_vars=[col for col in df.columns if col.startswith("Classificacao_humana")],
            var_name="Class_humana",
            value_name="TAGs"
        )
        .dropna(subset=["TAGs"])
        .groupby("TAGs")
        .size()
        .reset_index(name="n")
    )
        # Calcular percentual e ordenar
        tags_df["percentual"] = round(tags_df["n"] / tags_df["n"].sum(), 3)
        tags_df = tags_df.sort_values(by="n", ascending=False)

        return tags_df
    
# 2. Preparação das tags
def prepare_multilabel_data(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as tags para formato multilabel"""
    todas_tags = sorted(set(tag for sublist in df["tags_resposta"] for tag in sublist))
    mlb = MultiLabelBinarizer(classes=todas_tags)
    tags_binarias = mlb.fit_transform(df["tags_resposta"])
    
    # Preserve o mesmo índice do df original
    df = df.reset_index(drop=True)
    df_final = pd.DataFrame(tags_binarias, columns=mlb.classes_)
    df_final.insert(0, "resposta", df["resposta"])  # agora os índices batem
    
    return df_final


def get_word2vec_features(texts, w2v_model, vector_size):
    features = []
    for text in texts:
        tokens = word_tokenize(text, language="portuguese")
        # Filtra só tokens que estão no vocabulário
        valid_tokens = [token for token in tokens if token in w2v_model.wv]
        if valid_tokens:
            # Média dos vetores das palavras
            vec = np.mean([w2v_model.wv[token] for token in valid_tokens], axis=0)
        else:
            # Se não tiver nenhuma palavra no vocabulário, retorna vetor de zeros
            vec = np.zeros(vector_size)
        features.append(vec)
    return np.vstack(features)

class medidas_multilabel:
    def __init__(self, df, name_cols=['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']):
        self.df = df
        self.name_cols = name_cols

    
    def acrescentar_colunas(self):
        self.df['acerto_1'] = np.nan
        self.df['acerto_2'] = np.nan
        self.df['acerto_3'] = np.nan
        for i in range(len(self.df)):
            if (self.df.iloc[i]["TAG_1"] == self.df.iloc[i][self.name_cols[0]]) or (self.df.iloc[i]["TAG_1"] == self.df.iloc[i][self.name_cols[1]]) or (self.df.iloc[i]["TAG_1"] == self.df.iloc[i][self.name_cols[2]]):
                self.df.iloc[i, self.df.columns.get_loc('acerto_1')] = 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('acerto_1')] = 0

            if (self.df.iloc[i]["TAG_2"] == self.df.iloc[i][self.name_cols[0]]) or (self.df.iloc[i]["TAG_2"] == self.df.iloc[i][self.name_cols[1]]) or (self.df.iloc[i]["TAG_2"] == self.df.iloc[i][self.name_cols[2]]):
                self.df.iloc[i, self.df.columns.get_loc('acerto_2')] = 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('acerto_2')] = 0

            if (self.df.iloc[i]["TAG_3"] == self.df.iloc[i][self.name_cols[0]]) or (self.df.iloc[i]["TAG_3"] == self.df.iloc[i][self.name_cols[1]]) or (self.df.iloc[i]["TAG_3"] == self.df.iloc[i][self.name_cols[2]]):
                self.df.iloc[i, self.df.columns.get_loc('acerto_3')] = 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('acerto_3')] = 0

        self.df['qtd_humano'] = self.df[self.name_cols].notnull().sum(axis=1)
        self.df['qtd_modelo'] = self.df[['TAG_1', 'TAG_2', 'TAG_3']].notnull().sum(axis=1)
        self.df['qtd_acerto'] = self.df[['acerto_1', 'acerto_2', 'acerto_3']].sum(axis=1)
        self.df['precision_sample'] = np.where(self.df['qtd_modelo'] == 0,
                                               0,
                                               self.df['qtd_acerto'] / self.df['qtd_modelo']
                                               )
        self.df['recall_sample'] = self.df['qtd_acerto'].divide(self.df['qtd_humano'])

    def recall_sample(self):
        recall_sample = self.df['recall_sample'].mean() * 100
        return recall_sample
    
    def recall_micro(self):
        qtd_acerto_geral = self.df[['acerto_1', 'acerto_2', 'acerto_3']].sum().sum()
        qtd_class_hum = self.df[['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']].count().sum()
        percentual_acerto_geral = qtd_acerto_geral / qtd_class_hum * 100
        return percentual_acerto_geral
    
    def precision_sample(self):
        precision_sample = self.df['precision_sample'].mean() * 100
        return precision_sample
    
    def precision_micro(self):
        qtd_acerto_geral = self.df[['acerto_1', 'acerto_2', 'acerto_3']].sum().sum()
        qtd_class_modelo = self.df[['TAG_1', 'TAG_2', 'TAG_3']].count().sum()
        precision_micro = qtd_acerto_geral / qtd_class_modelo * 100
        return precision_micro
    
    def f1_micro(self, precision_micro, recall_micro):
        f1_micro = ( (2 * precision_micro * recall_micro) / (precision_micro + recall_micro))
        return f1_micro
    
    
    def hamming_loss_manual(self):
        soma_fp = 0
        soma_fn = 0
        for i in range(len(self.df)):
            if pd.notna(self.df.iloc[i]["TAG_1"]):
                if self.df.iloc[i]["TAG_1"] not in [self.df.iloc[i][self.name_cols[0]], self.df.iloc[i][self.name_cols[1]], self.df.iloc[i][self.name_cols[2]]]:
                    soma_fp += 1
            if pd.notna(self.df.iloc[i]["TAG_2"]):
                if self.df.iloc[i]["TAG_2"] not in [self.df.iloc[i][self.name_cols[0]], self.df.iloc[i][self.name_cols[1]], self.df.iloc[i][self.name_cols[2]]]:
                    soma_fp += 1
            if pd.notna(self.df.iloc[i]["TAG_3"]):
                if self.df.iloc[i]["TAG_3"] not in [self.df.iloc[i][self.name_cols[0]], self.df.iloc[i][self.name_cols[1]], self.df.iloc[i][self.name_cols[2]]]:
                    soma_fp += 1
            if pd.notna(self.df.iloc[i][self.name_cols[0]]):
                if self.df.iloc[i][self.name_cols[0]] not in [self.df.iloc[i]['TAG_1'], self.df.iloc[i]['TAG_2'], self.df.iloc[i]['TAG_3']]:
                    soma_fn += 1
            if pd.notna(self.df.iloc[i][self.name_cols[1]]):
                if self.df.iloc[i][self.name_cols[1]] not in [self.df.iloc[i]['TAG_1'], self.df.iloc[i]['TAG_2'], self.df.iloc[i]['TAG_3']]:
                        soma_fn += 1
            if pd.notna(self.df.iloc[i][self.name_cols[2]]):
                if self.df.iloc[i][self.name_cols[2]] not in [self.df.iloc[i]['TAG_1'], self.df.iloc[i]['TAG_2'], self.df.iloc[i]['TAG_3']]:
                        soma_fn += 1
        hamming_loss_manual = (soma_fp + soma_fn) / (len(self.df) * 3)
        return hamming_loss_manual, soma_fp, soma_fn
    


def labels_em_cols(tags_list, data, col_name):
                for tag in tags_list:
                    coluna = list()
                    for linha_tag in data[col_name]:
                        if tag in linha_tag:
                            coluna.append(1)
                        else:
                            coluna.append(0)
                    data[tag] = coluna
                return data

def remover_NsNr(dados: pd.DataFrame, label_NsNr: str, inplace: bool = False) -> pd.DataFrame:
    """
    Aplica as regras para a label NS/NR nas colunas TAG_1, TAG_2, TAG_3:
      1) Se TAG_1 == NS/NR  -> TAG_2 = None, TAG_3 = None
      2) Se TAG_2 == NS/NR  -> TAG_2 recebe TAG_3, TAG_3 = None
      3) Se TAG_3 == NS/NR  -> TAG_3 = None
    A ordem de aplicação é importante (1 -> 2 -> 3).
    """
    if not inplace:
        df = dados.copy()
    else:
        df = dados

    req = {"TAG_1", "TAG_2", "TAG_3"}
    falta = req - set(df.columns)
    if falta:
        raise KeyError(f"Colunas ausentes: {sorted(falta)}")

    # Garantir dtype objeto para aceitar None
    df[["TAG_1", "TAG_2", "TAG_3"]] = df[["TAG_1", "TAG_2", "TAG_3"]].astype("object")

    # 1) TAG_1 == NS/NR -> zera TAG_2 e TAG_3
    m1 = df["TAG_1"].eq(label_NsNr)
    df.loc[m1, ["TAG_2", "TAG_3"]] = None

    # 2) (onde NÃO caiu na 1) e TAG_2 == NS/NR -> sobe TAG_3 para TAG_2 e zera TAG_3
    m2 = (~m1) & df["TAG_2"].eq(label_NsNr)
    df.loc[m2, "TAG_2"] = df.loc[m2, "TAG_3"].values
    df.loc[m2, "TAG_3"] = None

    # 3) (onde NÃO caiu na 1) e TAG_3 == NS/NR -> zera TAG_3
    m3 = (~m1) & df["TAG_3"].eq(label_NsNr)
    df.loc[m3, "TAG_3"] = None

    return df
