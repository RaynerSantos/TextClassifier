import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
import io
import pyodbc

from src.preprocess_text import preprocess_text, prepare_multilabel_data, Pre_Processamento, labels_em_cols, remover_NsNr
from src.utils import medidas_multilabel, labels_promotor, labels_neutro, labels_detrator, conexao, get_proba_matrix_ovr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from nltk.tokenize import word_tokenize

# scikit-multilearn
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# CSS personalizado
st.markdown(
    """
    <style>
    /* Cor de fundo da página */
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
    }

    /* Cor de fundo do cabeçalho */
    [data-testid="stHeader"] {
        background-color: #000000;
    }

    /* Esconde o menu lateral */
    [data-testid="stSidebar"] {
        display: none;  /* 👈 Esconde o menu lateral */
    }

    /* Remove o espaço lateral */
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 0;  /* 👈 Remove o espaço lateral */
    }

    /* Cor de fundo da barra lateral */
    [data-testid="stSidebar"] {
        background-color: #333333;
    }

    /* Cor do título */
    h1 {
    color: white !important;
    text-align: center;
    font-weight: bold;
}

    /* Cor do subtítulo */
    h2 {
        color: #FFD700;
    }

    /* Cor do texto normal */
    p, span {
        color: #FFFFFF;
    }

    /* Cor dos botões */
    button {
        background-color: #20541B !important;
        color: white !important;
    }

    /* Caixa do formulário */
    div[data-testid="stForm"] {
        background-color: #1e1e1e;  /* cinza escuro */
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #444444;
        max-width: 600px;
        margin: auto;
    }

    /* Campos de texto */
    input, select, textarea {
        /* background-color: #2e2e2e !important; */
        /* color: white !important; */
        border: none !important;
        border-radius: 6px !important;
    }

    /* Botão */
    button[kind="primary"] {
        background-color: #20541B !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* table {
    background-color: #000000;
    color: white;
    border-collapse: collapse;
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
    font-size: 14px;
    }
    th, td {
        border: 1px solid #333;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #111111;
        color: #FFFFFF;
    }
    tr:nth-child(even) {
        background-color: #1c1c1c;
    } */
    </style>
    """,
    unsafe_allow_html=True
)

col_valor_nota = 'Q1'
col_texto_promo = 'Q2A'
col_texto_neutro = 'Q2B'
col_texto_detra = 'Q2C'
label_NsNr = 'Não sabe / Não respondeu'
col_id = "CODIGO"

# Configurações da página
st.set_page_config(page_title="PÓS Codificação - Cielo NPS", layout="centered")  # "wide"

#=== Título ===#
st.title("PÓS Codificação - UNIMED")
st.write("")
st.write("")

# 1. Carregue o modelo e o vetorizador
@st.cache_resource
def load_model(model_name):
    model = joblib.load(model_name)
    return model

model_promotor = load_model('modelo_multilabel_unimed_promotor.pkl')
model_neutro = load_model('modelo_multilabel_unimed_neutro.pkl')
model_detrator = load_model('modelo_multilabel_unimed_detrator.pkl')

# if "RODAR_CLASSIFICACAO" not in st.session_state:
#     st.session_state.RODAR_CLASSIFICACAO = False

data = conexao.select_banco()

# st.write("💾 Banco de dados UNIMED")
# st.dataframe(data.head(10), hide_index=True)
# st.write("")


# if st.button("Rodar classificação"):
data['Categoria_NPS'] = np.where(data[col_valor_nota] <= 7, "Detrator", 
                                np.where(data[col_valor_nota] <= 9, "Neutro", 
                                        np.where(data[col_valor_nota] <= 11, "Promotor", "Verificar")))
df_promo = data.loc[data['Categoria_NPS'] == "Promotor"]
df_neutro = data.loc[data["Categoria_NPS"] == "Neutro"]
df_detra = data.loc[data["Categoria_NPS"] == "Detrator"]

df_resultados = []
X_pred_models = []
for categoria in ["Promotor", "Neutro", "Detrator"]:
    if categoria == "Promotor":
        model = model_promotor
        df = df_promo
        labels = labels_promotor
        col_texto = col_texto_promo
    elif categoria == "Neutro":
        model = model_neutro
        df = df_neutro
        labels = labels_neutro
        col_texto = col_texto_neutro
    elif categoria == "Detrator":
        model = model_detrator
        df = df_detra
        labels = labels_detrator
        col_texto = col_texto_detra

    # 5. Predição
    # Pre-processamento igual ao treinamento
    # X_pred = df[col_texto].fillna("").astype(str)
    # Usar list comprehension com tqdm para barras de progresso em scripts
    X_pred = [preprocess_text(text) for text in tqdm(df[col_texto], desc="Pré-processando treino")]
    
    proba_matrix = get_proba_matrix_ovr(model, X_pred)  # shape: (n_amostras, n_labels)

    threshold = 0.5  # limiar/threshold 0.35
    tag_columns = [f"TAG_{i+1}" for i in range(3)]
    tags_preditas = []

    for row in proba_matrix:
        # Índices das 3 maiores probabilidades (em ordem decrescente)
        top_indices = row.argsort()[-3:][::-1]
        top_tags = []
        for idx in top_indices:
            if row[idx] >= threshold:
                top_tags.append(labels[idx])
            else:
                top_tags.append(None)
        tags_preditas.append(top_tags)

    # 3. Adiciona ao DataFrame
    for i, col in enumerate(tag_columns):
        df[col] = [tags[i] for tags in tags_preditas]

    df_resultados.append(df)
    X_pred_models.append(X_pred)

df_result = pd.concat(df_resultados, axis=0)
df_result = remover_NsNr(dados=df_result, label_NsNr=label_NsNr)
st.session_state.df_result = df_result

# # Exibe e oferece download
# st.success("✅ Classificação realizada com sucesso!")
# st.dataframe(df_result.head(10), hide_index=True)
buffer = io.BytesIO()
df_result.to_excel(buffer, index=False)
buffer.seek(0)
st.download_button(
    label="📥 Baixar planilha com resultados",
    data=buffer,
    file_name="resultado_classificacao.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
# st.session_state.RODAR_CLASSIFICACAO = True


# if st.session_state.RODAR_CLASSIFICACAO:
# #=== Automatizar para inserir automaticamente no banco do SQL Server ===#
# # Escolher a coluna que representa o ID (usuário pode selecionar)
# st.write("")
# st.write("")
# st.subheader("📤 Salvar a PÓS Codificação UNIMED no banco do servidor")
# col_id = st.selectbox(
#     "👇 Selecione a coluna que representa o ID da entrevista:",
#     st.session_state.df_result.columns
#     )
# if st.button("Enviar classificações para o Banco no servidor"):
df_sql = conexao.select_banco()
conexao.update_banco(df_sql=df_sql, df_classificacao=st.session_state.df_result, ID=col_id, categoria_NPS="Categoria_NPS")
st.success("✅ Classificações enviadas para o banco de dados no servidor com sucesso!")


