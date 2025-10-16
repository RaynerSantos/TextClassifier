from fastapi import FastAPI

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import joblib
from sklearn.pipeline import Pipeline
import io
import pyodbc

from src.preprocess_text import preprocess_text, prepare_multilabel_data, Pre_Processamento, remover_NsNr, labels_em_cols
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

from pathlib import Path

app = FastAPI()

@app.get('/')
def initial_app():
    return("Aplicação iniciada com Sucesso!")


caminho = Path(__file__).resolve().parent

col_valor_nota = 'Q1'
col_texto_promo = 'Q2A'
col_texto_neutro = 'Q2B'
col_texto_detra = 'Q2C'
label_NsNr = 'Não sabe / Não respondeu'
col_id = "CODIGO"


def load_model(model_name):
    model = joblib.load(model_name)
    return model

model_promotor = load_model(caminho / 'modelo_multilabel_unimed_promotor.pkl')
model_neutro = load_model(caminho / 'modelo_multilabel_unimed_neutro.pkl')
model_detrator = load_model(caminho / 'modelo_multilabel_unimed_detrator.pkl')

data = conexao.select_banco()
print("\nBanco de dados:\n", data.head(10))

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
        print("\n----- Promotor -----\n")
    elif categoria == "Neutro":
        model = model_neutro
        df = df_neutro
        labels = labels_neutro
        col_texto = col_texto_neutro
        print("\n----- Neutro -----\n")
    elif categoria == "Detrator":
        model = model_detrator
        df = df_detra
        labels = labels_detrator
        col_texto = col_texto_detra
        print("\n----- Detrator -----\n")
    # print("\nLabels:\n", labels)
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
                # print("labels[idx]:\t", labels[idx])
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

df_sql = conexao.select_banco()
conexao.update_banco(df_sql=df_sql, df_classificacao=df_result, ID=col_id, categoria_NPS="Categoria_NPS")

@app.get('/')
def conclusion_app():
    return("✅ Classificações enviadas para o banco de dados no servidor com sucesso!")