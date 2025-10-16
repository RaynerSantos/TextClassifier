import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import streamlit as st
import joblib

# Fonte
from preprocess_text import preprocess_text, prepare_multilabel_data, get_word2vec_features, Pre_Processamento, labels_em_cols, medidas_multilabel, extrair_entrevistado, remover_NsNr

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# scikit-multilearn
from sklearn.multiclass import OneVsRestClassifier

"""
Especificação:
    Promotor: Com stop word e ngram do TF-IDF (1,4)
    Neutro: Com stop word e ngram do TF-IDF (1,4)
    Detrator: Sem stop word e ngram do TF-IDF (2,4)
"""

# --- 2. Carregar os Dados ---
print("Carregando dados...")
file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
sheet_name = "Neutro"
pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name)

data = pre_processador.pre_processamento()

data_multilabel = prepare_multilabel_data(data)
# print('\ntrain_df após o prepare_multilabel:\n', train_df)
print('\nColunas/Labels:\n', data_multilabel.columns[1:])

# labels = tags_df['TAGs'].str.lower()
labels = data_multilabel.columns[1:]
print(f'\nQuantidade de labels: {len(labels)}')

data_multilabel = pd.concat(
    objs=[data[['ID_UNICO','CODIGO','id','ONDA','NPS','Classificacao_humana_1','Classificacao_humana_2','Classificacao_humana_3']].reset_index(drop=True),
          data_multilabel.reset_index(drop=True)],
    axis=1
)

# --- 3. Pré-processamento de Texto ---
print("\n--- Pré-processando textos ---")
data_multilabel['resposta'] = data_multilabel['resposta'].fillna('')
# Usar list comprehension com tqdm para barras de progresso em scripts
data_multilabel['processed_text'] = [preprocess_text(text) for text in tqdm(data_multilabel['resposta'], desc="Pré-processando dos dados")]

train_df = data_multilabel.loc[(data_multilabel['ONDA'] != "Set/25")] 
test_df = data_multilabel.loc[(data_multilabel['ONDA'] == "Set/25")] 

print(f'\nShape de train_df:\t{train_df.shape}')
print(f'\nShape de test_df:\t{test_df.shape}')


# --- 4. Modelagem e Treinamento (Classificação Multilabel) ---
# 4.1. Regressão Logística
# Modelo MultiOutput Regressão Logística
print("\n\n--- Treinando Regressão Logística ---\n")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
                min_df=2, # só considera palavras/ngrams que aparecem em pelo menos 3 documentos - Se você tem 100 textos, e min_df=3, só serão consideradas as palavras (ou n-gramas) que aparecem em pelo menos 3 textos diferentes.
                max_df=0.7,  # ignora palavras/ngrams que aparecem em mais de 80% dos documentos - Se você tem 100 textos, e max_df=0.8, as palavras que aparecem em mais de 80% dos textos serão ignoradas.
                ngram_range=(1, 4), # usa palavras isoladas e pares de palavras, Ou seja, o modelo vai olhar para palavras sozinhas e para pares de palavras consecutivas.
                sublinear_tf=True, # (TF logarítmico ajuda muito em LR)
                analyzer='char_wb', # Some char n-grams (ex.: analyzer='char_wb', ngram_range=(3,5)); isso ajuda em erros ortográficos/ariações.
                max_features=None, # limita o número de características (features) a 10000 - Se o vocabulário gerado for maior que 10000, só as 10000 palavras/ngrams mais frequentes serão mantidas.
                strip_accents='unicode' # desconsidera acentos
                )),
    ('clf', OneVsRestClassifier(
        LogisticRegression(solver='liblinear', # liblinear | lbfgs
                           max_iter=2000,
                           C=2.0, 
                           class_weight='balanced')
    ))
])

# Treinando o pipeline
pipeline.fit(train_df['processed_text'], train_df[labels].values)

# Salvando o pipeline inteiro (TF-IDF + modelo) em 1 arquivo
joblib.dump(pipeline, r'C:\PROJETOS\TextClassifier\Codigos\Unimed\app\modelo_multilabel_unimed_neutro.pkl')
print("Pipeline salvo com sucesso!")