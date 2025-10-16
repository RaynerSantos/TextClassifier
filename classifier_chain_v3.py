import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Fonte
from app.src.preprocess_text import preprocess_text, prepare_multilabel_data, get_word2vec_features, Pre_Processamento, labels_em_cols, medidas_multilabel, extrair_entrevistado, remover_NsNr

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# scikit-multilearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain 
from sklearn.svm import LinearSVC


# --- 2. Carregar os Dados ---
print("Carregando dados...")
file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
sheet_name = "Promotor" # TREINO  |  Completo_semNS

pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name) 
data = pre_processador.pre_processamento()
data = data.loc[(data['ONDA'] != "Set/25")]
# data = data.loc[data["NPS"]=="Neutro"]

# print('\ndataframe:\n', train_df[['resposta','Classificacao_humana_1','Classificacao_humana_2','Classificacao_humana_3']])
print(f'\nShape do banco de dados completo:\t{data.shape}')

tags_df = pre_processador.pre_processamento_tags()
print(f'\nFrequência das TAGs:\n{tags_df}')
print(f'\nQtd TAGs: {len(tags_df)}')

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
data_multilabel['processed_text'] = [preprocess_text(text, modelo="Promotor") for text in tqdm(data_multilabel['resposta'], desc="Pré-processando dos dados")]

# train_df = data_multilabel.loc[(data_multilabel['ONDA'] != "Set/25")] 
# test_df = data_multilabel.loc[(data_multilabel['ONDA'] == "Set/25")] 
train_df = data_multilabel.loc[(data_multilabel['NPS'] != "Promotor_TESTE")] 
test_df = data_multilabel.loc[(data_multilabel['NPS'] == "Promotor_TESTE")] # Categoria NPS e de Julho/25

print(f'\nShape de train_df:\t{train_df.shape}')
print(f'\nShape de test_df:\t{test_df.shape}')

# --- 4. Vetorização de Texto (TF-IDF) ---
print("\n--- Vetorizando textos com TF-IDF... ---")
tfidf_vectorizer = TfidfVectorizer(
                min_df=2, # só considera palavras/ngrams que aparecem em pelo menos 3 documentos - Se você tem 100 textos, e min_df=3, só serão consideradas as palavras (ou n-gramas) que aparecem em pelo menos 3 textos diferentes.
                max_df=0.7,  # ignora palavras/ngrams que aparecem em mais de 80% dos documentos - Se você tem 100 textos, e max_df=0.8, as palavras que aparecem em mais de 80% dos textos serão ignoradas.
                ngram_range=(1, 4), # usa palavras isoladas e pares de palavras, Ou seja, o modelo vai olhar para palavras sozinhas e para pares de palavras consecutivas.
                sublinear_tf=True, # (TF logarítmico ajuda muito em LR)
                analyzer='char_wb', # Some char n-grams (ex.: analyzer='char_wb', ngram_range=(3,5)); isso ajuda em erros ortográficos/ariações.
                max_features=None, # limita o número de características (features) a 10000 - Se o vocabulário gerado for maior que 10000, só as 10000 palavras/ngrams mais frequentes serão mantidas.
                strip_accents='unicode' # desconsidera acentos
                )

# tfidf_vectorizer.fit_transform(data_multilabel['processed_text'])
X_train = tfidf_vectorizer.fit_transform(train_df['processed_text'])
X_test = tfidf_vectorizer.transform(test_df['processed_text'])
Y_train = train_df[labels].values

print(f"Shape de X_train (vetorizado): {X_train.shape}")
print(f"Shape de X_test (vetorizado): {X_test.shape}")
print(f"Shape de Y_train (rótulos): {Y_train.shape}")


# --- 5. Modelagem e Treinamento (Classificação Multilabel) ---

print("\n--- Treinando Modelos Multilabel ---\n")

# 5.5. Regressão Logística
# Modelo MultiOutput Regressão Logística
print("\n--- Treinando Regressão Logística ---\n")

rl_multi = ClassifierChain(LinearSVC())

rl_multi.fit(X_train, Y_train)
# # Prever probabilidades
# preds_ml = rl_multi.predict_proba(X_test)  # lista de arrays
# # Montar array (n_amostras, n_classes) com probabilidades das classes positivas
# proba_matrix = np.array([p[:,1] for p in preds_ml]).T  # shape: (n_amostras, n_classes)

preds_ml = [estimator.predict_proba(X_test) for estimator in rl_multi.estimators_]
probs_pos = []
for p in preds_ml:
    # p pode ser (n_amostras, 2) ou (n_amostras,)
    if p.ndim == 2 and p.shape[1] > 1:
        probs_pos.append(p[:, 1])
    else:
        probs_pos.append(p.ravel())
    proba_matrix = np.vstack(probs_pos).T  # (n_amostras, n_classes)

# print('\nproba_matrix:\n', proba_matrix, '\n')

# Para cada texto, pegar até 3 TAGs mais prováveis (maiores probabilidades)
top_k = 3
preds_topk = np.zeros_like(proba_matrix, dtype=int)
for i, row in enumerate(proba_matrix):
    # indices das top 3 probabilidades
    top_indices = row.argsort()[-top_k:][::-1]
    # Marca como 1 as top TAGs
    preds_topk[i, top_indices] = 1

# DataFrame de predições
result_df_ml = test_df.copy()

# (Opcional) Adicionar coluna com nomes das TAGs previstas (listas de TAGs)
def tags_preditas(row):
    return [labels[i] for i, val in enumerate(row) if val == 1]

# result_df_ml['Tags_Preditas_RF'] = pred_df_ml.values.tolist()
# result_df_ml['Tags_Preditas_RF'] = result_df_ml['Tags_Preditas_RF'].apply(tags_preditas)

# Para cada linha, pegue os índices das top 3 probabilidades
THRESHOLD = 0.5  # Defina o limiar desejado 0.35
tag_columns = [f"TAG_{i+1}" for i in range(3)]  # ['TAG_1', 'TAG_2', 'TAG_3']
tags_preditas = []

for row in proba_matrix:
    # Pega os índices das 3 maiores probabilidades em ordem decrescente
    top_indices = row.argsort()[-3:][::-1]  # índices das 3 maiores probabilidades
    top_tags = []
    for idx in top_indices:
        # Só inclui se a probabilidade for >= threshold
        if row[idx] >= THRESHOLD:
            top_tags.append(labels[idx])
        else:
            top_tags.append(None)
    tags_preditas.append(top_tags)

# Criar as colunas TAG_1, TAG_2, TAG_3
for i, col in enumerate(tag_columns):
    result_df_ml[col] = [tags[i] for tags in tags_preditas]


# colunas_para_juntar = ['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']
# # colunas_para_juntar = ['NET_1', 'NET_2', 'NET_3']
# result_df_ml['TAGs'] = result_df_ml[colunas_para_juntar].astype(str).agg(' '.join, axis=1)

# medidas = medidas_multilabel(df=result_df_ml, name_cols=colunas_para_juntar)
# assertividade = medidas.assertividade()
# hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()


# print('\nresultado:\n', result_df_ml[['TAG_1','TAG_2','TAG_3']].head(30))
# print('Quantidade de TAG_1 vazias: ', result_df_ml['TAG_1'].isna().sum())
# print('Quantidade de TAG_2 vazias: ', result_df_ml['TAG_2'].isna().sum())
# print('Quantidade de TAG_3 vazias: ', result_df_ml['TAG_3'].isna().sum())

# print(f'\nTHRESHOLD: {THRESHOLD}')
# print(f"Média geral do percentual de acerto por texto:\t **{round(assertividade, 1)}%**")
# print(f'Hamming Loss manual: ', round(hamming_loss_manual,3))
# print(f'Quantidade de Falsos Positivos: ', fp)
# print(f'Quantidade de Falsos Negativos: ', fn, '\n')

# print('Quantidade total de TAGs previstas (não nulas): ', result_df_ml[['TAG_1']].value_counts().sum() + result_df_ml[['TAG_2']].value_counts().sum() + result_df_ml[['TAG_3']].value_counts().sum())
# print('Quantidade total de TAGs da classificação humana (não nulas): ', result_df_ml[['Classificacao_humana_1']].value_counts().sum() + result_df_ml[['Classificacao_humana_2']].value_counts().sum() + result_df_ml[['Classificacao_humana_3']].value_counts().sum())

# df_teste = labels_em_cols(tags_list=labels, data=result_df_ml, col_name='TAGs')
# previsao = rl_multi.predict(X_test)
# y_true = df_teste[labels].values
# hamming_loss_model = hamming_loss(y_true=y_true, y_pred=previsao)
# print(f"\nHamming Loss (fração de tags incorretamente previstos):\t **{round(hamming_loss_model*100, 1)}%**")
# print('jaccard: **', round(jaccard_score(y_true, previsao, average='samples'),2), '**')
# print('F1-micro: **', round(f1_score(y_true, previsao, average='micro'),2), '**')
# print('F1-macro: **', round(f1_score(y_true, previsao, average='macro'),2), '**')
# print('F1-samples: **', round(f1_score(y_true, previsao, average='samples'),2), '**')
# print('Precision-micro: **', round(precision_score(y_true, previsao, average='micro'),2), '**')
# print('Precision-macro: **', round(precision_score(y_true, previsao, average='macro'),2), '**')
# print('Precision-samples: **', round(precision_score(y_true, previsao, average='samples'),2), '**')
# print('Recall-micro: **', round(recall_score(y_true, previsao, average='micro'),2), '**')
# print('Recall-macro: **', round(recall_score(y_true, previsao, average='macro'),2), '**')
# print('Recall-samples: **', round(recall_score(y_true, previsao, average='samples'),2), '**')
# print('\nClassification report: \n', classification_report(y_true, previsao))

# # Exporta
result_df_ml = result_df_ml.drop(columns=[c for c in labels if c in test_df.columns], errors='ignore')
result_df_ml = result_df_ml.drop(columns=[c for c in result_df_ml.columns if 'TAGs' in c], errors='ignore')
result_df_ml = remover_NsNr(dados=result_df_ml, label_NsNr='Não sabe / Não respondeu')
# result_df_ml.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\RegLog.xlsx", index=False)
# print('\nArquivo exportado com sucesso!')
print("\nTreinamento Regressão Logística concluído.")
print("\n--- Processo Concluído ---")
