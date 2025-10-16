import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from app.src.preprocess_text import preprocess_text, prepare_multilabel_data, get_word2vec_features, Pre_Processamento, labels_em_cols, medidas_multilabel, extrair_entrevistado, remover_NsNr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score

# Vetorização por Word2Vec
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Modelos
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

LABEL_NsNr = 'Não sabe / Não respondeu'

# --- 2. Carregar os Dados ---
print("Carregando dados...")
file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
sheet_name = "Promotor" # Completo | Detrator | Neutro | Promotor

pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name) 

data = pre_processador.pre_processamento()
# data = data.loc[data["NPS"]=="Detrator"]
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
    objs=[data[['ID_UNICO','ONDA','NPS','Classificacao_humana_1','Classificacao_humana_2','Classificacao_humana_3']].reset_index(drop=True),
          data_multilabel.reset_index(drop=True)],
    axis=1
)

# --- 3. Pré-processamento de Texto ---

print("\n--- Pré-processando textos ---")
data_multilabel['resposta'] = data_multilabel['resposta'].fillna('')
# Usar list comprehension com tqdm para barras de progresso em scripts
data_multilabel['processed_text'] = [preprocess_text(text) for text in tqdm(data_multilabel['resposta'], desc="Pré-processando dos dados")]


# train_df = data_multilabel.loc[(data_multilabel['ONDA'] != "Jun25(Mensal)") & (data_multilabel['ONDA'] != "Ago25(Mensal)")] 
# test_df = data_multilabel.loc[(data_multilabel['ONDA'] == "Jun25(Mensal)") | (data_multilabel['ONDA'] == "Ago25(Mensal)")] 
train_df = data_multilabel.loc[(data_multilabel['NPS'] != "Promotor_TESTE")] 
test_or = data_multilabel.loc[(data_multilabel['NPS'] == "Promotor_TESTE")] # Categoria NPS e de Julho/25

print(f'\nShape de train_df:\t{train_df.shape}')
print(f'\nShape de test_df:\t{test_or.shape}')


# --- 4. Vetorização de Texto (TF-IDF) ---
print("\n--- Vetorizando textos com TF-IDF... ---")
tfidf_vectorizer = TfidfVectorizer(min_df=2, # só considera palavras/ngrams que aparecem em pelo menos 3 documentos - Se você tem 100 textos, e min_df=3, só serão consideradas as palavras (ou n-gramas) que aparecem em pelo menos 3 textos diferentes.
                                   max_df=0.7,  # ignora palavras/ngrams que aparecem em mais de 80% dos documentos - Se você tem 100 textos, e max_df=0.8, as palavras que aparecem em mais de 80% dos textos serão ignoradas.
                                   ngram_range=(1, 4), # usa palavras isoladas e pares de palavras, Ou seja, o modelo vai olhar para palavras sozinhas e para pares de palavras consecutivas.
                                   sublinear_tf=True, # (TF logarítmico ajuda muito em LR)
                                   analyzer='char_wb', # Some char n-grams (ex.: analyzer='char_wb', ngram_range=(3,5)); isso ajuda em erros ortográficos/variações.
                                   max_features=None, # limita o número de características (features) a 10000 - Se o vocabulário gerado for maior que 10000, só as 10000 palavras/ngrams mais frequentes serão mantidas.
                                   strip_accents='unicode' # desconsidera acentos
                                   )

tfidf_vectorizer.fit_transform(data_multilabel['processed_text'])
X_train = tfidf_vectorizer.transform(train_df['processed_text'])
Y_train = train_df[labels].values


print("\n--- Treinando Regressão Logística ---\n")
rl_multi = OneVsRestClassifier(LogisticRegression(solver='liblinear', # liblinear | lbfgs
                                                    max_iter=3000,
                                                    C=2.0,  # 2.0
                                                    class_weight='balanced'))

rl_multi.fit(X_train, Y_train)


#===== Loop para simular o ERRO ABSOLUTO MÉDIO e o IC da diferença em cada TAG =====#

col_metrics = list(labels) + ['MAE']
df_metrics = pd.DataFrame(columns=col_metrics)

for epoch in range(1000):
    # # Dividir em treino e teste (80% treino, 20% teste) com stratify
    # index_X_train, index_X_test =  train_test_split(range(len(data_multilabel)), test_size=0.06)
    # test_df = data_multilabel.iloc[index_X_test]
    # # teste = data.iloc[index_X_test]
    # train_df = data_multilabel.iloc[index_X_train]

    test_df = test_or.sample(n=len(test_or), replace=True, random_state=(epoch+1))

    X_test = tfidf_vectorizer.transform(test_df['processed_text'])
    y_true = test_df[labels].values

    print(f"Shape de X_train (vetorizado): {X_train.shape}")
    print(f"Shape de Y_train (rótulos): {Y_train.shape}")


    preds_ml = [estimator.predict_proba(X_test) for estimator in rl_multi.estimators_]
    probs_pos = []
    for p in preds_ml:
        # p pode ser (n_amostras, 2) ou (n_amostras,)
        if p.ndim == 2 and p.shape[1] > 1:
            probs_pos.append(p[:, 1])
        else:
            probs_pos.append(p.ravel())
        proba_matrix = np.vstack(probs_pos).T  # (n_amostras, n_classes)


    # Para cada texto, pegar até 3 TAGs mais prováveis (maiores probabilidades)
    top_k = 3
    preds_topk = np.zeros_like(proba_matrix, dtype=int)
    for i, row in enumerate(proba_matrix):
        # indices das top 3 probabilidades
        top_indices = row.argsort()[-top_k:][::-1]
        # Marca como 1 as top TAGs
        preds_topk[i, top_indices] = 1

    # DataFrame de predições
    result_df_ml = test_df.reset_index(drop=True).copy()

    def tags_preditas(row):
        return [labels[i] for i, val in enumerate(row) if val == 1]

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
    result_df_ml = remover_NsNr(dados=result_df_ml, label_NsNr=LABEL_NsNr)

    colunas_para_juntar = ['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']
    # colunas_para_juntar = ['NET_1', 'NET_2', 'NET_3']
    result_df_ml['TAGs'] = result_df_ml[colunas_para_juntar].astype(str).agg(' '.join, axis=1)

    medidas = medidas_multilabel(df=result_df_ml, name_cols=colunas_para_juntar)
    medidas.acrescentar_colunas()
    recall_sample = medidas.recall_sample()
    recall_micro = medidas.recall_micro()
    precision_sample = medidas.precision_sample()
    precision_micro = medidas.precision_micro()
    f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
    hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()

    print('\n----- Métricas -----')
    print(f"Média geral do percentual de acerto por texto (recall-sample):\t **{round(recall_sample, 1)}%**")
    print(f"Percentual de acerto geral (recall-micro):\t **{round(recall_micro, 1)}%**")
    print(f"Média geral do percentual de acerto por texto sobre o modelo (precision-sample):\t **{round(precision_sample, 1)}%**")
    print(f"Percentual de acerto geral sobre o modelo (precision-micro):\t **{round(precision_micro, 1)}%**")
    print(f"F1-micro:\t **{round(f1_micro, 1)}%**")
    print(f'Hamming Loss manual: ', round(hamming_loss_manual,3))

    def freq_tags(data, col_class, chave="humano"):
        # Consolidar tags
        data["tags_resposta"] = data[col_class].values.tolist()

        data["tags_resposta"] = data["tags_resposta"].apply(
                    # lambda x: list({str(i).strip().lower() for i in x if pd.notna(i)})
                    lambda x: list({str(i).strip() for i in x if pd.notna(i)})
                )

        tags_df = (
            data.melt(
                value_vars=col_class,
                var_name="Classificacao",
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

        if chave == "humano":
            tags_df.columns = ['TAGs_hum', 'n_hum', 'percentual_hum']
        else:
            tags_df.columns = ['TAGs_modelo', 'n_modelo', 'percentual_modelo']

        return tags_df

    test_df = test_df[[col for col in test_df.columns if col not in labels]]
    tags_df_hum = freq_tags(data=test_df, 
                        col_class=["Classificacao_humana_1", "Classificacao_humana_2", "Classificacao_humana_3"],
                        chave="humano")

    result_df_ml = result_df_ml[[col for col in result_df_ml.columns if col not in labels]]
    result_df_ml = result_df_ml[[col for col in result_df_ml.columns if 'TAGs' not in col]]
    tags_df_modelo = freq_tags(data=result_df_ml, 
                        col_class=["TAG_1", "TAG_2", "TAG_3"],
                        chave="modelo")

    tags_df_hum.index = tags_df_hum['TAGs_hum']
    tags_df_modelo.index = tags_df_modelo['TAGs_modelo']
    result = pd.concat([tags_df_hum, tags_df_modelo], axis=1)

    result['dif_abs'] = np.abs(result['percentual_hum'] - result['percentual_modelo']).fillna(result['percentual_hum']).fillna(result['percentual_modelo'])

    # colunas que devem virar 0 quando a label não existe
    cols_zero = ['n_hum', 'n_modelo', 'percentual_hum', 'percentual_modelo', 'dif_abs']
    # cria as linhas ausentes na ordem da lista
    result = result.reindex(labels) 
    result['TAGs_hum'] = result.get('TAGs_hum', pd.Series(index=result.index, dtype=object))
    result['TAGs_hum'] = result['TAGs_hum'].fillna(result.index.to_series())
    result['TAGs_modelo'] = result.get('TAGs_modelo', pd.Series(index=result.index, dtype=object))
    result['TAGs_modelo'] = result['TAGs_modelo'].fillna(result.index.to_series())

    for c in cols_zero:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors='coerce').fillna(0)

    MAE = result['dif_abs'].sum() / len(result)

    df_metrics.loc[epoch, labels[0]] = result.loc[labels[0], 'dif_abs']
    df_metrics.loc[epoch, labels[1]] = result.loc[labels[1], 'dif_abs']
    df_metrics.loc[epoch, labels[2]] = result.loc[labels[2], 'dif_abs']
    df_metrics.loc[epoch, labels[3]] = result.loc[labels[3], 'dif_abs']
    df_metrics.loc[epoch, labels[4]] = result.loc[labels[4], 'dif_abs']
    df_metrics.loc[epoch, labels[5]] = result.loc[labels[5], 'dif_abs']
    df_metrics.loc[epoch, labels[6]] = result.loc[labels[6], 'dif_abs']
    df_metrics.loc[epoch, labels[7]] = result.loc[labels[7], 'dif_abs']
    df_metrics.loc[epoch, labels[8]] = result.loc[labels[8], 'dif_abs']
    df_metrics.loc[epoch, labels[9]] = result.loc[labels[9], 'dif_abs']
    df_metrics.loc[epoch, labels[10]] = result.loc[labels[10], 'dif_abs']
    df_metrics.loc[epoch, labels[11]] = result.loc[labels[11], 'dif_abs']
    df_metrics.loc[epoch, labels[12]] = result.loc[labels[12], 'dif_abs']
    df_metrics.loc[epoch, labels[13]] = result.loc[labels[13], 'dif_abs']
    df_metrics.loc[epoch, labels[14]] = result.loc[labels[14], 'dif_abs']
    df_metrics.loc[epoch, labels[15]] = result.loc[labels[15], 'dif_abs']
    df_metrics.loc[epoch, labels[16]] = result.loc[labels[16], 'dif_abs']
    df_metrics.loc[epoch, labels[17]] = result.loc[labels[17], 'dif_abs']
    df_metrics.loc[epoch, labels[18]] = result.loc[labels[18], 'dif_abs']
    df_metrics.loc[epoch, labels[19]] = result.loc[labels[19], 'dif_abs']
    df_metrics.loc[epoch, labels[20]] = result.loc[labels[20], 'dif_abs']
    df_metrics.loc[epoch, labels[21]] = result.loc[labels[21], 'dif_abs']
    df_metrics.loc[epoch, labels[22]] = result.loc[labels[22], 'dif_abs']
    df_metrics.loc[epoch, labels[23]] = result.loc[labels[23], 'dif_abs']
    df_metrics.loc[epoch, labels[24]] = result.loc[labels[24], 'dif_abs']
    df_metrics.loc[epoch, labels[25]] = result.loc[labels[25], 'dif_abs']
    df_metrics.loc[epoch, labels[26]] = result.loc[labels[26], 'dif_abs']
    df_metrics.loc[epoch, labels[27]] = result.loc[labels[27], 'dif_abs']
    df_metrics.loc[epoch, labels[28]] = result.loc[labels[28], 'dif_abs']

    # df_metrics.loc[epoch, labels[29]] = result.loc[labels[29], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[30]] = result.loc[labels[30], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[31]] = result.loc[labels[31], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[32]] = result.loc[labels[32], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[33]] = result.loc[labels[33], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[34]] = result.loc[labels[34], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[35]] = result.loc[labels[35], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[36]] = result.loc[labels[36], 'dif_abs']  # Detrator
    # df_metrics.loc[epoch, labels[37]] = result.loc[labels[37], 'dif_abs']  # Detrator

    # df_metrics.loc[epoch, labels[38]] = result.loc[labels[38], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[39]] = result.loc[labels[39], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[40]] = result.loc[labels[40], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[41]] = result.loc[labels[41], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[42]] = result.loc[labels[42], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[43]] = result.loc[labels[43], 'dif_abs']  # Neutro
    # df_metrics.loc[epoch, labels[44]] = result.loc[labels[44], 'dif_abs']  # Neutro
    
    df_metrics.loc[epoch, 'MAE'] = MAE
    print(f'Amostra {epoch+1} finalizada')
    print('-------------------------------\n')


df_metrics.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao_MAE\df_metrics.xlsx", index=False)
print('\nArquivo df_metrics.xlsx foi salvo com sucesso!')
