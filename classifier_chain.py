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
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression

# scikit-multilearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from skmultilearn.problem_transform import ClassifierChain
from lightgbm import LGBMClassifier
from scipy.sparse import issparse

MODELOS = ['Promotor', 'Neutro', 'Detrator']
df_resultados = []
df_frequencia_labels = []

for modelo in MODELOS:
    # --- 2. Carregar os Dados ---
    print("Carregando dados...")
    file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
    if modelo == "Promotor":
        sheet_name = "Promotor"
        sheet_name_base_teste = "Promotor_TESTE"
        ngram_range = (1, 4)
    elif modelo == "Neutro":
        sheet_name = "Neutro"
        sheet_name_base_teste = "Neutro_TESTE"
        ngram_range = (1, 4)
    elif modelo == "Detrator":
        sheet_name = "Detrator"
        sheet_name_base_teste = "Detrator_TESTE"
        ngram_range = (2, 4)

    pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name) 
    data = pre_processador.pre_processamento()
    data = data.loc[(data['ONDA'] != "Set/25")] 
    # data = data.loc[data["NPS"]=="Neutro"]

    # print('\ndataframe:\n', train_df[['resposta','Classificacao_humana_1','Classificacao_humana_2','Classificacao_humana_3']])
    print(f'\nShape do banco de dados completo: {data.shape}')

    tags_df = pre_processador.pre_processamento_tags()
    print(f'\nFrequência das TAGs:\n{tags_df}')
    print(f'\nQtd TAGs: {len(tags_df)}')
    tags_df['Categoria'] = modelo
    df_frequencia_labels.append(tags_df)

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
    data_multilabel['processed_text'] = [preprocess_text(text, modelo=modelo) for text in tqdm(data_multilabel['resposta'], desc="Pré-processando dos dados")]

    # train_df = data_multilabel.loc[(data_multilabel['NPS'] != sheet_name_base_teste) & (data_multilabel['ONDA'] != "Ago/25")] 
    # test_df_2 = data_multilabel.loc[(data_multilabel['ONDA'] == "Ago/25")] # Categoria NPS e de Agosto/25
    test_df = data_multilabel.loc[(data_multilabel['NPS'] == sheet_name_base_teste)] # Categoria NPS e de Julho/25
    train_df = data_multilabel.loc[(data_multilabel['NPS'] != sheet_name_base_teste)] 


    bancos_teste = [test_df, 
                    # test_df_2
                    ]

    print(f'\nShape de train_df:\t{train_df.shape}')
    print(f'\nShape de test_df:\t{test_df.shape}')
    # print(f'\nShape de test_df:\t{test_df_2.shape}')

    # --- 4. Vetorização de Texto (TF-IDF) ---
    print("\n--- Vetorizando textos com TF-IDF... ---")
    tfidf_vectorizer = TfidfVectorizer(
                    min_df=2, # só considera palavras/ngrams que aparecem em pelo menos 3 documentos - Se você tem 100 textos, e min_df=3, só serão consideradas as palavras (ou n-gramas) que aparecem em pelo menos 3 textos diferentes.
                    max_df=0.7,  # ignora palavras/ngrams que aparecem em mais de 70% dos documentos - Se você tem 100 textos, e max_df=0.8, as palavras que aparecem em mais de 70% dos textos serão ignoradas.
                    ngram_range=ngram_range, # usa palavras isoladas e pares de palavras, Ou seja, o modelo vai olhar para palavras sozinhas e para pares de palavras consecutivas. (1, 4)
                    sublinear_tf=True, # (TF logarítmico ajuda muito em LR)
                    analyzer='char_wb', # Some char n-grams (ex.: analyzer='char_wb', ngram_range=(3,5)); isso ajuda em erros ortográficos/ariações.
                    max_features=None, # limita o número de características (features) a 10000 - Se o vocabulário gerado for maior que 10000, só as 10000 palavras/ngrams mais frequentes serão mantidas.
                    strip_accents='unicode' # desconsidera acentos
                    )

    # tfidf_vectorizer.fit_transform(data_multilabel['processed_text'])
    X_train = tfidf_vectorizer.fit_transform(train_df['processed_text'])
    Y_train = train_df[labels].values

    # print(f"Shape de X_train (vetorizado): {X_train.shape}")
    # print(f"Shape de Y_train (rótulos): {Y_train.shape}")


    # --- 5. Modelagem e Treinamento (Classificação Multilabel) ---
    print("\n--- Treinando Modelos Multilabel ---\n")
    # 5.5. Regressão Logística
    # Modelo MultiOutput Regressão Logística
    print("\n--- Treinando Regressão Logística ---\n")

    base_lr = LogisticRegression(solver='liblinear', # liblinear | lbfgs
                                                        max_iter=3000,
                                                        C=2.0,  # 2.0
                                                        class_weight='balanced')
    # calibrated_lr = CalibratedClassifierCV(base_lr, method='isotonic', cv=3) # isotonic | sigmoid

    rl_multi = ClassifierChain(base_lr,
                               require_dense=[False, True])

    # classifier_cc = ClassifierChain(LGBMClassifier(
    #                     boosting_type="dart",     # "gbdt" |  "dart" p/ dropout (anti-overfit)
    #                     objective="binary",
    #                     n_estimators=200,        # com early stopping
    #                     learning_rate=0.05,
    #                     num_leaves=30,            # controle a complexidade da árvore (mais baixo melhor)
    #                     max_depth=-1,             # deixe livre; use num_leaves/min_data_in_leaf p/ domar
    #                     min_data_in_leaf=50,      # ↑ aumenta = modelo mais simples (bom p/ texto)
    #                     min_sum_hessian_in_leaf=1.0,
    #                     feature_fraction=0.3,     # amostra de colunas (forte em TF-IDF) baixo (0.2–0.5)
    #                     bagging_fraction=0.8,     # amostra de linhas (0.7–0.9)
    #                     bagging_freq=1,
    #                     reg_alpha=2.0,            # L1
    #                     reg_lambda=6.0,           # L2
    #                     n_jobs=-1,
    #                     random_state=42,
    #                     class_weight="balanced"
    #                 ), require_dense=[False, True]
    #             )

    rl_multi.fit(X_train, Y_train)
    
    for amostra_teste in bancos_teste:
        
        X_test = tfidf_vectorizer.transform(amostra_teste['processed_text'])

        # Probabilidades por label
        proba_raw = rl_multi.predict_proba(X_test)  # sua linha
        if issparse(proba_raw):
            proba_matrix = proba_raw.toarray()  # ou .A
        else:
            proba_matrix = np.asarray(proba_raw)

        print('shape de proba_matrix:\n', proba_matrix.shape)  # (n_amostras, n_classes)
        print('proba_matrix:\n', proba_matrix)

        # Para cada texto, pegar até 3 TAGs mais prováveis (maiores probabilidades)
        top_k = 3
        preds_topk = np.zeros_like(proba_matrix, dtype=int)
        for i, row in enumerate(proba_matrix):
            # indices das top 3 probabilidades
            top_indices = row.argsort()[-top_k:][::-1]
            # Marca como 1 as top TAGs
            preds_topk[i, top_indices] = 1

        # DataFrame de predições
        # result_df_ml = test_df.copy()
        result_df_ml = amostra_teste.copy()

        # (Opcional) Adicionar coluna com nomes das TAGs previstas (listas de TAGs)
        def tags_preditas(row):
            return [labels[i] for i, val in enumerate(row) if val == 1]

        # Para cada linha, pegue os índices das top 3 probabilidades
        THRESHOLD = 0.5  # Defina o limiar desejado 0.5
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

        colunas_para_juntar = ['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']
        # colunas_para_juntar = ['NET_1', 'NET_2', 'NET_3']
        result_df_ml['TAGs'] = result_df_ml[colunas_para_juntar].astype(str).agg(' '.join, axis=1)

        result_df_ml = remover_NsNr(dados=result_df_ml, label_NsNr='Não sabe / Não respondeu')

        medidas = medidas_multilabel(df=result_df_ml, name_cols=colunas_para_juntar)
        medidas.acrescentar_colunas()
        recall_sample = medidas.recall_sample()
        recall_micro = medidas.recall_micro()
        precision_sample = medidas.precision_sample()
        precision_micro = medidas.precision_micro()
        f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
        hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()
        
        print(f'\n----- Métricas do modelo {modelo} -----')
        # print('\nresultado:\n', result_df_ml[['TAG_1','TAG_2','TAG_3']].head(10))
        print('\nQuantidade de TAG_1 vazias: ', result_df_ml['TAG_1'].isna().sum())
        print('Quantidade de TAG_2 vazias: ', result_df_ml['TAG_2'].isna().sum())
        print('Quantidade de TAG_3 vazias: ', result_df_ml['TAG_3'].isna().sum())

        print(f'\nTHRESHOLD: {THRESHOLD}')
        print(f"Percentual de acerto geral (recall-micro):\t **{round(recall_micro, 1)}%**")
        print(f"Percentual de acerto geral sobre o modelo (precision-micro):\t **{round(precision_micro, 1)}%**")
        print(f"F1-micro:\t **{round(f1_micro, 1)}%**")
        print(f"Média geral do percentual de acerto por texto (recall-sample):\t **{round(recall_sample, 1)}%**")
        print(f"Média geral do percentual de acerto por texto sobre o modelo (precision-sample):\t **{round(precision_sample, 1)}%**")
        
        print(f'Hamming Loss manual: ', round(hamming_loss_manual,3))
        print(f'Quantidade de Falsos Positivos: ', fp)
        print(f'Quantidade de Falsos Negativos: ', fn, '\n')

        Qtd_tags_previstas = result_df_ml[['TAG_1']].value_counts().sum() + result_df_ml[['TAG_2']].value_counts().sum() + result_df_ml[['TAG_3']].value_counts().sum()
        print('Quantidade total de TAGs previstas (não nulas): ', Qtd_tags_previstas)
        Qtd_tags_humano = result_df_ml[['Classificacao_humana_1']].value_counts().sum() + result_df_ml[['Classificacao_humana_2']].value_counts().sum() + result_df_ml[['Classificacao_humana_3']].value_counts().sum()
        print('Quantidade total de TAGs da classificação humana (não nulas): ', Qtd_tags_humano)

        result_df_ml = result_df_ml.drop(columns=[c for c in labels if c in test_df.columns], errors='ignore')
        result_df_ml = result_df_ml.drop(columns=[c for c in result_df_ml.columns if 'TAGs' in c], errors='ignore')
        df_resultados.append(result_df_ml)
        print("\nTreinamento Regressão Logística concluído.")
        print("--- Processo Concluído ---")



# df_frequencia_labels = pd.concat(df_frequencia_labels, axis=0)  
# df_frequencia_labels.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\TAGs_Unimed.xlsx", index=False)
# print('\nArquivo exportado com sucesso!')
df_resultados = pd.concat(df_resultados, axis=0)
# df_resultados.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\RegLog.xlsx", index=False)
# print('\nArquivo exportado com sucesso!')

medidas = medidas_multilabel(df=df_resultados, name_cols=colunas_para_juntar)
medidas.acrescentar_colunas()
recall_sample = medidas.recall_sample()
recall_micro = medidas.recall_micro()
precision_sample = medidas.precision_sample()
precision_micro = medidas.precision_micro()
f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()

print('\n----- 📊 Métricas do resultado consolidado -----')
print('Quantidade de TAG_1 vazias: ', df_resultados['TAG_1'].isna().sum())
print('Quantidade de TAG_2 vazias: ', df_resultados['TAG_2'].isna().sum())
print('Quantidade de TAG_3 vazias: ', df_resultados['TAG_3'].isna().sum())

print(f'\nTHRESHOLD: {THRESHOLD}')
print(f"Percentual de acerto geral (recall-micro):\t **{round(recall_micro, 1)}%**")
print(f"Percentual de acerto geral sobre o modelo (precision-micro):\t **{round(precision_micro, 1)}%**")
print(f"F1-micro:\t **{round(f1_micro, 1)}%**")
print(f'Hamming Loss manual: ', round(hamming_loss_manual,3))

Qtd_tags_humano = df_resultados[['Classificacao_humana_1']].value_counts().sum() + df_resultados[['Classificacao_humana_2']].value_counts().sum() + df_resultados[['Classificacao_humana_3']].value_counts().sum()
print('Quantidade total de TAGs da classificação humana (não nulas): ', Qtd_tags_humano)
Qtd_tags_previstas = df_resultados[['TAG_1']].value_counts().sum() + df_resultados[['TAG_2']].value_counts().sum() + df_resultados[['TAG_3']].value_counts().sum()
print('Quantidade total de TAGs previstas (não nulas): ', Qtd_tags_previstas)
Qtd_tags_corretas_modelo = df_resultados[['acerto_1','acerto_2','acerto_3']].sum().sum()
print('Quantidade total de TAGs previstas corretamente pelo modelo: ', int(Qtd_tags_corretas_modelo), '\n')

