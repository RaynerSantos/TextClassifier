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
from sklearn.multioutput import ClassifierChain


# =============================================================================
# FUNÇÃO PARA OTIMIZAR THRESHOLD POR LABEL
# =============================================================================

def optimize_thresholds_per_label(model, X_train, Y_train, labels, metric='f1'):
    """
    Otimiza threshold individualmente para cada label baseado em F1-score, Precision ou Recall.
    
    Args:
        model: Modelo OneVsRestClassifier treinado
        X_train: Features de treino (matriz esparsa TF-IDF)
        Y_train: Labels de treino (n_samples, n_labels)
        labels: Lista de nomes das labels
        metric: 'f1', 'precision', ou 'recall' - métrica para otimização
    
    Returns:
        Dicionário com threshold ótimo para cada label
    """
    print(f"\n--- Otimizando thresholds por label (métrica: {metric}) ---")
    optimal_thresholds = {}
    
    # Obter probabilidades no conjunto de treino
    print("Calculando probabilidades no conjunto de treino...")
    proba_train = np.zeros((X_train.shape[0], len(labels)))
    
    for i, estimator in enumerate(tqdm(model.estimators_, desc="Extraindo probabilidades")):
        proba = estimator.predict_proba(X_train)
        # Verificar se retorna probabilidades para ambas as classes
        if proba.ndim == 2 and proba.shape[1] > 1:
            proba_train[:, i] = proba[:, 1]
        else:
            proba_train[:, i] = proba.ravel()
    
    # Para cada label, encontrar threshold ótimo
    print("Otimizando threshold para cada label...")
    for i, label in enumerate(tqdm(labels, desc="Otimizando labels")):
        y_true_label = Y_train[:, i]
        y_proba_label = proba_train[:, i]
        
        # Verificar se há exemplos positivos
        if y_true_label.sum() == 0:
            print(f"  ⚠️  Label '{label}' não possui exemplos positivos no treino. Usando threshold padrão 0.5")
            optimal_thresholds[label] = 0.5
            continue
        
        # Calcular curva precision-recall
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true_label, y_proba_label)
        except Exception as e:
            print(f"  ⚠️  Erro ao calcular curva para label '{label}': {e}. Usando threshold padrão 0.5")
            optimal_thresholds[label] = 0.5
            continue
        
        # Calcular métrica desejada para cada threshold
        if metric == 'f1':
            # F1-score
            scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        elif metric == 'precision':
            scores = precisions
        elif metric == 'recall':
            scores = recalls
        else:
            raise ValueError("metric deve ser 'f1', 'precision' ou 'recall'")
        
        # Encontrar threshold ótimo (excluir último elemento que é sempre threshold=1)
        if len(thresholds) > 0:
            optimal_idx = np.argmax(scores[:-1])
            optimal_thresholds[label] = float(thresholds[optimal_idx])
        else:
            optimal_thresholds[label] = 0.5  # Fallback
    
    # Estatísticas dos thresholds
    thresholds_values = list(optimal_thresholds.values())
    print(f"\n📊 Estatísticas dos thresholds otimizados:")
    print(f"   Mínimo: {np.min(thresholds_values):.3f}")
    print(f"   Máximo: {np.max(thresholds_values):.3f}")
    print(f"   Média: {np.mean(thresholds_values):.3f}")
    print(f"   Mediana: {np.median(thresholds_values):.3f}")
    
    return optimal_thresholds


# =============================================================================
# FUNÇÃO PARA APLICAR THRESHOLD POR LABEL NA PREDIÇÃO
# =============================================================================

def predict_with_per_label_threshold(model, X_test, labels, optimal_thresholds, top_k=3):
    """
    Faz predição aplicando threshold específico para cada label.
    
    Args:
        model: Modelo OneVsRestClassifier treinado
        X_test: Features de teste (matriz esparsa TF-IDF)
        labels: Lista de nomes das labels
        optimal_thresholds: Dicionário com threshold para cada label
        top_k: Número máximo de labels por amostra
    
    Returns:
        tags_preditas: Lista de listas com as tags preditas
        proba_matrix: Matriz de probabilidades (n_samples, n_labels)
    """
    # Obter probabilidades
    preds_ml = [estimator.predict_proba(X_test) for estimator in model.estimators_]
    probs_pos = []
    
    for p in preds_ml:
        if p.ndim == 2 and p.shape[1] > 1:
            probs_pos.append(p[:, 1])
        else:
            probs_pos.append(p.ravel())
    
    proba_matrix = np.vstack(probs_pos).T  # (n_amostras, n_classes)
    
    # Aplicar threshold específico por label
    predictions_binary = np.zeros_like(proba_matrix, dtype=int)
    
    for i, label in enumerate(labels):
        threshold = optimal_thresholds.get(label, 0.5)
        predictions_binary[:, i] = (proba_matrix[:, i] >= threshold).astype(int)
    
    # Selecionar top-k das predições válidas
    tags_preditas = []
    
    for i, row in enumerate(proba_matrix):
        # Índices onde predição é 1 (passou no threshold específico)
        valid_indices = np.where(predictions_binary[i] == 1)[0]
        
        if len(valid_indices) == 0:
            # Nenhuma label passou no threshold
            tags_preditas.append([None] * top_k)
            continue
        
        # Ordenar válidos por probabilidade (decrescente)
        sorted_valid = sorted(valid_indices, key=lambda idx: row[idx], reverse=True)
        
        # Pegar top-k
        top_tags = []
        for idx in sorted_valid[:top_k]:
            top_tags.append(labels[idx])
        
        # Preencher com None se necessário
        while len(top_tags) < top_k:
            top_tags.append(None)
        
        tags_preditas.append(top_tags)
    
    return tags_preditas, proba_matrix


# =============================================================================
# CÓDIGO PRINCIPAL
# =============================================================================

MODELOS = ['Promotor', 'Neutro', 'Detrator']
df_resultados = []

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
    calibrated_lr = CalibratedClassifierCV(base_lr, method='isotonic', cv=3) # isotonic | sigmoid
    rl_multi = OneVsRestClassifier(calibrated_lr)

    rl_multi.fit(X_train, Y_train)
    
    # =============================================================================
    # NOVA SEÇÃO: OTIMIZAR THRESHOLDS POR LABEL
    # =============================================================================
    
    print("\n" + "="*80)
    print("OTIMIZANDO THRESHOLDS ESPECÍFICOS POR LABEL")
    print("="*80)
    
    # Otimizar thresholds usando F1-score como métrica
    # Você pode mudar para 'precision' se quiser focar mais em reduzir falsos positivos
    # ou 'recall' se quiser focar em capturar mais verdadeiros positivos
    optimal_thresholds = optimize_thresholds_per_label(
        model=rl_multi,
        X_train=X_train,
        Y_train=Y_train,
        labels=labels,
        metric='f1'  # Opções: 'f1', 'precision', 'recall'
    )
    
    # Salvar thresholds otimizados em arquivo para referência
    thresholds_df = pd.DataFrame([
        {'Label': label, 'Threshold_Otimizado': threshold}  
        for label, threshold in optimal_thresholds.items()
    ])
    thresholds_df = thresholds_df.sort_values('Threshold_Otimizado', ascending=False)
    
    # Salvar em arquivo Excel
    output_path = rf"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Thresholds_Otimizados_{modelo}.xlsx"
    thresholds_df.to_excel(output_path, index=False)
    print(f"\n✅ Thresholds otimizados salvos em: {output_path}")
    
    # Mostrar alguns exemplos
    print(f"\n📋 Exemplos de thresholds otimizados:")
    print(thresholds_df.head(10).to_string(index=False))
    
    # =============================================================================
    # PREDIÇÃO COM THRESHOLDS OTIMIZADOS
    # =============================================================================
    
    for amostra_teste in bancos_teste:
        
        X_test = tfidf_vectorizer.transform(amostra_teste['processed_text'])

        # USAR A NOVA FUNÇÃO DE PREDIÇÃO COM THRESHOLD POR LABEL
        print("\n--- Aplicando predição com thresholds específicos por label ---")
        tags_preditas, proba_matrix = predict_with_per_label_threshold(
            model=rl_multi,
            X_test=X_test,
            labels=labels,
            optimal_thresholds=optimal_thresholds,
            top_k=3
        )

        # DataFrame de predições
        result_df_ml = amostra_teste.copy()

        # Criar as colunas TAG_1, TAG_2, TAG_3
        tag_columns = [f"TAG_{i+1}" for i in range(3)]
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
        
        print(f'\n----- Métricas do modelo {modelo} (COM THRESHOLD POR LABEL) -----')
        # print('\nresultado:\n', result_df_ml[['TAG_1','TAG_2','TAG_3']].head(10))
        print('\nQuantidade de TAG_1 vazias: ', result_df_ml['TAG_1'].isna().sum())
        print('Quantidade de TAG_2 vazias: ', result_df_ml['TAG_2'].isna().sum())
        print('Quantidade de TAG_3 vazias: ', result_df_ml['TAG_3'].isna().sum())

        print(f'\nMÉTODO: Threshold Específico por Label (otimizado por F1)')
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


    
df_resultados = pd.concat(df_resultados, axis=0)
# df_resultados.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\RegLog_ThresholdPorLabel.xlsx", index=False)
# print('\nArquivo exportado com sucesso!')

medidas = medidas_multilabel(df=df_resultados, name_cols=colunas_para_juntar)
medidas.acrescentar_colunas()
recall_sample = medidas.recall_sample()
recall_micro = medidas.recall_micro()
precision_sample = medidas.precision_sample()
precision_micro = medidas.precision_micro()
f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()

print('\n' + '='*80)
print('----- MÉTRICAS DO RESULTADO CONSOLIDADO (THRESHOLD POR LABEL) -----')
print('='*80)
print('Quantidade de TAG_1 vazias: ', df_resultados['TAG_1'].isna().sum())
print('Quantidade de TAG_2 vazias: ', df_resultados['TAG_2'].isna().sum())
print('Quantidade de TAG_3 vazias: ', df_resultados['TAG_3'].isna().sum())

print(f'\nMÉTODO: Threshold Específico por Label (otimizado por F1)')
print(f"Percentual de acerto geral (recall-micro):\t **{round(recall_micro, 1)}%**")
print(f"Percentual de acerto geral sobre o modelo (precision-micro):\t **{round(precision_micro, 1)}%**")
print(f"F1-micro:\t **{round(f1_micro, 1)}%**")
print(f'Hamming Loss manual: ', round(hamming_loss_manual,3))

Qtd_tags_humano = df_resultados[['Classificacao_humana_1']].value_counts().sum() + df_resultados[['Classificacao_humana_2']].value_counts().sum() + df_resultados[['Classificacao_humana_3']].value_counts().sum()
print('Quantidade total de TAGs da classificação humana (não nulas): ', Qtd_tags_humano)
Qtd_tags_previstas = df_resultados[['TAG_1']].value_counts().sum() + df_resultados[['TAG_2']].value_counts().sum() + df_resultados[['TAG_3']].value_counts().sum()
print('Quantidade total de TAGs previstas (não nulas): ', Qtd_tags_previstas)
Qtd_tags_corretas_modelo = df_resultados[['acerto_1','acerto_2','acerto_3']].sum().sum()
print('Quantidade total de TAGs previstas corretamente pelo modelo: ', int(Qtd_tags_corretas_modelo))

print('\n' + '='*80)
print('✅ PROCESSO COMPLETO FINALIZADO COM SUCESSO!')
print('='*80)

