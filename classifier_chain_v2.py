import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

# Fonte
from app.src.preprocess_text import preprocess_text, prepare_multilabel_data, get_word2vec_features, Pre_Processamento, labels_em_cols, medidas_multilabel, extrair_entrevistado, remover_NsNr

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# scikit-multilearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain  # NOVO: Importar ClassifierChain


# =============================================================================
# CLASSE WRAPPER PARA LIDAR COM LABELS PROBLEMÁTICAS
# =============================================================================

class RobustLogisticRegression:
    """
    Wrapper para LogisticRegression que lida com labels que têm apenas uma classe.
    Se detectar apenas uma classe, usa DummyClassifier ao invés de LogisticRegression.
    """
    
    def __init__(self, solver='liblinear', max_iter=3000, C=2.0, class_weight='balanced'):
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self.model_ = None
        self.is_dummy_ = False
        
    def fit(self, X, y):
        """Treina o modelo, usando DummyClassifier se necessário."""
        # Verificar quantas classes únicas existem
        unique_classes = np.unique(y)
        
        if len(unique_classes) < 2:
            # Apenas uma classe presente - usar DummyClassifier
            self.is_dummy_ = True
            self.model_ = DummyClassifier(strategy='most_frequent')
            warnings.warn(f"Label com apenas uma classe ({unique_classes[0]}). Usando DummyClassifier.")
        else:
            # Duas ou mais classes - usar LogisticRegression normal
            self.is_dummy_ = False
            self.model_ = LogisticRegression(
                solver=self.solver,
                max_iter=self.max_iter,
                C=self.C,
                class_weight=self.class_weight
            )
        
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        """Faz predição."""
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Retorna probabilidades."""
        if self.is_dummy_:
            # DummyClassifier pode não ter predict_proba, então criar manualmente
            pred = self.model_.predict(X)
            n_samples = X.shape[0]
            # Retornar probabilidades "dummy" (0 ou 1)
            proba = np.zeros((n_samples, 2))
            proba[np.arange(n_samples), pred.astype(int)] = 1.0
            return proba
        else:
            return self.model_.predict_proba(X)
    
    def get_params(self, deep=True):
        """Retorna parâmetros (necessário para sklearn)."""
        return {
            'solver': self.solver,
            'max_iter': self.max_iter,
            'C': self.C,
            'class_weight': self.class_weight
        }
    
    def set_params(self, **params):
        """Define parâmetros (necessário para sklearn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# FUNÇÃO PARA TREINAR ENSEMBLE DE CLASSIFIER CHAINS
# =============================================================================

def train_classifier_chain_ensemble(X_train, Y_train, labels, n_chains=5, random_state=42, C=2.0, max_iter=3000):
    """
    Treina ensemble de Classifier Chains com diferentes ordens aleatórias.
    
    Args:
        X_train: Features de treino (matriz esparsa TF-IDF)
        Y_train: Labels de treino (n_samples, n_labels)
        labels: Lista de nomes das labels
        n_chains: Número de chains no ensemble
        random_state: Seed base para reprodutibilidade
        C: Parâmetro de regularização do LogisticRegression
        max_iter: Número máximo de iterações
    
    Returns:
        Lista de modelos ClassifierChain treinados
    """
    print(f"\n--- Treinando Ensemble de {n_chains} Classifier Chains ---")
    
    # Verificar labels problemáticas antes de treinar
    print("\n🔍 Verificando distribuição das labels...")
    problematic_labels = []
    
    for i, label in enumerate(labels):
        unique_classes = np.unique(Y_train[:, i])
        n_positive = np.sum(Y_train[:, i] == 1)
        n_negative = np.sum(Y_train[:, i] == 0)
        
        if len(unique_classes) < 2:
            problematic_labels.append({
                'label': label,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'only_class': unique_classes[0]
            })
    
    if problematic_labels:
        print(f"\n⚠️  Encontradas {len(problematic_labels)} labels problemáticas (apenas uma classe):")
        for info in problematic_labels[:5]:  # Mostrar apenas as primeiras 5
            print(f"   - {info['label']}: {info['n_positive']} positivos, {info['n_negative']} negativos")
        if len(problematic_labels) > 5:
            print(f"   ... e mais {len(problematic_labels) - 5} labels")
        print("   Essas labels usarão DummyClassifier automaticamente.\n")
    else:
        print("✅ Todas as labels têm pelo menos 2 classes. Prosseguindo...\n")
    
    chains = []
    
    for i in range(n_chains):
        print(f"Treinando Chain {i+1}/{n_chains}...")
        
        # Modelo base ROBUSTO
        base_lr = RobustLogisticRegression(
            solver='liblinear',
            max_iter=max_iter,
            C=C,
            class_weight='balanced'
        )
        
        # Criar ClassifierChain com ordem aleatória
        chain = ClassifierChain(
            base_lr,
            order='random',
            random_state=random_state + i,
            verbose=False
        )
        
        # Treinar com tratamento de erros
        try:
            chain.fit(X_train, Y_train)
            chains.append(chain)
            print(f"   ✅ Chain {i+1} treinado com sucesso!")
        except Exception as e:
            print(f"   ❌ Erro ao treinar Chain {i+1}: {e}")
            print(f"   Pulando este chain...")
            continue
    
    if len(chains) == 0:
        raise ValueError("Nenhum chain foi treinado com sucesso. Verifique seus dados.")
    
    print(f"\n✅ Ensemble de {len(chains)} chains treinado com sucesso!")
    return chains


# =============================================================================
# FUNÇÃO PARA PREDIÇÃO COM ENSEMBLE DE CHAINS
# =============================================================================

def predict_with_chain_ensemble(chains, X_test, labels, threshold=0.5, top_k=3, aggregation='mean'):
    """
    Faz predição usando ensemble de Classifier Chains.
    
    Args:
        chains: Lista de modelos ClassifierChain treinados
        X_test: Features de teste (matriz esparsa TF-IDF)
        labels: Lista de nomes das labels
        threshold: Threshold para binarização das probabilidades
        top_k: Número máximo de labels por amostra
        aggregation: Método de agregação ('mean', 'voting', 'max')
    
    Returns:
        tags_preditas: Lista de listas com as tags preditas
        proba_matrix: Matriz de probabilidades agregadas (n_samples, n_labels)
    """
    print(f"\n--- Fazendo predições com ensemble de {len(chains)} chains ---")
    print(f"Método de agregação: {aggregation}")
    print(f"Threshold: {threshold}")
    
    # Coletar predições de todos os chains
    all_predictions = []
    
    for i, chain in enumerate(chains):
        print(f"Predizendo com Chain {i+1}/{len(chains)}...")
        try:
            pred = chain.predict(X_test)
            all_predictions.append(pred)
        except Exception as e:
            print(f"   ⚠️  Erro ao predizer com Chain {i+1}: {e}")
            print(f"   Pulando este chain...")
            continue
    
    if len(all_predictions) == 0:
        raise ValueError("Nenhum chain conseguiu fazer predições. Verifique seus dados.")
    
    # Converter para array numpy: (n_chains, n_samples, n_labels)
    all_predictions = np.array(all_predictions)
    
    # Agregar predições
    if aggregation == 'mean':
        # Média das predições (probabilidade estimada)
        proba_matrix = all_predictions.mean(axis=0)
    elif aggregation == 'voting':
        # Votação majoritária (proporção de chains que predizem 1)
        proba_matrix = all_predictions.mean(axis=0)
    elif aggregation == 'max':
        # Máximo (se pelo menos um chain prediz 1 com alta confiança)
        proba_matrix = all_predictions.max(axis=0)
    else:
        raise ValueError("aggregation deve ser 'mean', 'voting' ou 'max'")
    
    # Selecionar top-k com threshold
    tags_preditas = []
    
    for row in proba_matrix:
        # Pegar índices das top-k probabilidades
        top_indices = row.argsort()[-top_k:][::-1]
        top_tags = []
        
        for idx in top_indices:
            # Só incluir se >= threshold
            if row[idx] >= threshold:
                top_tags.append(labels[idx])
            else:
                top_tags.append(None)
        
        tags_preditas.append(top_tags)
    
    return tags_preditas, proba_matrix


# =============================================================================
# CÓDIGO PRINCIPAL
# =============================================================================

MODELOS = ['Promotor', 'Neutro', 'Detrator']
RECALL_MICRO_LISTA = []
PRECISION_MICRO_LISTA = []
F1_MICRO_LISTA = []
HL = []
df_resultados = []

# CONFIGURAÇÕES DO CLASSIFIER CHAIN
N_CHAINS = 5  # Número de chains no ensemble (recomendado: 3-7)
THRESHOLD = 0.5  # Threshold para binarização
AGGREGATION_METHOD = 'mean'  # 'mean', 'voting', ou 'max'

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

    # tags_df = pre_processador.pre_processamento_tags()
    # print(f'\nFrequência das TAGs:\n{tags_df}')
    # print(f'\nQtd TAGs: {len(tags_df)}')

    data_multilabel = prepare_multilabel_data(data)
    # print('\ntrain_df após o prepare_multilabel:\n', train_df)
    print('\nColunas/Labels:\n', data_multilabel.columns[1:])

    # labels = tags_df['TAGs'].str.lower()
    labels = list(data_multilabel.columns[1:])  # Converter para lista
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


    # =============================================================================
    # NOVA SEÇÃO: TREINAR ENSEMBLE DE CLASSIFIER CHAINS
    # =============================================================================
    
    print("\n" + "="*80)
    print("TREINANDO CLASSIFIER CHAIN ENSEMBLE")
    print("="*80)
    
    # Treinar ensemble de chains
    chains = train_classifier_chain_ensemble(
        X_train=X_train,
        Y_train=Y_train,
        labels=labels,
        n_chains=N_CHAINS,
        random_state=42,
        C=2.0,
        max_iter=3000
    )
    
    # Salvar informações sobre o ensemble
    ensemble_info = {
        'Modelo': modelo,
        'N_Chains': len(chains),
        'Threshold': THRESHOLD,
        'Aggregation': AGGREGATION_METHOD,
        'N_Labels': len(labels),
        'N_Train_Samples': X_train.shape[0]
    }
    
    print(f"\n📊 Informações do Ensemble:")
    for key, value in ensemble_info.items():
        print(f"   {key}: {value}")
    
    # =============================================================================
    # PREDIÇÃO COM CLASSIFIER CHAIN ENSEMBLE
    # =============================================================================
    
    for amostra_teste in bancos_teste:
        
        X_test = tfidf_vectorizer.transform(amostra_teste['processed_text'])

        # USAR A NOVA FUNÇÃO DE PREDIÇÃO COM ENSEMBLE DE CHAINS
        tags_preditas, proba_matrix = predict_with_chain_ensemble(
            chains=chains,
            X_test=X_test,
            labels=labels,
            threshold=THRESHOLD,
            top_k=3,
            aggregation=AGGREGATION_METHOD
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
        
        print(f'\n----- Métricas do modelo {modelo} (CLASSIFIER CHAIN ENSEMBLE) -----')
        # print('\nresultado:\n', result_df_ml[['TAG_1','TAG_2','TAG_3']].head(10))
        print('\nQuantidade de TAG_1 vazias: ', result_df_ml['TAG_1'].isna().sum())
        print('Quantidade de TAG_2 vazias: ', result_df_ml['TAG_2'].isna().sum())
        print('Quantidade de TAG_3 vazias: ', result_df_ml['TAG_3'].isna().sum())

        print(f'\nMÉTODO: Classifier Chain Ensemble ({len(chains)} chains, agregação: {AGGREGATION_METHOD})')
        print(f'THRESHOLD: {THRESHOLD}')
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

        RECALL_MICRO_LISTA.append(recall_micro)
        PRECISION_MICRO_LISTA.append(precision_micro)
        F1_MICRO_LISTA.append(f1_micro)
        HL.append(hamming_loss_manual * 100)

        # Exporta
        result_df_ml = result_df_ml.drop(columns=[c for c in labels if c in test_df.columns], errors='ignore')
        result_df_ml = result_df_ml.drop(columns=[c for c in result_df_ml.columns if 'TAGs' in c], errors='ignore')
        df_resultados.append(result_df_ml)
        print("\nTreinamento Classifier Chain Ensemble concluído.")
        print("--- Processo Concluído ---")


    
df_resultados = pd.concat(df_resultados, axis=0)
# df_resultados.to_excel(r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\ClassifierChain.xlsx", index=False)
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
print('----- MÉTRICAS DO RESULTADO CONSOLIDADO (CLASSIFIER CHAIN ENSEMBLE) -----')
print('='*80)
print('Quantidade de TAG_1 vazias: ', df_resultados['TAG_1'].isna().sum())
print('Quantidade de TAG_2 vazias: ', df_resultados['TAG_2'].isna().sum())
print('Quantidade de TAG_3 vazias: ', df_resultados['TAG_3'].isna().sum())

print(f'\nMÉTODO: Classifier Chain Ensemble ({len(chains)} chains, agregação: {AGGREGATION_METHOD})')
print(f'THRESHOLD: {THRESHOLD}')
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