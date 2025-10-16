import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Fonte
from app.src.preprocess_text import preprocess_text, prepare_multilabel_data, get_word2vec_features, Pre_Processamento, labels_em_cols, medidas_multilabel, extrair_entrevistado, remover_NsNr

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# scikit-multilearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# --- 2. Carregar os Dados ---
print("Carregando dados...")
file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
sheet_name = "Promotor"

pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name) 
data = pre_processador.pre_processamento()
data = data.loc[(data['ONDA'] != "Set/25")] 

print(f'\nShape do banco de dados completo: {data.shape}')


train_df = prepare_multilabel_data(data)
print('\nColunas/Labels:\n', train_df.columns[1:])

labels = train_df.columns[1:]
print(f'\nQuantidade de labels: {len(labels)}')

train_df = pd.concat(
    objs=[data[['ID_UNICO','ONDA','NPS','Classificacao_humana_1','Classificacao_humana_2','Classificacao_humana_3']].reset_index(drop=True),
          train_df.reset_index(drop=True)],
    axis=1
)

# --- 3. Pré-processamento de Texto ---

print("\n--- Pré-processando textos ---")
train_df['resposta'] = train_df['resposta'].fillna('')
# Usar list comprehension com tqdm para barras de progresso em scripts
train_df['processed_text'] = [preprocess_text(text) for text in tqdm(train_df['resposta'], desc="Pré-processando dos dados")]


# 2. Crie o pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LogisticRegression(
                                                    #  solver='liblinear', 
                                                     max_iter=1000)))
])

# 3. Defina os hiperparâmetros para grid search
param_grid = {
    'tfidf__min_df': [2, 3, 5],
    'tfidf__max_df': [0.5, 0.6, 0.7],
    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (2, 4)],
    # 'tfidf__max_features': [50000, None],
    'clf__estimator__C': [1.0, 2.0, 3.0],  # regularização
    'clf__estimator__class_weight': ['balanced'], # None
    'clf__estimator__solver': ['liblinear', 'lbfgs']
}

recall_samples_scorer = make_scorer(recall_score, average='samples')
cv = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# 4. Inicie o GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1_micro',  # para multilabel, pode usar também 'f1_micro' / 'f1_macro' / 'f1_samples'
    cv=cv,
    verbose=2,
    n_jobs=-1
)

# 5. Treine
grid.fit(train_df['processed_text'], train_df[labels])

# 6. Veja os melhores parâmetros e resultados
print("Melhores hiperparâmetros:", grid.best_params_)
print("Melhor score:", grid.best_score_)

print("\n--- Processo Concluído ---")

"""
    Promotor:
        Melhores hiperparâmetros: {'clf__estimator__C': 3.0, 'clf__estimator__class_weight': 'balanced', 'clf__estimator__solver': 'lbfgs', 'tfidf__max_df': 0.5, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}
        Melhor score: 0.6084382395537946
"""