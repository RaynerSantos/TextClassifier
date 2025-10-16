import pandas as pd
import numpy as np
import pyodbc
import scipy.sparse as sp
from datetime import datetime, timedelta

data_atual = datetime.now().date()
# data_hora_formatada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
dia_anterior = data_atual - timedelta(days=1)


labels_promotor = ['Abrangência do plano / atendimento em outras cidades / abrangência nacional',
       'Agilidade / facilidade na autorização de exames e procedimentos',
       'Bom acompanhamento de doenças (ex: diabetes)', 'Cobertura do plano',
       'Conseguir marcar sem passar pelo clínico geral',
       'Desconto em medicamentos em algumas farmácias',
       'Facilidade / agilidade para consultas / exames / procedimentos',
       'Facilidade com a carteirinha digital', 'Facilidade de reembolso',
       'Marca conhecida / confiável',
       'Nunca teve problemas / não tem reclamações / está satisfeito',
       'Não sabe / Não respondeu', 'Preço do plano / custo benefício',
       'Programas interessantes (ex: Mais 60, curso para parar de fumar)',
       'Qualidade / agilidade do atendimento',
       'Qualidade / facilidade do aplicativo / site',
       'Qualidade de atendimento da Central Unimed',
       'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Qualidade do atendimento no hospital / pronto atendimento / pronto socorro / urgência',
       'Qualidade dos médicos credenciados',
       'Qualidade dos médicos especialistas',
       'Quantidade de médicos credenciados',
       'Quantidade de médicos especialistas',
       'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Rede / hospital próprio',
       'Telemedicina / consulta pelo aplicativo / whatsapp',
       'Ter atendimento 24h para urgência / emergência',
       'Ter rede credenciada perto de onde mora / o hospital que gosta',
       'Transparência']

labels_neutro = ['Abrangência do plano', 'Agenda disponível dos médicos credenciados',
       'Atendimento na rede credenciada (laboratórios / clínicas / hospitais)',
       'Atendimento no Pronto Socorro / Pronto Atendimento',
       'Aumentar cobertura do plano odontológico',
       'Bloqueio do plano com atraso do pagamento do boleto',
       'Cobertura do plano', 'Conseguir marcar pelo aplicativo',
       'Demora do extrato / boleto',
       'Demora na marcação de exames / consultas / procedimentos',
       'Demora no atendimento online / teleconsulta',
       'Descredenciamento / saída de médicos',
       'Descredenciamento de rede (laboratórios / clínicas / hospitais)',
       'Desmarcam consulta / procedimento sem comunicar',
       'Dificuldade com reembolso',
       'Dificuldade de atendimento sem estar com a carteirinha',
       'Dificuldade de conseguir atendimento em outras cidades mesmo o plano sendo nacional',
       'Dificuldade de incluir / excluir dependente',
       'Dificuldade para conseguir informações como valor da co-participação / falta de clareza nas cobranças',
       'Facilitar / agilizar a autorização de exames e procedimentos',
       'Falta de atendimento por teleconsula',
       'Guia médico / rede credenciada com informações desatualizadas / incorretas',
       'Incluir terapias', 'Inovação (novas tecnologias / equipamentos)',
       'Melhorar atendimento na rede própria',
       'Melhorar canal de atendimento (Whatsapp / Telefone / 0800 / chat)',
       'Médico solicitou valores por fora do plano',
       'Médicos credenciados que falam que só tem vaga no particular',
       'Não sabe / Não respondeu', 'Plantões aos finais de semana',
       'Prazo de carência',
       'Precisar passar em um clínico geral antes de ir ao especialista',
       'Preço do plano / da co-participação',
       'Problema com a carteirinha digital',
       'Problemas / dificuldades com o aplicativo / site',
       'Problemas com pagamento / cobrança indevida',
       'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Qualidade dos médicos credenciados',
       'Quantidade de médicos credenciados',
       'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Quantidade médicos especialistas credenciados / dificuldade de conseguir algumas especialidades',
       'Tentativa de cancelamento do plano de forma unilateral pela Unimed',
       'Ter mais rede credenciada perto de onde mora / o hospital que gosta',
       'Ter planos para pessoa física (sem precisar ser vinculado a uma empresa)',
       'Ter planos sem co-participação']


labels_detrator = ['A Unimed está com dificuldades financeiras',
       'Atendimento na rede credenciada (laboratórios / clínicas / hospitais)',
       'Canal de atendimento / suporte (Whatsapp / Telefone / chat)',
       'Cancelamento do plano de forma unilateral pela Unimed',
       'Cobertura do plano', 'Cobrança indevida',
       'Demora na marcação de exames / consultas / procedimentos',
       'Demora no atendimento de urgência / emergência',
       'Descredenciamento / saída de médicos',
       'Descredenciamento de rede (laboratórios / clínicas / hospitais)',
       'Dificuldade / demora na autorização de exames e procedimentos',
       'Dificuldade com intercâmbio / relacionamento com a Unimed local',
       'Dificuldade com reembolso', 'Dificuldade para cancelar o plano',
       'Dificuldade para conseguir informações sobre o plano',
       'Falta de transparência nos valores cobrados de cooparticipação',
       'Falta rede credenciada perto de onde mora / o hospital que gosta',
       'Guia médico / rede credenciada no aplicativo com informações desatualizadas / incorretas',
       'Médico solicitou valores por fora do plano para fazer cirurgia',
       'Não conseguir agendar pelo aplicativo',
       'Não recebeu/ teve problema com o cartão do plano',
       'Não sabe / Não respondeu', 'Prazo de carência',
       'Preço do plano / da co-participação',
       'Problema com abrangência / não consegue atendimento em outras localidades',
       'Problema para receber o informe de rendimentos',
       'Problemas com aplicativo / site',
       'Problemas para incluir / excluir dependente',
       'Qualidade da rede credenciada (laboratórios / clínicas / hospitais)',
       'Qualidade dos médicos credenciados',
       'Quantidade de médicos credenciados',
       'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Quantidade médicos especialistas credenciados / dificuldade de conseguir algumas especialidades',
       'Realizou o pagamento e a Unimed não deu baixa e teve problemas para usar o plano',
       'Rede credenciada muito cheia', 'Suspensão do plano pela Unimed',
       'Ter que passar pelo clínico geral antes de ir ao especialista',
       'Unimed desmarcou consulta / exame / procedimento']

class medidas_multilabel:
    def __init__(self, df, name_cols=['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']):
        self.df = df
        self.name_cols = name_cols

    def assertividade(self):
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
        self.df['qtd_acerto'] = self.df[['acerto_1', 'acerto_2', 'acerto_3']].sum(axis=1)
        self.df['recall'] = self.df['qtd_acerto'].divide(self.df['qtd_humano'])
        recall = self.df['recall'].mean() * 100
        return recall
    
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


# Conexão para criar o banco de dados 
Server='10.0.1.66'
Database='BI_DW_BETA'
Usuario='rayner.santos'
Senha='8J1"hP^Kgr}4'
banco = "teste_classificacao_unimed"

mapping = {
        'Promotor': ['Q2A_1', 'Q2A_2', 'Q2A_3'],
        'Neutro': ['Q2B_1', 'Q2B_2', 'Q2B_3'],
        'Detrator': ['Q2C_1', 'Q2C_2', 'Q2C_3']
    }

class ConexaoSQL:
    def __init__(self, Server, Database, Usuario, Senha, tabela, mapping):
        self.Server = Server
        self.Database = Database
        self.Usuario = Usuario
        self.Senha = Senha
        self.tabela = tabela
        self.mapping = mapping

    def conectar(self):
        dados_conexao = (
        "DRIVER=ODBC Driver 17 for SQL Server;"   # "Driver={SQL Server};"
        "Server="+self.Server+";"
        "Database="+self.Database+";"
        "UID="+self.Usuario+";"
        "PWD="+self.Senha+";"
        "Encrypt=yes;TrustServerCertificate=yes"
        )
        return pyodbc.connect(dados_conexao)

    
    #=== Buscar banco no SQL Server ===#
    def select_banco(self):
        conexao = self.conectar()
        print("Conexão bem sucedida para SELECT")

        # comando_sql = f"""SELECT * FROM {self.tabela}
        #                   WHERE data_final >= '{dia_anterior}'"""
        comando_sql = f"""SELECT * FROM {self.tabela}"""
        df_sql = pd.read_sql(comando_sql, conexao)
        conexao.close()
        # print(f'\ndf_sql:\n{df_sql}')
        return df_sql
    
    #=== Fazer UPDATE no banco no SQL Server ===#
    def update_banco(self, df_sql, df_classificacao, ID, categoria_NPS):        
        conexao = self.conectar()
        print("Conexão bem sucedida para UPDATE")
        # Crie um cursor
        cursor = conexao.cursor()

        def tratar_valor(valor):
            if pd.isna(valor):
                return None
            return str(valor).replace("'", "''")

        # Loop para atualização condicional
        for _, linha in df_classificacao.iterrows():
            codigo = linha[ID]
            classe = linha[categoria_NPS]

            for key, value in self.mapping.items():
                if (codigo in df_sql[ID].values) and (classe == key):
                    TAG_1 = tratar_valor(linha['TAG_1'])
                    TAG_2 = tratar_valor(linha['TAG_2'])
                    TAG_3 = tratar_valor(linha['TAG_3'])
                    valor_1 = value[0]
                    valor_2 = value[1]
                    valor_3 = value[2]

                    # Usado ? com parâmetros no execute para evitar SQL Injection e problemas de aspas.
                    comando_sql = f"""
                        UPDATE {self.tabela}
                        SET [{valor_1}] = ?, [{valor_2}] = ?, [{valor_3}] = ?
                        WHERE [{ID}] = ?
                    """
                    cursor.execute(comando_sql, TAG_1, TAG_2, TAG_3, codigo)

        # Confirma as alterações
        conexao.commit()
        print("\n✅ Classificações enviadas para o banco de dados no servidor com sucesso!\n")

        # Fecha a conexão
        cursor.close()
        conexao.close()

conexao = ConexaoSQL(Server=Server, Database=Database, Usuario=Usuario, Senha=Senha, tabela=banco, mapping=mapping)


def get_proba_matrix_ovr(model, X):
    """
    Retorna uma matriz (n_amostras, n_labels) com P(y=1) por label,
    funcionando tanto para OneVsRestClassifier do scikit-learn quanto do scikit-multilearn.
    """
    probas = model.predict_proba(X)

    # Caso 1: scikit-multilearn normalmente retorna matriz esparsa
    if sp.issparse(probas):
        return probas.toarray()

    # Caso 2: scikit-learn retorna ndarray 2D
    if isinstance(probas, np.ndarray):
        if probas.ndim == 2:
            return probas
        else:
            # raro, mas garante fallback
            return np.asarray(probas).reshape(len(X), -1)

    # Caso 3: (fallback) alguns pipelines podem devolver lista/iterável
    # com vetores linha; empilhe verticalmente
    try:
        arr = np.vstack([np.asarray(row).ravel() for row in probas])
        return arr
    except Exception:
        raise TypeError("Formato inesperado em predict_proba para OneVsRestClassifier.")