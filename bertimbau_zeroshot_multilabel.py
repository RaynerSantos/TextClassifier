"""
Classificação Multilabel Zero-Shot com BERTimbau
=================================================

Este código utiliza BERTimbau (BERT treinado em português) para classificação
multilabel zero-shot, onde o modelo decide quais labels se aplicam a cada texto
sem necessidade de treinamento específico.

Instalação necessária:
pip install transformers torch pandas numpy tqdm openpyxl sentencepiece protobuf

IMPORTANTE: Execute o comando de instalação acima antes de rodar este código!
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Verificar dependências
print("🔍 Verificando dependências...")
try:
    from transformers import pipeline
    print("   ✅ transformers instalado")
except ImportError:
    print("   ❌ transformers não instalado. Execute: pip install transformers")
    exit(1)

try:
    import sentencepiece
    print("   ✅ sentencepiece instalado")
except ImportError:
    print("   ❌ sentencepiece não instalado. Execute: pip install sentencepiece")
    exit(1)

try:
    import torch
    print("   ✅ torch instalado")
except ImportError:
    print("   ❌ torch não instalado. Execute: pip install torch")
    exit(1)

# Fonte
from app.src.preprocess_text import (
    preprocess_text, prepare_multilabel_data, Pre_Processamento, 
    medidas_multilabel, remover_NsNr
)


# =============================================================================
# CLASSE PARA ZERO-SHOT CLASSIFICATION COM BERTIMBAU
# =============================================================================

class BERTimbauZeroShot:
    """
    Classificador zero-shot multilabel usando modelo multilíngue.
    """
    
    def __init__(self, model_name='joeddav/xlm-roberta-large-xnli', device=None): # neuralmind/bert-large-portuguese-cased
        """
        Inicializa o modelo para zero-shot.
        
        Args:
            model_name: Nome do modelo no HuggingFace
            device: 'cuda' para GPU, 'cpu' para CPU, None para auto-detect
        """
        print(f"\n🤖 Inicializando Zero-Shot Classification...")
        print(f"   Modelo: {model_name}")
        
        # Detectar dispositivo
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
        
        device_name = 'GPU (CUDA)' if self.device == 0 else 'CPU'
        print(f"   Dispositivo: {device_name}")
        
        if self.device == -1:
            print("   ⚠️  Usando CPU - Será MUITO lento. Recomenda-se usar GPU.")
        
        # Criar pipeline de zero-shot
        print("   Carregando modelo e tokenizer...")
        print("   (Primeira vez pode demorar - download de ~1.5GB)")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.device,
                use_fast=True  # Usar tokenizer rápido
            )
            print("   ✅ Modelo carregado com sucesso!\n")
        except Exception as e:
            print(f"   ❌ Erro ao carregar modelo: {e}")
            print("\n   Tentando com tokenizer lento...")
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=self.device,
                    use_fast=False  # Fallback para tokenizer lento
                )
                print("   ✅ Modelo carregado com tokenizer lento!\n")
            except Exception as e2:
                print(f"   ❌ Erro fatal: {e2}")
                print("\n   Verifique se todas as dependências estão instaladas:")
                print("   pip install transformers torch sentencepiece protobuf")
                raise
    
    def predict_single(self, text, candidate_labels, threshold=0.5, top_k=3):
        """
        Prediz labels para um único texto.
        
        Args:
            text: Texto a ser classificado
            candidate_labels: Lista de labels candidatas
            threshold: Threshold mínimo de confiança
            top_k: Número máximo de labels a retornar
        
        Returns:
            labels_preditas: Lista de labels preditas
            scores: Lista de scores correspondentes
        """
        # Validar entrada
        if not text or len(text.strip()) == 0:
            return [None] * top_k, [0.0] * top_k
        
        # Truncar texto se muito longo (limite do modelo: 512 tokens)
        if len(text) > 2000:  # ~512 tokens
            text = text[:2000]
        
        try:
            # Fazer predição zero-shot
            result = self.classifier(
                text,
                candidate_labels,
                multi_label=True  # Importante: permite múltiplas labels
            )
            
            # Filtrar por threshold e top-k
            labels_preditas = []
            scores = []
            
            for label, score in zip(result['labels'], result['scores']):
                if score >= threshold and len(labels_preditas) < top_k:
                    labels_preditas.append(label)
                    scores.append(score)
            
            # Preencher com None se necessário
            while len(labels_preditas) < top_k:
                labels_preditas.append(None)
                scores.append(0.0)
            
            return labels_preditas, scores
        
        except Exception as e:
            print(f"   ⚠️  Erro ao processar texto: {e}")
            return [None] * top_k, [0.0] * top_k
    
    def predict_batch(self, texts, candidate_labels, threshold=0.5, top_k=3):
        """
        Prediz labels para múltiplos textos.
        
        Args:
            texts: Lista de textos
            candidate_labels: Lista de labels candidatas
            threshold: Threshold mínimo de confiança
            top_k: Número máximo de labels por texto
        
        Returns:
            all_predictions: Lista de listas com labels preditas
            all_scores: Lista de listas com scores
        """
        print(f"\n🔮 Fazendo predições zero-shot para {len(texts)} textos...")
        print(f"   Threshold: {threshold}")
        print(f"   Top-K: {top_k}")
        print(f"   Número de labels candidatas: {len(candidate_labels)}")
        
        all_predictions = []
        all_scores = []
        
        for text in tqdm(texts, desc="Classificando textos"):
            labels_pred, scores = self.predict_single(
                text, 
                candidate_labels, 
                threshold=threshold, 
                top_k=top_k
            )
            all_predictions.append(labels_pred)
            all_scores.append(scores)
        
        return all_predictions, all_scores


# =============================================================================
# FUNÇÃO PARA CRIAR DESCRIÇÕES DAS LABELS (IMPORTANTE!)
# =============================================================================

def create_label_descriptions(labels):
    """
    OPÇÃO 3: Template de frases completas.
    
    Use esta opção para máxima clareza e naturalidade.
    """
    
    # Templates de frases
    templates = {
        # Formato: 'Label': 'Frase completa descrevendo quando esta label se aplica'
        'Utilizar quando: O cliente relata que consegue agendar, realizar ou acessar consultas, exames ou procedimentos de maneira rápida e sem burocracia': 'Facilidade / agilidade para consultas / exames / procedimentos',
        'Utilizar quando: O comentário destaca que o atendimento recebido (presencial, telefônico ou digital) é eficiente, cordial e/ou atende as expectativas em termos de rapidez e tratamento': 'Qualidade / agilidade do atendimento',
        'Utilizar quando: O cliente menciona positivamente o número ou variedade de opções disponíveis na rede credenciada, como muitos laboratórios, clínicas ou hospitais': 'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O feedback ressalta que nunca enfrentou dificuldades, não possui motivos de reclamação ou expressa satisfação geral com o serviço': 'Nunca teve problemas / não tem reclamações / está satisfeito',
        'Utilizar quando: O usuário destaca que processos de autorização para exames ou procedimentos são rápidos, simples ou sem obstáculos': 'Agilidade / facilidade na autorização de exames e procedimentos',
        'Utilizar quando: A resposta elogia diretamente os médicos da rede credenciada, mencionando conhecimento, profissionalismo ou bom atendimento': 'Qualidade dos médicos credenciados',
        'Utilizar quando: O feedback enfatiza experiências positivas com atendimento em hospitais, prontos-socorros, pronto atendimento, rede credenciada ou situações de urgência/emergência': 'Qualidade do atendimento no hospital / pronto atendimento / pronto socorro / urgência / rede credenciada',
        'Utilizar quando: O cliente menciona a grande quantidade ou diversidade de médicos disponíveis pela rede credenciada': 'Quantidade de médicos credenciados',
        'Utilizar quando: A resposta elogia a excelência, qualidade, reputação ou estrutura dos laboratórios, clínicas ou hospitais que fazem parte da rede credenciada': 'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O cliente valoriza poder usar o plano em diferentes regiões, cidades ou em todo o território nacional': 'Abrangência do plano / atendimento em outras cidades / abrangência nacional',
        'Utilizar quando: O usuário comenta que o plano cobre uma ampla gama de procedimentos, exames, especialidades ou tratamentos': 'Cobertura do plano',
        'Utilizar quando: É citada positivamente a proximidade da rede credenciada com a residência do cliente ou a presença de unidades/hospitais preferidos': 'Ter rede credenciada perto de onde mora / o hospital que gosta',
        'Utilizar quando: O cliente realça a oferta variada ou abundante de médicos especialistas (cardiologista, endocrinologista, ortopedista, neuropediatra, peadiatra, pneumologista, entre outros) disponíveis na rede': 'Quantidade de médicos especialistas',
        'Utilizar quando: A resposta destaca positivamente o aplicativo ou site da operadora, mencionando facilidade de uso, agilidade ou bons recursos': 'Qualidade / facilidade do aplicativo / site',
        'Utilizar quando: O participante não fornece um motivo específico, menciona somente aspectos negativos sendo promotor, menciona algo diícil de compreender ou deixa em branco a justificativa, mesmo que promotor': 'Não sabe / Não respondeu',
        'Utilizar quando: O cliente enaltece o valor que paga pelo plano, ou avalia como justo, acessível ou com ótimo custo-benefício': 'Preço do plano / custo benefício',
        'Utilizar quando: Há elogios ao serviço de telemedicina ou possibilidade de consultas via aplicativo, WhatsApp ou meios digitais': 'Telemedicina / consulta pelo aplicativo / whatsapp',
        'Utilizar quando: O feedback expressa satisfação com o atendimento oferecido pela central telefônica, presencial ou canais oficiais da Unimed': 'Qualidade de atendimento da Central Unimed',
        'Utilizar quando: O cliente valoriza o atendimento de especialistas, elogiando sua competência, atenção ou resolutividade': 'Qualidade dos médicos especialistas',
        'Utilizar quando: O cliente menciona positivamente a existência, facilidade ou qualidade de hospitais próprios da operadora': 'Rede / hospital próprio',
        'Utilizar quando: O usuário destaca a praticidade, rapidez ou clareza no processo de solicitação e recebimento de reembolsos': 'Facilidade de reembolso',
        'Utilizar quando: O motivo do promotor é a reputação, tradição ou confiança na marca da operadora': 'Marca conhecida / confiável',
        'Utilizar quando: O cliente enaltece a clareza das informações, ausência de surpresas ou comunicação transparente da empresa': 'Transparência',
        'Utilizar quando: A resposta destaca positivamente o uso da carteirinha digital, mencionando praticidade ou facilidade de acesso': 'Facilidade com a carteirinha digital',
        'Utilizar quando: O cliente aprecia poder agendar consultas diretamente com especialistas, sem a necessidade de encaminhamento do clínico geral': 'Conseguir marcar sem passar pelo clínico geral',
        'Utilizar quando: O feedback valoriza o acompanhamento, suporte ou acompanhamento efetivo de doenças crônicas ou condições específicas': 'Bom acompanhamento de doenças (ex: diabetes)',
        'Utilizar quando: O cliente elogia o benefício de desconto em medicamentos oferecido pelo plano, citando uso em farmácias conveniadas': 'Desconto em medicamentos em algumas farmácias',
        'Utilizar quando: O cliente se mostra satisfeito com a disponibilidade de atendimento para urgências/emergências em qualquer horário, inclusive 24h': 'Ter atendimento 24h para urgência / emergência',
        'Utilizar quando: O usuário comenta positivamente sobre programas de prevenção, promoção de saúde ou iniciativas diferenciadas oferecidas pela operadora': 'Programas interessantes (ex: Mais 60, curso para parar de fumar)',

        'Utilizar quando: O cliente diz que gostaria de ter mais opções de locais credenciados para exames, consultas ou procedimentos, sugerindo ampliar a rede para atendimento mais acessível e confortável': 'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O feedback aponta que o tempo para agendamento é maior do que o desejado, mencionando espera para conseguir marcar exames, consultas ou procedimentos, e sugere maior agilidade': 'Demora na marcação de exames / consultas / procedimentos',
        'Utilizar quando: O cliente sinaliza insatisfação pela saída ou descredenciamento de estabelecimentos da rede, sugerindo manter ou aumentar a quantidade de locais disponíveis': 'Descredenciamento de rede (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O usuário pede o aumento do número de médicos credenciados na rede, para ampliar as opções de atendimento sem especificar nenhuma especialidade': 'Quantidade de médicos credenciados',
        'Utilizar quando: O cliente expressa o desejo de mais especialistas disponíveis ou aponta dificuldade em encontrar certas especialidades na rede, sugerindo ampliação desse quadro. Considerar médico especialista cardiologista, pneumologista, endocrinologista, pedriatra, ortopedista, ginecolosgista entre outros': 'Quantidade médicos especialistas credenciados / dificuldade de conseguir algumas especialidades',
        'Utilizar quando: O feedback recomenda que o processo de autorização de exames e procedimentos seja mais simples e rápido, citando burocracia ou atrasos': 'Facilitar / agilizar a autorização de exames e procedimentos',
        'Utilizar quando: O cliente sugere ajustes no preço do plano ou na co-participação, buscando valores mais acessíveis ou melhor relação custo-benefício': 'Preço do plano / da co-participação',
        'Utilizar quando: O comentário sugere manter médicos na rede, reclamando da saída de profissionais e da dificuldade para encontrar substitutos': 'Descredenciamento / saída de médicos',
        'Utilizar quando: O cliente sugere que o plano deveria incluir mais serviços, procedimentos ou tratamentos na cobertura': 'Cobertura do plano',
        'Utilizar quando: O usuário pede que existam mais opções próximas à sua residência ou na região de interesse, citando localização como ponto de melhoria': 'Ter mais rede credenciada perto de onde mora / o hospital que gosta',
        'Utilizar quando: O feedback sugere melhoria no atendimento realizado por esses parceiros, indicando que a experiência poderia ser mais eficiente ou cordial': 'Atendimento na rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O cliente não especificou um motivo para sua avaliação, omitindo sugestões de melhoria ou especificou algo que não foi possível compreender': 'Não sabe / Não respondeu',
        'Utilizar quando: O usuário sugere aprimorar o processo de reembolso, mencionando lentidão, dificuldade de ser aprovado ou excesso de burocracia': 'Dificuldade com reembolso',
        'Utilizar quando: O cliente aponta dificuldades técnicas ou limitações na plataforma digital para uso do plano, e sugere solução de bugs ou melhoria de usabilidade': 'Problemas / dificuldades com o aplicativo / site',
        'Utilizar quando: O cliente pede que sejam selecionados ou mantidos profissionais com maior qualificação, mencionando experiências ou qualidade do atendimento aquém do esperado': 'Qualidade dos médicos credenciados',
        'Utilizar quando: O usuário reclama que o guia médico ou a lista de credenciados está desatualizada ou contém erros, e sugere atualização frequente': 'Guia médico / rede credenciada com informações desatualizadas / incorretas',
        'Utilizar quando: O cliente recomenda aprimorar atendimento via canais digitais ou telefônicos, citando falta de eficiência, demora ou falhas de comunicação': 'Melhorar canal de atendimento (Whatsapp / Telefone / 0800 / chat)',
        'Utilizar quando: O usuário sugere melhorar a qualidade dos prestadores da rede credenciada, buscando locais melhor equipados, modernos ou com atendimento superior': 'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O cliente pede por experiência mais ágil, atenciosa ou humananos Pronto Socorros e Pronto Atendimentos, mencionando eventuais deficiências': 'Atendimento no Pronto Socorro / Pronto Atendimento',
        'Utilizar quando: O feedback pede maior transparência sobre valores cobrados, co-participação ou regras relacionadas, reportando falta de clareza ou dificuldades para conseguir a informação': 'Dificuldade para conseguir informações como valor da co-participação / falta de clareza nas cobranças',
        'Utilizar quando: O usuário reclama que médicos credenciados priorizam atendimento particular, dificultando ou demorando para atendimento pelo plano': 'Médicos credenciados que falam que só tem vaga no particular',
        'Utilizar quando: O cliente sugere expandir o alcance do plano para outras cidades ou estados': 'Abrangência do plano',
        'Utilizar quando: O feedback pede melhora na disponibilidade de horários dos médicos credenciados, facilitando marcações': 'Agenda disponível dos médicos credenciados',
        'Utilizar quando: O cliente aponta que não consegue atendimento fora da cidade de origem, sugerindo efetivação da abrangência nacional prometida': 'Dificuldade de conseguir atendimento em outras cidades mesmo o plano sendo nacional',
        'Utilizar quando: O usuário sugere reduzir o tempo de carência antes de acessar certos serviços': 'Prazo de carência',
        'Utilizar quando: O cliente sugere flexibilizar ou eliminar a obrigatoriedade do encaminhamento do clínico geral para especialistas': 'Precisar passar em um clínico geral antes de ir ao especialista',
        'Utilizar quando: O usuário sugere aprimorar ou permitir o agendamento via aplicativo, tornando-o mais rápido, fácil e estável': 'Conseguir marcar pelo aplicativo',
        'Utilizar quando: O feedback reporta dificuldades ou erros recorrentes no processo de pagamento ou identificação de cobranças indevidas': 'Problemas com pagamento / cobrança indevida',
        'Utilizar quando: O cliente relata cobrança de valores extras, não cobertos pelo plano, por médicos da rede': 'Médico solicitou valores por fora do plano',
        'Utilizar quando: O usuário diz que consultas ou procedimentos foram desmarcados sem aviso prévio, sugerindo maior organização e comunicação': 'Desmarcam consulta / procedimento sem comunicar',
        'Utilizar quando: O feedback cobra a inclusão ou melhoria do serviço de atendimento por teleconsulta': 'Falta de atendimento por teleconsula',
        'Utilizar quando: O cliente relata dificuldade de ser atendido sem portar a carteirinha, sugerindo alternativas': 'Dificuldade de atendimento sem estar com a carteirinha',
        'Utilizar quando: O usuário aponta atrasos no envio de extratos, boletos ou comprovantes, dificultando a gestão financeira': 'Demora do extrato / boleto',
        'Utilizar quando: O cliente relata instabilidades ou falhas na carteirinha digital e sugere sua correção': 'Problema com a carteirinha digital',
        'Utilizar quando: O usuário enfrenta dificuldades para alterar ou gerenciar dependentes e pede um processo mais simples': 'Dificuldade de incluir / excluir dependente',
        'Utilizar quando: O cliente sugere a ampliação da cobertura para terapias (como fisioterapia, psicologia, etc.)': 'Incluir terapias',
        'Utilizar quando: O feedback relata que o atendimento digital poderia ser mais rápido': 'Demora no atendimento online / teleconsulta',
        'Utilizar quando: O cliente pede por mais opções ou maior disponibilidade de atendimento em plantões de finais de semana': 'Plantões aos finais de semana',
        'Utilizar quando: O usuário fala da necessidade de melhorar o atendimento nos hospitais ou clínicas próprias da operadora': 'Melhorar atendimento na rede própria',
        'Utilizar quando: O cliente sugere investimento em tecnologia, novos equipamentos ou soluções inovadoras para o atendimento': 'Inovação (novas tecnologias / equipamentos)', 
        'Utilizar quando: O usuário propõe flexibilização ou negociação para evitar suspensão imediata em caso de atraso de pagamento': 'Bloqueio do plano com atraso do pagamento do boleto',
        'Utilizar quando: O feedback relata tentativas de cancelamento do plano sem seu interesse, sugerindo revisão desses procedimentos': 'Tentativa de cancelamento do plano de forma unilateral pela Unimed',
        'Utilizar quando: O usuário sugere que o plano odontológico inclua mais serviços': 'Aumentar cobertura do plano odontológico',
        'Utilizar quando: O cliente pede opções de contratação do plano para pessoa física, sem necessidade de vínculo empresarial': 'Ter planos para pessoa física (sem precisar ser vinculado a uma empresa)',
        'Utilizar quando: O feedback sugere criação ou ampliação de alternativas de planos sem cobrança de co-participação': 'Ter planos sem co-participação',   

        'Utilizar quando: O cliente expressa insatisfação pelo número limitado de laboratórios, clínicas ou hospitais credenciados, destacando dificuldade de acesso ou poucas opções para atendimento': 'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O usuário relata frustração com a lentidão para agendar exames, consultas ou procedimentos, indicando que a espera prejudicou sua experiência ou atendimento': 'Demora na marcação de exames / consultas / procedimentos',
        'Utilizar quando: O cliente destaca que a quantidade de médicos disponíveis é insuficiente, levando a dificuldades para encontrar profissionais ou marcar consultas': 'Quantidade de médicos credenciados',
        'Utilizar quando: O usuário reclama de estabelecimentos que foram descredenciados, causando perda de acesso a serviços importantes ou impactando negativamente sua região': 'Descredenciamento de rede (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O cliente sinaliza que há poucos especialistas ou dificuldade relevante para acessar certos tipos de atendimento, afetando o tratamento necessário. Considerar especialista médicos como cardiologista, penumologista, endocrinologista, pediatra, ortopedista, neuro, ginecologista entre outros': 'Quantidade médicos especialistas credenciados / dificuldade de conseguir algumas especialidades',
        'Utilizar quando: O cliente considera o valor do plano ou da co-participação abusivo, injusto ou desproporcional à qualidade do serviço oferecido ou que os reajustes foram excessivos': 'Preço do plano / da co-participação',
        'Utilizar quando: Há reclamação sobre médicos que deixaram de integrar a rede, resultando em queda da qualidade ou em rupturas frequentes nos atendimentos': 'Descredenciamento / saída de médicos',
        'Utilizar quando: O usuário relata que o plano não cobre procedimentos, exames, terapias ou doenças que considerava essenciais, causando insatisfação relevante': 'Cobertura do plano',
        'Utilizar quando: O cliente manifesta insatisfação pela demora, burocracia ou negação de autorizações para exames ou procedimentos médicos importantes': 'Dificuldade / demora na autorização de exames e procedimentos',
        'Utilizar quando: Há queixa sobre o atendimento, conhecimento técnico ou postura dos médicos da rede credenciada, apontando deficiência grave': 'Qualidade dos médicos credenciados',
        'Utilizar quando: O cliente reclama que não há unidades, hospitais ou clínicas próximas ou de sua preferência, dificultando o acesso ao serviço': 'Falta rede credenciada perto de onde mora / o hospital que gosta',
        'Utilizar quando: O participante não justifica claramente a nota ou não especifica o problema, mas demonstra insatisfação geral, ou menciona algo difícil de compreender ou positivo': 'Não sabe / Não respondeu',
        'Utilizar quando: O usuário relata processos lentos, confusos ou falhos na solicitação e recebimento de reembolso, frequentemente gerando prejuízos': 'Dificuldade com reembolso',
        'Utilizar quando: O cliente relata experiências negativas com o atendimento de parceiros credenciados, citando falta de qualidade, respeito ou eficiência': 'Atendimento na rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O feedback denuncia erros ou desatualizações graves no guia médico ou listas do aplicativo, que resultam em perda de tempo ou marcações equivocadas': 'Guia médico / rede credenciada no aplicativo com informações desatualizadas / incorretas',
        'Utilizar quando: O usuário acredita que laboratórios, clínicas e hospitais credenciados prestam atendimento precário, possuindo infraestrutura insuficiente ou inadequada': 'Qualidade da rede credenciada (laboratórios / clínicas / hospitais)',
        'Utilizar quando: O cliente relata falhas, demora ou inadequação nos canais de suporte, tornando difícil obter esclarecimentos ou resolver demandas': 'Canal de atendimento / suporte (Whatsapp / Telefone / chat)',
        'Utilizar quando: O usuário encontra instabilidades, falta de funcionalidades ou dificuldades graves no aplicativo ou site para acesso ao plano': 'Problemas com aplicativo / site',
        'Utilizar quando: O cliente descreve que estabelecimentos credenciados estão frequentemente lotados, resultando em demora no atendimento e desconforto': 'Rede credenciada muito cheia',
        'Utilizar quando: O feedback denuncia atrasos ou tempo elevado para receber atendimento em situações urgentes nos Pronto Antedimentos ou Pronto Socorro': 'Demora no atendimento de urgência / emergência',
        'Utilizar quando: O usuário não consegue acessar informações claras sobre regras, cobertura, procedimentos ou custos, prejudicando sua experiência': 'Dificuldade para conseguir informações sobre o plano',
        'Utilizar quando: O cliente reclama de prazos elevados de carência para uso completo do serviço, dificultando o acesso a atendimento essencial': 'Prazo de carência',
        'Utilizar quando: Há relato de cancelamento inesperado do plano por decisão da operadora, gerando insegurança e prejuízo ao beneficiário': 'Cancelamento do plano de forma unilateral pela Unimed',
        'Utilizar quando: O cliente não consegue realizar agendamentos via aplicativo por falhas técnicas, indisponibilidade do serviço ou limitações recorrentes': 'Não conseguir agendar pelo aplicativo',
        'Utilizar quando: O usuário efetuou o pagamento, mas a operadora não reconheceu, impedindo o acesso ou utilização dos serviços': 'Realizou o pagamento e a Unimed não deu baixa e teve problemas para usar o plano',
        'Utilizar quando: O cliente relata obstáculos ou demoras no processo de gestão de dependentes (incluir ou excluir usuário no plano)': 'Problemas para incluir / excluir dependente',
        'Utilizar quando: A operadora cancelou sem aviso a consulta, exame ou procedimento': 'Unimed desmarcou consulta / exame / procedimento',
        'Utilizar quando: Médicos credenciados que exigem pagamentos além do plano para procedimentos importantes': 'Médico solicitou valores por fora do plano para fazer cirurgia',
        'Utilizar quando: O usuário encontrou barreiras ou problemas para ser atendido por unidades Unimed fora de sua região de contratação': 'Dificuldade com intercâmbio / relacionamento com a Unimed local',
        'Utilizar quando: O cliente relata não ter recebido ou ter encontrado dificuldades funcionais com o cartão do plano de saúde': 'Não recebeu/ teve problema com o cartão do plano',
        'Utilizar quando: O usuário relata que seu plano foi suspenso pela operadora, seja por inadimplência, erro administrativo ou outro motivo, gerando prejuízo importante': 'Suspensão do plano pela Unimed',
        'Utilizar quando: O cliente manifesta insatisfação ao ter que agendar obrigatoriamente com clínico geral, dificultando acesso rápido a especialistas': 'Ter que passar pelo clínico geral antes de ir ao especialista',
        'Utilizar quando: O usuário aponta instabilidade financeira na operadora, gerando preocupação quanto à continuidade e qualidade do serviço ofertado': 'A Unimed está com dificuldades financeiras',
        'Utilizar quando: O cliente relata que, apesar da promessa de atendimento nacional, não consegue ser atendido fora da região de origem': 'Problema com abrangência / não consegue atendimento em outras localidades',
        'Utilizar quando: O usuário enfrenta barreiras excessivas, demora ou burocracia ao tentar cancelar seu plano de saúde': 'Dificuldade para cancelar o plano',
        'Utilizar quando: O cliente indica ter sido cobrado por valores que não correspondem ao serviço contratado ou previstos em contrato, causando insatisfação relevante': 'Cobrança indevida',
        'Utilizar quando: O usuário encontra dificuldade ou atrasos para acessar o documento de informe de rendimentos': 'Problema para receber o informe de rendimentos',
        'Utilizar quando: O cliente relata que a operadora não esclarece os valores de cooparticipação cobrados, gerando desconfiança e insatisfação': 'Falta de transparência nos valores cobrados de cooparticipação' 
    }
    
    descriptions = {}
    for label in labels:
        descriptions[label] = templates.get(label, label)
    
    return descriptions


# =============================================================================
# CÓDIGO PRINCIPAL
# =============================================================================

def main():
    """Função principal para executar o pipeline completo."""
    
    MODELOS = ['Promotor', 'Neutro', 'Detrator']
    df_resultados = []
    
    # CONFIGURAÇÕES DO ZERO-SHOT
    THRESHOLD = 0.7  # Threshold para considerar uma label (0.0 a 1.0)
    TOP_K = 3  # Número máximo de labels por texto
    
    # Inicializar modelo zero-shot (uma vez, fora do loop)
    print("\n" + "="*80)
    print("INICIALIZANDO ZERO-SHOT CLASSIFICATION")
    print("="*80)
    
    try:
        zero_shot_model = BERTimbauZeroShot()
    except Exception as e:
        print(f"\n❌ Erro ao inicializar modelo: {e}")
        print("\nVerifique se instalou todas as dependências:")
        print("pip install transformers torch sentencepiece protobuf")
        return
    
    for modelo in MODELOS:
        print("\n" + "="*80)
        print(f"PROCESSANDO MODELO: {modelo}")
        print("="*80)
        
        # --- 2. Carregar os Dados ---
        print("\n📂 Carregando dados...")
        file_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\BD_CNU_NPS_Rayner.xlsx"
        
        if modelo == "Promotor":
            sheet_name = "Promotor"
            sheet_name_base_teste = "Promotor_TESTE"
        elif modelo == "Neutro":
            sheet_name = "Neutro"
            sheet_name_base_teste = "Neutro_TESTE"
        elif modelo == "Detrator":
            sheet_name = "Detrator"
            sheet_name_base_teste = "Detrator_TESTE"
        
        pre_processador = Pre_Processamento(file_path=file_path, sheet_name=sheet_name)
        data = pre_processador.pre_processamento()
        data = data.loc[(data['ONDA'] != "Set/25")]
        
        print(f'   Shape do banco de dados completo: {data.shape}')
        
        data_multilabel = prepare_multilabel_data(data)
        labels = list(data_multilabel.columns[1:])
        print(f'   Quantidade de labels: {len(labels)}')
        
        data_multilabel = pd.concat(
            objs=[
                data[['ID_UNICO', 'ONDA', 'NPS', 'Classificacao_humana_1', 
                      'Classificacao_humana_2', 'Classificacao_humana_3']].reset_index(drop=True),
                data_multilabel.reset_index(drop=True)
            ],
            axis=1
        )
        
        # --- 3. Preparar Dados de Teste ---
        print("\n📋 Preparando dados de teste...")
        data_multilabel['resposta'] = data_multilabel['resposta'].fillna('')
        
        # Para zero-shot, NÃO precisamos de pré-processamento pesado
        # O modelo funciona melhor com texto natural
        test_df = data_multilabel.loc[(data_multilabel['NPS'] == sheet_name_base_teste)]
        
        print(f'   Shape de test_df: {test_df.shape}')
        
        # --- 4. Criar Descrições das Labels ---
        print("\n📝 Criando descrições das labels...")
        label_descriptions = create_label_descriptions(labels)
        
        # Usar descrições como candidate labels
        candidate_labels = list(label_descriptions.values())
        
        print(f"   Exemplos de labels candidatas:")
        for i, (label, desc) in enumerate(list(label_descriptions.items())[:5]):
            print(f"   - {label}: {desc}")
        if len(labels) > 5:
            print(f"   ... e mais {len(labels) - 5} labels")
        
        # --- 5. Fazer Predições Zero-Shot ---
        print("\n" + "="*80)
        print("FAZENDO PREDIÇÕES ZERO-SHOT")
        print("="*80)
        
        # Obter textos
        texts = test_df['resposta'].tolist()
        
        # Predizer
        predictions, scores = zero_shot_model.predict_batch(
            texts=texts,
            candidate_labels=candidate_labels,
            threshold=THRESHOLD,
            top_k=TOP_K
        )
        
        # --- 6. Processar Resultados ---
        print("\n📊 Processando resultados...")
        result_df = test_df.copy()
        
        # Criar colunas TAG_1, TAG_2, TAG_3
        result_df['TAG_1'] = [pred[0] for pred in predictions]
        result_df['TAG_2'] = [pred[1] for pred in predictions]
        result_df['TAG_3'] = [pred[2] for pred in predictions]
        
        # Criar colunas de scores
        result_df['SCORE_1'] = [score[0] for score in scores]
        result_df['SCORE_2'] = [score[1] for score in scores]
        result_df['SCORE_3'] = [score[2] for score in scores]
        
        # Remover NsNr
        colunas_para_juntar = ['Classificacao_humana_1', 'Classificacao_humana_2', 'Classificacao_humana_3']
        result_df = remover_NsNr(dados=result_df, label_NsNr='Não sabe / Não respondeu')
        
        # --- 7. Calcular Métricas ---
        print("\n📈 Calculando métricas...")
        medidas = medidas_multilabel(df=result_df, name_cols=colunas_para_juntar)
        medidas.acrescentar_colunas()
        
        recall_sample = medidas.recall_sample()
        recall_micro = medidas.recall_micro()
        precision_sample = medidas.precision_sample()
        precision_micro = medidas.precision_micro()
        f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
        hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()
        
        # --- 8. Mostrar Resultados ---
        print(f'\n{"="*80}')
        print(f'MÉTRICAS DO MODELO {modelo} (ZERO-SHOT)')
        print(f'{"="*80}')
        
        print('\n📋 Estatísticas de predições:')
        print(f'   Quantidade de TAG_1 vazias: {result_df["TAG_1"].isna().sum()}')
        print(f'   Quantidade de TAG_2 vazias: {result_df["TAG_2"].isna().sum()}')
        print(f'   Quantidade de TAG_3 vazias: {result_df["TAG_3"].isna().sum()}')
        
        print(f'\n⚙️  Configurações:')
        print(f'   THRESHOLD: {THRESHOLD}')
        print(f'   TOP_K: {TOP_K}')
        print(f'   Modelo: XLM-RoBERTa Zero-Shot')
        
        print(f'\n📊 Métricas de Performance:')
        print(f'   Percentual de acerto geral (recall-micro): **{round(recall_micro, 1)}%**')
        print(f'   Percentual de acerto sobre o modelo (precision-micro): **{round(precision_micro, 1)}%**')
        print(f'   F1-micro: **{round(f1_micro, 1)}%**')
        print(f'   Média de acerto por texto (recall-sample): **{round(recall_sample, 1)}%**')
        print(f'   Média de acerto por texto sobre o modelo (precision-sample): **{round(precision_sample, 1)}%**')
        
        print(f'\n🎯 Erros:')
        print(f'   Hamming Loss: {round(hamming_loss_manual, 3)}')
        print(f'   Falsos Positivos: {fp}')
        print(f'   Falsos Negativos: {fn}')
        
        Qtd_tags_previstas = (result_df[['TAG_1']].value_counts().sum() + 
                             result_df[['TAG_2']].value_counts().sum() + 
                             result_df[['TAG_3']].value_counts().sum())
        Qtd_tags_humano = (result_df[['Classificacao_humana_1']].value_counts().sum() + 
                          result_df[['Classificacao_humana_2']].value_counts().sum() + 
                          result_df[['Classificacao_humana_3']].value_counts().sum())
        
        print(f'\n📈 Totais:')
        print(f'   TAGs previstas (não nulas): {Qtd_tags_previstas}')
        print(f'   TAGs classificação humana (não nulas): {Qtd_tags_humano}')
        
        # Limpar colunas antes de adicionar aos resultados
        result_df = result_df.drop(columns=[c for c in labels if c in result_df.columns], errors='ignore')
        
        df_resultados.append(result_df)
        
        print(f"\n✅ Modelo {modelo} processado com sucesso!")
    
    # --- 9. Consolidar Resultados ---
    print("\n" + "="*80)
    print("CONSOLIDANDO RESULTADOS")
    print("="*80)
    
    df_resultados = pd.concat(df_resultados, axis=0)
    
    # Salvar resultados
    output_path = r"C:\PROJETOS\Classificacao de Textos\Dados\Unimed\Predicao\ZeroShot_Results.xlsx"
    df_resultados.to_excel(output_path, index=False)
    print(f'\n💾 Resultados salvos em: {output_path}')
    
    # Métricas consolidadas
    medidas = medidas_multilabel(df=df_resultados, name_cols=colunas_para_juntar)
    medidas.acrescentar_colunas()
    
    recall_sample = medidas.recall_sample()
    recall_micro = medidas.recall_micro()
    precision_sample = medidas.precision_sample()
    precision_micro = medidas.precision_micro()
    f1_micro = medidas.f1_micro(precision_micro=precision_micro, recall_micro=recall_micro)
    hamming_loss_manual, fp, fn = medidas.hamming_loss_manual()
    
    print('\n' + '='*80)
    print('MÉTRICAS CONSOLIDADAS (TODOS OS MODELOS)')
    print('='*80)
    
    print(f'\n📊 Performance Geral:')
    print(f'   Recall-micro: **{round(recall_micro, 1)}%**')
    print(f'   Precision-micro: **{round(precision_micro, 1)}%**')
    print(f'   F1-micro: **{round(f1_micro, 1)}%**')
    print(f'   Hamming Loss: {round(hamming_loss_manual, 3)}')
    
    Qtd_tags_previstas = (df_resultados[['TAG_1']].value_counts().sum() + 
                         df_resultados[['TAG_2']].value_counts().sum() + 
                         df_resultados[['TAG_3']].value_counts().sum())
    Qtd_tags_humano = (df_resultados[['Classificacao_humana_1']].value_counts().sum() + 
                      df_resultados[['Classificacao_humana_2']].value_counts().sum() + 
                      df_resultados[['Classificacao_humana_3']].value_counts().sum())
    Qtd_tags_corretas = df_resultados[['acerto_1', 'acerto_2', 'acerto_3']].sum().sum()
    
    print(f'\n📈 Totais Consolidados:')
    print(f'   TAGs previstas: {Qtd_tags_previstas}')
    print(f'   TAGs classificação humana: {Qtd_tags_humano}')
    print(f'   TAGs corretas: {int(Qtd_tags_corretas)}')
    
    print('\n' + '='*80)
    print('✅ PROCESSO COMPLETO FINALIZADO COM SUCESSO!')
    print('='*80)


if __name__ == "__main__":
    main()
