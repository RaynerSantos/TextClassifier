"""
Template para Criar Descrições Personalizadas de Labels
========================================================

Este arquivo fornece um template para você personalizar as descrições
das labels para uso com BERTimbau Zero-Shot.

IMPORTANTE: Descrições melhores = Resultados melhores!

Instruções:
1. Copie este código para o arquivo bertimbau_zeroshot_multilabel.py
2. Substitua a função create_label_descriptions() existente
3. Preencha o dicionário manual_mapping com suas labels reais
4. Execute o código e veja a melhoria nos resultados!
"""

def create_label_descriptions(labels):
    """
    Cria descrições em linguagem natural para as labels.
    
    IMPORTANTE: Descrições melhores = Melhores resultados!
    
    Você pode personalizar as descrições para suas labels específicas.
    
    Args:
        labels: Lista de nomes das labels
    
    Returns:
        Dicionário {label: descrição}
    """
    # Descrições padrão (você deve personalizar!)
    descriptions = {}
    
    for label in labels:
        # Estratégia padrão: Usar o nome da label como descrição
        # RECOMENDADO: Personalizar com descrições mais detalhadas
        descriptions[label] = label
        
        # Exemplo de personalização (descomente e adapte):
        # label_lower = label.lower()
        # if 'atendimento' in label_lower:
        #     descriptions[label] = f"Este texto fala sobre atendimento ao cliente"
        # elif 'preço' in label_lower:
        #     descriptions[label] = f"Este texto menciona preços ou custos"
        # elif 'qualidade' in label_lower:
        #     descriptions[label] = f"Este texto avalia qualidade de produtos ou serviços"
        # else:
        #     descriptions[label] = label
    
    return descriptions


def create_label_descriptions_TEMPLATE(labels):
    """
    Template para criar descrições personalizadas das labels.
    
    COPIE ESTE CÓDIGO e substitua a função create_label_descriptions()
    no arquivo bertimbau_zeroshot_multilabel.py
    """
    
    # ==========================================================================
    # OPÇÃO 1: MAPEAMENTO MANUAL COMPLETO (MELHOR RESULTADO!)
    # ==========================================================================
    
    manual_mapping = {
        # Exemplo de como preencher:
        # 'Nome_da_Label': 'Descrição em linguagem natural',
        
        # EXEMPLOS (substitua com suas labels reais):
        'Atendimento': 'Este texto fala sobre atendimento ao cliente, qualidade do atendimento ou experiência com atendentes',
        'Preço': 'Este texto menciona preços, valores, custos ou questões financeiras',
        'Qualidade': 'Este texto avalia a qualidade de produtos ou serviços',
        'Rapidez': 'Este texto comenta sobre velocidade, agilidade ou tempo de resposta',
        'Produto': 'Este texto descreve características ou problemas com produtos',
        
        # ADICIONE SUAS 48 LABELS AQUI:
        # 'Sua_Label_1': 'Descrição da label 1',
        # 'Sua_Label_2': 'Descrição da label 2',
        # 'Sua_Label_3': 'Descrição da label 3',
        # ... continue até a label 48
        
        'Abrangência do plano / atendimento em outras cidades / abrangência nacional': 'Cliente informa que o plano tem alta abrangência, atende o brasil inteiro, tanto no interior e capital',
       'Agilidade / facilidade na autorização de exames e procedimentos': 'Cliente informa que é rápido e fácil obter a autorização de exames e procedimentos, sem muita burocracia',
       'Bom acompanhamento de doenças (ex: diabetes)': 'O plano possui um bom acompanhamento para pessoas com doenças', 
       'Cobertura do plano': 'Informa que o plano cobre muitos procedimentos e exames, possui uma cobertura boa',
       'Conseguir marcar sem passar pelo clínico geral': 'Menciona que consegue marcar exames e consultas sem passar pelo clínico',
       'Desconto em medicamentos em algumas farmácias': 'Menciona que com o plano consegue descontos em medicamentos',
       'Facilidade / agilidade para consultas / exames / procedimentos': 'Rapidez e facilidade para marcar as consultar e exames',
       'Facilidade com a carteirinha digital': 'Facilidade de resolver as questões do plano utilizando a carteirinha digital', 
       'Facilidade de reembolso': 'Facilidade em obter o reembolso caso necessário',
       'Marca conhecida / confiável': 'Quando informar que a marca é muito conhecida no mercado e é uma marca confiável',
       'Nunca teve problemas / não tem reclamações / está satisfeito': 'Menciona que nunca teve problemas, ou atendeu todas as expectativas, ou não tem nada a reclamar, ou está satisfeito com o plano',
       'Não sabe / Não respondeu': 'Menciona que não sabe, ou não sabe explicar, ou ainda não teve experiências para dizer', 
       'Preço do plano / custo benefício': 'Menciona que o plano é barato, ou sobre o custo benefício, ou descontos bom com o plano co-participativo, ou valores',
       'Programas interessantes (ex: Mais 60, curso para parar de fumar)': 'Menciona que faz parte ou que gosta do programa mais 60, ou sobre o curso parar de fumar',
       'Qualidade / agilidade do atendimento': 'Menciona sempre ser bem atendido, atendimento rápida, qualidade do atendimento',
       'Qualidade / facilidade do aplicativo / site': 'Fácil acesso ao aplicativo e site, praticidade ao usar aplicativo e site com informações bem esclarecidas',
       'Qualidade de atendimento da Central Unimed': 'Menciona sobre a qualidade do antendimento pela central telefônica da unimed',
       'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)': 'Menciona sobre a qualidade das redes credenciadas como clínicas, hospitais e laboratórios',
       'Qualidade do atendimento no hospital / pronto atendimento / pronto socorro / urgência': 'Cliente informa da qualidade e agilidade no atendimento de suas redes como hospital, pronto atendimento, pronto socorro e urgência',
       'Qualidade dos médicos credenciados': '',
       'Qualidade dos médicos especialistas': 'Quando informa que é atendida por bons especialistas, pelo cuidado dos médicos',
       'Quantidade de médicos credenciados': 'Menciona sobre a quantidade grande de médicos à disposição, variedade de profissionais, fácil conseguir médicos na região',
       'Quantidade de médicos especialistas': 'Menciona sobre a quantidade de médicos especialistas à disposição, variedade de especialistas, fácil conseguir um médico especialista na região',
       'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)': 'Menciona que o plano é aceito em vários lugares, a rede credenciada é grande, bastante opções de clínicas e hospitais',
       'Rede / hospital próprio': 'Menciona que o plano possui sua rede própria e hospital próprio',
       'Telemedicina / consulta pelo aplicativo / whatsapp': 'Menciona que consegue fazer consulta por whatsapp ou pelo aplicativo, pelo atendimento virtual facilitando a consulta online',
       'Ter atendimento 24h para urgência / emergência':  'Menciona tem atendiemnto 24 horas para qualquer urgência',
       'Ter rede credenciada perto de onde mora / o hospital que gosta': 'Informa ter facilidade com hospital e clínica próximo de casa',
       'Transparência': 'Menciona a honestidade e transparência da marca, sempre cumpriu o que prometeu'
    }
    
    # Criar dicionário de descrições
    descriptions = {}
    for label in labels:
        # Se a label está no mapeamento manual, usar descrição personalizada
        if label in manual_mapping:
            descriptions[label] = manual_mapping[label]
        else:
            # Senão, usar o nome da label como fallback
            descriptions[label] = label
    
    return descriptions


def create_label_descriptions_CATEGORIAS(labels):
    """
    OPÇÃO 2: Descrições baseadas em categorias.
    
    Use esta opção se suas labels seguem um padrão de categorias.
    """
    
    descriptions = {}
    
    for label in labels:
        label_lower = label.lower()
        
        # Categoria: Atendimento
        if any(word in label_lower for word in ['atendimento', 'atendente', 'atender']):
            descriptions[label] = f"Este texto fala sobre {label} relacionado ao atendimento ao cliente"
        
        # Categoria: Preço
        elif any(word in label_lower for word in ['preço', 'valor', 'custo', 'caro', 'barato']):
            descriptions[label] = f"Este texto menciona {label} sobre preços ou custos"
        
        # Categoria: Qualidade
        elif any(word in label_lower for word in ['qualidade', 'bom', 'ruim', 'péssimo', 'excelente']):
            descriptions[label] = f"Este texto avalia {label} de produtos ou serviços"
        
        # Categoria: Tempo/Rapidez
        elif any(word in label_lower for word in ['rápido', 'lento', 'demora', 'agilidade', 'tempo']):
            descriptions[label] = f"Este texto comenta sobre {label} relacionado a velocidade ou tempo"
        
        # Categoria: Produto
        elif any(word in label_lower for word in ['produto', 'item', 'mercadoria']):
            descriptions[label] = f"Este texto descreve {label} sobre produtos"
        
        # Categoria: Serviço
        elif any(word in label_lower for word in ['serviço', 'entrega', 'instalação']):
            descriptions[label] = f"Este texto fala sobre {label} relacionado a serviços"
        
        # Categoria: Reclamação
        elif any(word in label_lower for word in ['reclamação', 'problema', 'erro', 'falha']):
            descriptions[label] = f"Este texto contém {label} ou relato de problemas"
        
        # Categoria: Elogio
        elif any(word in label_lower for word in ['elogio', 'parabéns', 'satisfação', 'feliz']):
            descriptions[label] = f"Este texto expressa {label} ou satisfação"
        
        # Fallback: usar nome da label
        else:
            descriptions[label] = label
    
    return descriptions


def create_label_descriptions_TEMPLATE_COMPLETO(labels):
    """
    OPÇÃO 3: Template de frases completas.
    
    Use esta opção para máxima clareza e naturalidade.
    """
    
    # Templates de frases
    templates = {
        # Formato: 'Label': 'Frase completa descrevendo quando esta label se aplica'
        
        'Atendimento': 'O cliente está comentando sobre o atendimento que recebeu',
        'Atendimento_Bom': 'O cliente está elogiando o atendimento recebido',
        'Atendimento_Ruim': 'O cliente está reclamando do atendimento recebido',
        
        'Preço': 'O cliente está falando sobre preços ou valores',
        'Preço_Alto': 'O cliente está reclamando que o preço está alto ou caro',
        'Preço_Justo': 'O cliente está dizendo que o preço é justo ou adequado',
        
        'Qualidade': 'O cliente está avaliando a qualidade do produto ou serviço',
        'Qualidade_Boa': 'O cliente está satisfeito com a qualidade',
        'Qualidade_Ruim': 'O cliente está insatisfeito com a qualidade',
        
        'Rapidez': 'O cliente está comentando sobre a velocidade ou agilidade',
        'Demora': 'O cliente está reclamando de demora ou lentidão',
        
        'Produto': 'O cliente está falando sobre características do produto',
        'Produto_Defeito': 'O cliente está relatando defeito ou problema no produto',
        
        # ADICIONE SUAS LABELS AQUI com frases completas:
        # 'Sua_Label': 'Frase completa descrevendo quando esta label se aplica',

        'Cliente informa que o plano tem alta abrangência, atende o brasil inteiro, tanto no interior e capital': 'Abrangência do plano / atendimento em outras cidades / abrangência nacional',
       'Cliente informa que é rápido e fácil obter a autorização de exames e procedimentos, sem muita burocracia': 'Agilidade / facilidade na autorização de exames e procedimentos',
       'O plano possui um bom acompanhamento para pessoas com doenças': 'Bom acompanhamento de doenças (ex: diabetes)', 
       'Informa que o plano cobre muitos procedimentos e exames, possui uma cobertura boa': 'Cobertura do plano',
       'Menciona que consegue marcar exames e consultas sem passar pelo clínico': 'Conseguir marcar sem passar pelo clínico geral',
       'Menciona que com o plano consegue descontos em medicamentos': 'Desconto em medicamentos em algumas farmácias',
       'Rapidez e facilidade para marcar as consultar e exames': 'Facilidade / agilidade para consultas / exames / procedimentos',
       'Facilidade de resolver as questões do plano utilizando a carteirinha digital': 'Facilidade com a carteirinha digital', 
       'Facilidade em obter o reembolso caso necessário': 'Facilidade de reembolso',
       'Quando informar que a marca é muito conhecida no mercado e é uma marca confiável': 'Marca conhecida / confiável',
       'Menciona que nunca teve problemas, ou atendeu todas as expectativas, ou não tem nada a reclamar, ou está satisfeito com o plano': 'Nunca teve problemas / não tem reclamações / está satisfeito',
       'Menciona que não sabe, ou não sabe explicar, ou ainda não teve experiências para dizer': 'Não sabe / Não respondeu', 
       'Menciona que o plano é barato, ou sobre o custo benefício, ou descontos bom com o plano co-participativo, ou valores': 'Preço do plano / custo benefício',
       'Menciona que faz parte ou que gosta do programa mais 60, ou sobre o curso parar de fumar': 'Programas interessantes (ex: Mais 60, curso para parar de fumar)',
       'Menciona sempre ser bem atendido, atendimento rápida, qualidade do atendimento': 'Qualidade / agilidade do atendimento',
       'Fácil acesso ao aplicativo e site, praticidade ao usar aplicativo e site com informações bem esclarecidas': 'Qualidade / facilidade do aplicativo / site',
       'Menciona sobre a qualidade do antendimento pela central telefônica da unimed': 'Qualidade de atendimento da Central Unimed',
       'Menciona sobre a qualidade das redes credenciadas como clínicas, hospitais e laboratórios': 'Qualidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Cliente informa da qualidade e agilidade no atendimento de suas redes como hospital, pronto atendimento, pronto socorro e urgência': 'Qualidade do atendimento no hospital / pronto atendimento / pronto socorro / urgência',
       'Menciona que é muito bem atendido pelos médicos, a qualidade dos médicos': 'Qualidade dos médicos credenciados',
       'Quando informa que é atendida por bons especialistas, pelo cuidado dos médicos especialistas': 'Qualidade dos médicos especialistas',
       'Menciona sobre a quantidade grande de médicos à disposição, variedade de profissionais, fácil conseguir médicos na região': 'Quantidade de médicos credenciados',
       'Menciona sobre a quantidade de médicos especialistas à disposição, variedade de especialistas, fácil conseguir um médico especialista na região': 'Quantidade de médicos especialistas',

       'Menciona que o plano é aceito em vários lugares, a rede credenciada é grande, bastante opções de clínicas e hospitais': 'Quantidade de rede credenciada (laboratórios / clínicas / hospitais)',
       'Menciona que o plano possui sua rede própria e hospital próprio': 'Rede / hospital próprio',
       'Menciona que consegue fazer consulta por whatsapp ou pelo aplicativo, pelo atendimento virtual facilitando a consulta online': 'Telemedicina / consulta pelo aplicativo / whatsapp',
       'Menciona tem atendiemnto 24 horas para qualquer urgência': 'Ter atendimento 24h para urgência / emergência',
       'Informa ter facilidade com hospital e clínica próximo de casa': 'Ter rede credenciada perto de onde mora / o hospital que gosta',
       'Menciona a honestidade e transparência da marca, sempre cumpriu o que prometeu': 'Transparência'
    }
    
    descriptions = {}
    for label in labels:
        descriptions[label] = templates.get(label, label)
    
    return descriptions


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Exemplo de labels
    example_labels = [
        'Atendimento',
        'Preço',
        'Qualidade',
        'Rapidez',
        'Produto'
    ]
    
    print("="*80)
    print("EXEMPLOS DE DESCRIÇÕES")
    print("="*80)
    
    print("\n1. Mapeamento Manual:")
    print("-" * 80)
    desc1 = create_label_descriptions_TEMPLATE(example_labels)
    for label, desc in desc1.items():
        print(f"   {label:20s} → {desc}")
    
    print("\n2. Baseado em Categorias:")
    print("-" * 80)
    desc2 = create_label_descriptions_CATEGORIAS(example_labels)
    for label, desc in desc2.items():
        print(f"   {label:20s} → {desc}")
    
    print("\n3. Frases Completas:")
    print("-" * 80)
    desc3 = create_label_descriptions_TEMPLATE_COMPLETO(example_labels)
    for label, desc in desc3.items():
        print(f"   {label:20s} → {desc}")
    
    print("\n" + "="*80)
    print("DICAS PARA CRIAR BOAS DESCRIÇÕES")
    print("="*80)
    print("""
    1. Use linguagem natural e clara
    2. Seja específico sobre o que a label representa
    3. Evite ambiguidade
    4. Use frases completas quando possível
    5. Teste diferentes formulações
    6. Considere o contexto do seu domínio
    
    Exemplos de BOAS descrições:
    ✅ "O cliente está reclamando do atendimento recebido"
    ✅ "Este texto menciona preços altos ou custos elevados"
    ✅ "O cliente está satisfeito com a qualidade do produto"
    
    Exemplos de descrições RUINS:
    ❌ "Atendimento" (muito vago)
    ❌ "Coisas sobre preço" (informal demais)
    ❌ "Qualidade boa ou ruim" (ambíguo)
    """)

