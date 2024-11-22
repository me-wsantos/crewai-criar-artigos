import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from IPython.display import Markdown

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')

gpt4o = ChatOpenAI(model_name='gpt-4o')

search_tool = SerperDevTool() #busca dados do google
scrape_tool = ScrapeWebsiteTool() #faz scraping das páginas

#Buscador de conteúdo
buscador = Agent(
    role = 'Buscador de conteúdo',
    goal = 'Buscar conteúdo online atualizado sobre {tema}',
    backstory = 'Você é um excelente buscador de conteúdo.\n'
                'Você é um expert e entusiasta de {tema} e busca \n'
                'conteúdos de valor para compartilhar com as pessoas.',
    tools  = [search_tool, scrape_tool],
    verbose = True,
    llm = gpt4o,
    max_iter = 15,
    max_rpm = 30,
    memory = True,
    function_calling_llm = gpt4o,
    max_execution_time = 120,
    allow_code_execution = False,
    allow_delegation=False
)

#Redator
redator = Agent(
    role = 'Redator de conteúdo',
    goal = 'Escrever um artigo sobre {tema}',
    backstory = 'Você é um excelente escritor, reconhecido por escrever artigos envolventes e interessantes. \n'
                'Você utiliza os dados coletados pelo buscador de conteúdo, remove sua a complexidade e \n'
                'e transforma em um texto de fácil compreensão, interessante e divertido.\n'
                'Seus textos sempre são claros, pedagógicos e tão fáceis de ler que até uma criança de 10 anos entende perfeitamente.',
    tools  = [],
    verbose = True,
    llm = gpt4o,
    max_iter = 15,
    max_rpm = 30,
    memory = True,
    function_calling_llm = gpt4o,
    max_execution_time = 120,
    allow_code_execution = False,
)

#Editor
editor = Agent(
    role = 'Editor de conteúdo',
    goal = 'Revisar artigos sobre {tema}',
    backstory = 'Você é um exímio conhecedor da língua portuguesa que domina todas as regras gramaticais e concordância.\n'
                'Você transforma o texto escrito pelo redator deixando gramaticalmente correto,\n'
                'sempre com um tom formal e envolvente.',
    tools  = [],
    verbose = True,
    llm = gpt4o,
    max_iter = 15,
    max_rpm = 30,
    memory = True,
    function_calling_llm = gpt4o,
    max_execution_time = 120,
    allow_code_execution = False,
)

#Tasks
buscar = Task(
    description = ('1 - Priorize as últimas tendências, os principais atores e as notícias mais relevantes sobre {tema}.\n'
                  '2 - Identifique o público-alvo, considerando seus interesses e dores.\n'
                  '3 - Inclua palavras-chave de SEO e dados ou fontes reelvantes.'),
    agent=buscador,
    expected_output='Um lista de notícias relevantes sobre {tema}.'
)

redigir = Task(
    description = ('1 - Use os dados coletados par acriar um post para o Linkedin sobre o tema {tema}.\n'
                  '2 - Incorpore palavras-chave de SEO de forma natural.\n'
                  '3 - Certifique-se de que o post esteja estruturado de forma cativante, clara, de fácil leitura e compreensão, com uma conclusão que leve a reflexão e à ação.'),
    agent=redator,
    expected_output='Um artigo bem escrito sobre {tema}.'
)

editar = Task(
    description = '1 - Revisar o artigo do Linkedin para garantir que o texto esteja gramaticalmente correto, envolvente e claro.\n',
    agent=editor,
    expected_output='Um artigo pronto para publicação no Linkedin.'
)

equipe = Crew(
    agents=[buscador, redator, editor],
    tasks=[buscar, redigir, editar]
)

tema_artigo = 'O que é CrewAI?'
entradas = { "tema": tema_artigo }

response = equipe.kickoff(inputs=entradas)
Markdown(response.raw)

print("Processo finalizado!**********************")
print(response)