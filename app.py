import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv", sep=";")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visual
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown("""
Olá! 😊  
Eu sou o **SAFBot**, um ajudante do **Sítio Cristal**. Estou aqui pra bater um papo gostoso com você e explicar tudo sobre nosso sistema agroflorestal. 🌱💬  
Quer saber quais espécies temos? Quanto rendeu um certo ano? Ou o que é exatamente um SAF? Pode perguntar sem medo! Eu explico tudo de um jeito bem simples e direto, como se estivéssemos conversando na varanda. 🐝💛

---
📌 Exemplos do que você pode perguntar:
- Quais espécies tem no SAF Cristal?
- Qual foi o lucro em 2040?
- O que é um SAF?
- Como esse sistema ajuda o meio ambiente?
        """)

# Exibe o histórico de conversa
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 🔗 Cadeia com memória para manter o histórico
conversation = ConversationChain(
    llm=llm_chat,
    memory=st.session_state.memory,
    verbose=False
)

# 📊 Agente para consultar dados do DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# Funções auxiliares para cálculo com base na planilha
def faturamento_total(df):
    return df["faturamento (R$)"].sum()

def lucro_total(df):
    return df["lucro (R$)"].sum()

def despesas_total(df):
    return df["despesas (R$)"].sum()

def anos_de_duracao(df):
    return len(df["anos"].unique())

def media_anual(df, coluna):
    return df.groupby("anos")[coluna].sum().mean()

def media_mensal(df, coluna):
    return media_anual(df, coluna) / 12

def maior_menor_faturamento(df):
    faturamento_ano = df.groupby("anos")["faturamento (R$)"].sum()
    maior = faturamento_ano.idxmax()
    menor = faturamento_ano.idxmin()
    return maior, menor

# Detecta se a pergunta deve acessar os dados do DataFrame
def pergunta_envia_para_planilha(texto):
    palavras_chave = [
        "lucro", "renda", "espécies", "produzindo", "produção", "anos", "quantos",
        "qual foi", "em", "faturamento", "quanto gerou", "valores", "total"
    ]
    return any(p in texto.lower() for p in palavras_chave)

# Entrada de pergunta do usuário
query = st.chat_input("Pode perguntar qualquer coisa sobre o SAF Cristal!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    # Consulta aos dados se necessário
    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando os dados do Sítio Cristal... 📊"):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"Não consegui acessar os dados agora. 😢 Erro: {str(e)}"
    else:
        resposta_dados = ""

    # Constrói a mensagem com estilo e dados
    input_completo = (
        "Você é o SAFBot 🐝, um ajudante do Sítio Cristal. "
        "Responda como se estivesse conversando com alguém da zona rural, explicando tudo de forma gentil, didática e sem termos técnicos. "
        "Se tiver informações da planilha, use elas. Se não, apenas responda com simpatia.\n\n"
        f"Pergunta: {query}\n"
        f"{resposta_dados}"
    )

    resposta = conversation.run(input_completo)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
