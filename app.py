import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌿", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# Carrega a planilha
df = pd.read_csv("dados/data.csv", sep=";")

# Funções de apoio

def faturamento_total(df):
    return df["faturamento (R$)"].sum()

def faturamento_mensal(df):
    return faturamento_total(df) / (df['anos'].nunique() * 12)

def faturamento_anual_medio(df):
    return faturamento_total(df) / df['anos'].nunique()

def duracao_saf(df):
    return df['anos'].nunique()

def ano_maior_menor_faturamento(df):
    agrupado = df.groupby('anos')["faturamento (R$)"].sum()
    maior = agrupado.idxmax()
    menor = agrupado.idxmin()
    return maior, menor

# Memória de conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Histórico visual
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown("""
Olá! 😊  
Eu sou o **SAFBot** 🐝, seu assistente simpático e sempre pronto para ajudar. Como posso te ajudar hoje?
        """)

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# Agente
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

conversation = ConversationChain(
    llm=llm_chat,
    memory=st.session_state.memory,
    verbose=False
)

# Regras para acionar respostas baseadas nos dados
perguntas_especiais = {
    "faturamento total": lambda: f"O faturamento total desse SAF é de {faturamento_total(df):,.2f} reais.",
    "quantos anos": lambda: f"O SAF vai durar {duracao_saf(df)} anos.",
    "média por mês": lambda: f"O faturamento médio mensal é de aproximadamente {faturamento_mensal(df):,.2f} reais.",
    "média por ano": lambda: f"O faturamento médio anual é de aproximadamente {faturamento_anual_medio(df):,.2f} reais.",
    "maior e menor": lambda: (
        lambda maior, menor: f"O ano de maior faturamento foi {maior} e o de menor faturamento foi {menor}."
    )(*ano_maior_menor_faturamento(df))
}

# Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    resposta = None
    for chave, funcao in perguntas_especiais.items():
        if chave in query.lower():
            resposta = funcao()
            break

    if not resposta:
        resposta = conversation.run(query)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
