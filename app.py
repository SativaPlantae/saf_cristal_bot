# app.py — Sítio Cristal · Assistente IA (PT-BR)

import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
except ImportError:
    from langchain_experimental.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# 🌿 Variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Sítio Cristal · Assistente IA 🌱", layout="wide")
st.title("🐝 Sítio Cristal — Assistente IA")
st.markdown("Converse sobre os dados do SAF. Respostas claras e simples!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data_2.csv", sep=";")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visível + mensagem de boas-vindas
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(
            """Olá! 😊  
Eu sou a SAFBot, ajudante do Sítio Cristal. Estou aqui para explicar tudo sobre o nosso sistema agroflorestal. 🌱💬  

Quer saber quais espécies cultivamos, quanto rendeu em determinado ano ou o que é um SAF? Pergunte à vontade. 🐝💛  

---  
📌 Exemplos de perguntas que você pode fazer:  
- Quais espécies existem no SAF?  
- Qual foi o lucro total do Sítio Cristal?  
- Em que ano tivemos o maior faturamento?  
- Quantas espécies estão produzindo atualmente?  
- O que significa SAF?  
"""
        )

# Histórico
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 📊 Agente com acesso ao DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# =========================
# Funções auxiliares
# =========================

def faturamento_total(df_):
    return df_["faturamento"].sum() if "faturamento" in df_.columns else df_["faturamento (R$)"].sum()

def lucro_total(df_):
    return df_["lucro"].sum()

def despesas_total(df_):
    return df_["despesas"].sum()

def anos_de_duracao(df_):
    return len(df_["anos"].unique())

def media_anual(df_, coluna):
    return df_.groupby("anos")[coluna].sum().mean()

def media_mensal(df_, coluna):
    return media_anual(df_, coluna) / 12

def maior_menor_faturamento(df_):
    col = "faturamento" if "faturamento" in df_.columns else "faturamento (R$)"
    faturamento_ano = df_.groupby("anos")[col].sum()
    maior = faturamento_ano.idxmax()
    menor = faturamento_ano.idxmin()
    return maior, menor

def pergunta_envia_para_planilha(texto: str) -> bool:
    palavras_chave = [
        "lucro", "renda", "espécies", "especies", "produzindo", "produção", "anos",
        "quantos", "qual foi", "faturamento", "quanto gerou", "valores", "total",
        "tipo", "individuos", "preco", "produto"
    ]
    t = texto.lower()
    return any(k in t for k in palavras_chave)

# ===== ENTRADA =====
query = st.chat_input("Pergunte algo sobre o SAF do Sítio Cristal!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando os dados do Sítio Cristal... 📊"):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Ops! Não consegui acessar os dados agora: {str(e)}]"
    else:
        resposta_dados = ""

    entrada = (
        "Você é a SAFBot 🐝, ajudante do Sítio Cristal. "
        "Explique de forma acolhedora e simples, sem jargões técnicos — como quem conversa na varanda. "
        "Seja amigável e claro. Responda com base no contexto e, se houver, nos dados abaixo:\n\n"
        f"{resposta_dados}\n\n"
        f"Pergunta do usuário: {query}"
    )

    resposta_obj = llm_chat.invoke(
        st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=entrada)]
    )
    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
