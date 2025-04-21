import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv")

# Inicializa memória real de conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Boas-vindas (só uma vez)
if "has_greeted" not in st.session_state:
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown("""
Oi! 😊  
Eu sou o SAFBot, seu ajudante aqui no Sítio Cristal. Pode me perguntar qualquer coisa sobre as plantações, os lucros, as espécies ou até como funciona esse tal de SAF.  
Prometo explicar como se fosse uma boa conversa no campo 🌿🌽
        """)
    st.session_state.has_greeted = True

# Modelo e cadeia com memória
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=openai_key
)

prompt_template = PromptTemplate.from_template("""
Você é o SAFBot, um assistente simpático e acolhedor que conversa com pessoas que não conhecem nada sobre agricultura ou tecnologia.

Responda com empatia, explicações simples e num tom leve — como se estivesse explicando para um amigo curioso.

Use o histórico da conversa para manter o contexto da resposta e evitar repetições desnecessárias.

Histórico da conversa:
{history}

Usuário: {input}
SAFBot:
""")

conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False
)

# Campo de entrada
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

# Exibir histórico (manual)
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# Nova interação
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🦗"):
        with st.spinner("Deixa eu pensar aqui..."):
            resposta = conversation.run(query)
        st.markdown(resposta)

    # Atualiza histórico visível
    st.session_state.visible_history.append((query, resposta))
