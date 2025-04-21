import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Faça perguntas como se estivesse no ChatGPT!")

# Carrega a planilha
df = pd.read_csv("dados/data.csv")

# Histórico de conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Entrada do usuário
query = st.chat_input("Pergunte algo ao agente SAF:")

# Exibe o histórico anterior
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# Nova pergunta
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🦗"):
        with st.spinner("Pensando..."):

            # Inicializa o modelo com GPT-4o
            llm = ChatOpenAI(
                temperature=0.2,
                model="gpt-4o",
                openai_api_key=openai_key
            )

            # 🔄 PROMPT mais natural
            prompt_template = PromptTemplate.from_template("""
Você é um assistente inteligente que responde perguntas sobre um Sistema Agroflorestal (SAF) com base nos dados de um DataFrame.
Seja gentil, didático e fale como um humano que conhece muito de agroecologia.
Responda em português brasileiro.

Pergunta: {pergunta}
""")
            chain = LLMChain(llm=llm, prompt=prompt_template)

            try:
                resposta = chain.run(pergunta=query)
            except Exception as e:
                resposta = f"❌ Ocorreu um erro: {e}"

        st.markdown(resposta)

    # Atualiza o histórico
    st.session_state.chat_history.append((query, resposta))
