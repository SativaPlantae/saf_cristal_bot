import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI  # usa gpt-4o

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da chave
openai_key = os.getenv("OPENAI_API_KEY")

# Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🤖 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Faça perguntas como se estivesse no ChatGPT!")

# Carregar planilha
df = pd.read_csv("dados/data.csv")

# Mostrar a tabela, se quiser
with st.expander("📊 Visualizar base de dados"):
    st.dataframe(df)

# Sessão de histórico de conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Campo de entrada
query = st.chat_input("Digite sua pergunta sobre o SAF:")

# Mostrar histórico anterior
for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_msg)

# Quando houver nova pergunta
if query:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            # Cria o modelo e agente
            llm = ChatOpenAI(
                temperature=0.2,
                model="gpt-4o",
                openai_api_key=openai_key
            )
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            try:
                resposta = agent.run(query)
            except Exception as e:
                resposta = f"❌ Ocorreu um erro: {e}"

            st.markdown(resposta)

    # Atualiza histórico
    st.session_state.chat_history.append((query, resposta))
