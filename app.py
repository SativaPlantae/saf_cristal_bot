import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente (opcional)
load_dotenv()

st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🤖 Agente Inteligente do Sítio Cristal")
st.markdown("Faça perguntas sobre o projeto SAF com base na planilha!")

# Configuração da API
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.warning("⚠️ A chave da OpenAI não foi definida.")

# Carrega a planilha
df = pd.read_csv("dados/data.csv")

# Exibe a tabela (opcional)
with st.expander("📊 Visualizar base de dados"):
    st.dataframe(df)

# Cria o agente LangChain
llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_key)
agent = create_pandas_dataframe_agent(llm, df, verbose=False)

# Entrada do usuário
query = st.text_input("Digite sua pergunta sobre o SAF:")

# Resposta
if query:
    with st.spinner("Analisando..."):
        resposta = agent.run(query)
        st.success("Resposta:")
        st.write(resposta)
