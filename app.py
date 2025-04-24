import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha de dados
df = pd.read_csv("dados/data.csv")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visual da conversa
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown("""
Olá! 😊  
Eu sou o **SAFBot**, criado especialmente para conversar sobre o projeto agroflorestal do **Sítio Cristal**.  
Pode me perguntar qualquer coisa sobre espécies plantadas, lucros, tipos de produto ou até mesmo o que é um SAF.  
Fique à vontade, eu explico tudo de forma bem simples! 🌿

---

📌 Exemplos do que você pode perguntar:
- Quais espécies tem no SAF Cristal?
- Qual foi o lucro em 2040?
- O que é um SAF?
- Como esse sistema ajuda o meio ambiente?

Estou aqui pra conversar! 😄
        """)

# 💬 Mostra o histórico anterior
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Inicializa modelo OpenAI
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=openai_key
)

# 🧠 Cria o agente com base no DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# 🧑‍🌾 Campo de entrada
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

# Processa a pergunta
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.spinner("Consultando os dados do SAF Cristal..."):
        try:
            resposta = agent.run(query)
        except Exception as e:
            resposta = f"❌ Ocorreu um erro: {str(e)}"

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    # Salva o histórico para exibição
    st.session_state.visible_history.append((query, resposta))
