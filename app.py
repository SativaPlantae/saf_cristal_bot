import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Pode perguntar como se fosse pra um amigo — ele fala a sua língua!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv")

# 🔁 Inicializa histórico se ainda não existir
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 👋 Mensagem de boas-vindas
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown("""
Olá! 😊  
Eu sou o **SAFBot**, seu ajudante aqui no Sítio Cristal 🌽  
Pode me perguntar qualquer coisa sobre o projeto agroflorestal — desde o que está sendo produzido até como tudo funciona.  
Prometo responder sem enrolação, de um jeito fácil de entender. Vamos nessa? 🌱
""")

# 🧠 Prompt natural e amigável com memória
prompt_template = PromptTemplate.from_template("""
Você é o SAFBot, um assistente simpático que responde com carinho, sem termos técnicos.
Explique de forma simples, como se estivesse conversando com alguém da zona rural que nunca ouviu falar de SAF ou IA.
Use o contexto da conversa para continuar respondendo de forma natural e acolhedora.

Histórico:
{chat_history}

Pergunta: {pergunta}
""")

# 🤖 Inicializa modelo e cadeia com memória
llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4o",
    openai_api_key=openai_key
)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False
)

# Entrada do usuário
query = st.chat_input("O que você quer saber sobre o SAF?")

# Exibe o histórico anterior
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# Se houver nova pergunta
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🦗"):
        with st.spinner("Consultando os dados e lembranças do SAFBot..."):
            try:
                resposta = chain.run(pergunta=query)
            except Exception as e:
                resposta = f"❌ Ocorreu um erro: {e}"
        st.markdown(resposta)

    # Atualiza o histórico visível
    st.session_state.chat_history.append((query, resposta))
