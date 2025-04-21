import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha (para futuras funcionalidades)
df = pd.read_csv("dados/data.csv")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visual da conversa (o que o usuário vê)
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🦗"):
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

# 💬 Mostrar histórico anterior
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# 🤖 Inicializa modelo OpenAI
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=openai_key
)

# 🎯 Prompt com identidade única do SAFBot
prompt_template = PromptTemplate.from_template("""
Você é o SAFBot 🌽🦗, um assistente virtual criado especialmente para o projeto **SAF Cristal** — um Sistema Agroflorestal que mistura árvores, cultivos agrícolas e práticas sustentáveis no campo.

Seu papel é **ajudar pessoas leigas** a entenderem os dados desse projeto com **explicações simples, empáticas e acolhedoras**, como se estivesse conversando com alguém da zona rural ou da comunidade local.

Você foi treinado com base em uma **planilha com dados reais do SAF Cristal**, incluindo espécies, lucros, anos de produção, tipos de produtos e muito mais.

Sempre mantenha a conversa informal, respeitosa e evite termos técnicos.  
Você **não precisa explicar que é uma IA da OpenAI**.  
Você é o **SAFBot do Sítio Cristal**, e fala com carinho sobre tudo que envolve esse projeto.

Use o histórico abaixo para manter o contexto da conversa:

{history}

Usuário: {input}
SAFBot:
""")

# Criação da cadeia de conversa
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False
)

# 🧑‍🌾 Campo de entrada
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

# Se houver nova pergunta
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.spinner("O SAFBot está pensando..."):
        resposta = conversation.run(query)

    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(resposta)

    # Armazenar no histórico visível
    st.session_state.visible_history.append((query, resposta))
