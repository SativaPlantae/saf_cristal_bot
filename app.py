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

# 🌱 Configura página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")

# 🌾 Título e descrição
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# Estilo customizado: botão no canto inferior direito
st.markdown("""
    <style>
        .fixed-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
        }
    </style>
""", unsafe_allow_html=True)

# 🔁 Botão "Nova conversa"
with st.container():
    if st.markdown('<div class="fixed-button">', unsafe_allow_html=True):
        if st.button("🔄 Nova conversa"):
            st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            st.session_state.visible_history = []
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# 📊 Carrega dados (usado no futuro)
df = pd.read_csv("dados/data.csv")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visível (exibido no Streamlit)
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

# 💬 Mostra histórico anterior
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# 🤖 Modelo e cadeia com identidade SAFBot
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=openai_key
)

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

conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False
)

# 🧑‍🌾 Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.spinner("O SAFBot está pensando..."):
        resposta = conversation.run(query)

    # Renderiza APENAS uma vez
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(resposta)

    # Armazena no histórico visual (sem duplicação!)
    st.session_state.visible_history.append((query, resposta))
