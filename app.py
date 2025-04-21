import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🦗 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Pergunte como se fosse a um amigo curioso — ele fala a sua língua!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv")

# 💬 Histórico de conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 👋 Boas-vindas só no início
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown("""
Olá! 😊  
Eu sou o **SAFBot**, seu assistente aqui no Sítio Cristal 🌿  
Pode me perguntar qualquer coisa sobre as plantações, o que está sendo produzido, quanto se gastou, o que deu lucro...  
Não se preocupe com palavras difíceis. Eu explico tudo de um jeito simples, como uma boa prosa na sombra de um pé de açaí 🌳  
""")

# 🧑‍🌾 Entrada do usuário
query = st.chat_input("O que você quer saber sobre o SAF?")

# 🧾 Exibe o histórico anterior
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🦗"):
        st.markdown(bot_msg)

# 🚀 Nova pergunta
if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🦗"):
        with st.spinner("Puxando conversa com o SAFBot..."):

            # 🤖 Inicializa o modelo com GPT-4o
            llm = ChatOpenAI(
                temperature=0.2,
                model="gpt-4o",
                openai_api_key=openai_key
            )

            # 💬 Prompt com linguagem acessível e natural
            prompt_template = PromptTemplate.from_template("""
Você é um assistente simpático chamado SAFBot.

Seu público são pessoas que **não sabem nada sobre inteligência artificial, agricultura ou SAF**.  
Explique tudo com **palavras simples**, seja acolhedor e humano.  
Fale como se estivesse conversando com um amigo da roça, sem usar jargões ou termos técnicos.  
Dê exemplos práticos. Mantenha a resposta clara, leve e sem parecer que veio de uma máquina.  
Sempre responda em **português brasileiro**.

Pergunta: {pergunta}
""")

            # 🔗 Cadeia para gerar respostas amigáveis
            chain = LLMChain(llm=llm, prompt=prompt_template)

            try:
                resposta = chain.run(pergunta=query)
            except Exception as e:
                resposta = f"❌ Ocorreu um erro: {e}"

        st.markdown(resposta)

    # 📝 Atualiza o histórico
    st.session_state.chat_history.append((query, resposta))
