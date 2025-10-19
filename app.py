# app.py — Sítio Cristal · Assistente IA (PT-BR)

import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
except ImportError:
    from langchain_experimental.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# ============ Configuração base ============
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Sítio Cristal · Assistente IA 🌱", layout="wide")
st.title("🐝 Sítio Cristal — Assistente IA")
st.markdown("Converse sobre os dados do SAF. Respostas claras e simples!")

if not openai_key:
    st.warning("Defina a variável OPENAI_API_KEY nos Secrets/ambiente para o chatbot funcionar.")

df = pd.read_csv("dados/data_2.csv", sep=";")

# ============ Barra lateral ============
with st.sidebar:
    st.header("⚙️ Opções")
    DEBUG = st.toggle("Mostrar log de depuração", value=False, help="Exibe mensagens úteis durante a consulta")
    if st.button("🧹 Limpar conversa"):
        st.session_state.pop("visible_history", None)
        st.session_state.pop("memory", None)
        st.success("Conversa limpa! Você pode começar de novo.")

# ============ Memória ============
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Mensagem de boas-vindas + exemplos
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(
            """Olá! 😊  
Eu sou a SAFBot, ajudante do Sítio Cristal. Estou aqui para explicar tudo sobre o nosso sistema agroflorestal. 🌱💬  

Quer saber quais espécies cultivamos, quanto rendeu em determinado ano ou o que é um SAF? Pergunte à vontade. 🐝💛  

---  
📌 **Exemplos de perguntas**  
- Quais espécies existem no SAF?  
- Quantas espécies estão produzindo atualmente?  
- Qual foi o lucro total do Sítio Cristal?  
- Em que ano tivemos o maior faturamento?  
- O que significa SAF?  
"""
        )

# Histórico
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# ============ Modelos ============
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=openai_key)

# ============ Agente DataFrame ============
cols = ", ".join(df.columns.astype(str))
agent_prefix = (
    "Você é a SAFBot 🐝, ajudante do Sítio Cristal. "
    "Fale como alguém da roça: simples, acolhedor e, às vezes, até um pouco ignorante — daquele jeito que faz a gente rir, mas com o coração no lugar. "
    "Não use termos técnicos demais, e se não souber, invente uma explicação engraçada ou conte um ‘causo’. "
    "Mantenha o estilo de conversa leve, direta e natural, como um papo de varanda. "
    "Quando tiver acesso a dados, use-os para responder de forma objetiva e bem-humorada. "
    "As colunas disponíveis são: " + cols + ". "
    "Se houver campo de produção, considere `esta_produzindo == 'Sim'` como verdadeiro. "
    "Quando o usuário fizer perguntas, responda de forma descontraída e com um toque de sabedoria popular."
)

agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    agent_type="openai-tools",
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    prefix=agent_prefix,
)

# ============ Funções auxiliares ============
def pergunta_envia_para_planilha(texto: str) -> bool:
    palavras_chave = [
        "lucro", "renda", "espécies", "especies", "produzindo", "produção", "anos",
        "quantos", "qual foi", "faturamento", "quanto gerou", "valores", "total",
        "tipo", "individuos", "preco", "produto", "ano", "maior", "menor"
    ]
    t = texto.lower()
    return any(k in t for k in palavras_chave)

# ============ Entrada ============
query = st.chat_input("Pergunte algo sobre o SAF do Sítio Cristal!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    resposta_dados = ""
    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando os dados do Sítio Cristal... 📊"):
            try:
                if DEBUG:
                    st.info(f"[DEBUG] Enviando ao agente: {query}")
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Ops! Não consegui acessar os dados agora: {str(e)}]"

    mensagens_anteriores = st.session_state.memory.load_memory_variables({})["history"]
    mensagens = mensagens_anteriores + [
        HumanMessage(
            content=(
                "Você é a SAFBot 🐝, ajudante do Sítio Cristal. "
                "Fale como alguém simples do campo: acolhedor, simpático e um pouco ignorante às vezes — mas sempre bem-intencionado e divertido. "
                "Responda de forma leve, direta e com aquele jeitinho da roça, usando expressões populares e uma pitada de humor. "
                "Baseie suas respostas no contexto e, se houver, nos dados abaixo:\n\n"
                f"{resposta_dados}\n\n"
                f"Pergunta do usuário: {query}"
            )
        )
    ]

    resposta_obj = llm_chat.invoke(mensagens)
    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
