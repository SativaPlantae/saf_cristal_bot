import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv", sep=";")

# 🧠 Memória de conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visível
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
- Qual o faturamento total do SAF?
- Em que ano teve maior faturamento?
        """)

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 📊 Agente de planilha
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# 🔎 Funções de análise personalizada
def faturamento_total(df):
    return df["faturamento"].sum()

def faturamento_anual_medio(df):
    return faturamento_total(df) / 15  # SAF dura 15 anos

def faturamento_mensal_medio(df):
    return faturamento_total(df) / (15 * 12)

def obter_anos_maior_menor_faturamento(df):
    agrupado = df.groupby("anos")["faturamento"].sum()
    ano_maior = agrupado.idxmax()
    val_maior = agrupado.max()
    ano_menor = agrupado.idxmin()
    val_menor = agrupado.min()
    return ano_maior, val_maior, ano_menor, val_menor

# 🧠 Cadeia de conversa informal
conversation = ConversationChain(
    llm=llm_chat,
    memory=st.session_state.memory,
    verbose=False
)

# Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    resposta_dados = ""

    if "faturamento total" in query.lower():
        total = faturamento_total(df)
        resposta_dados = f"O faturamento total do SAF é de **R$ {total:,.2f}**."

    elif "faturamento médio por mês" in query.lower():
        mensal = faturamento_mensal_medio(df)
        resposta_dados = f"O faturamento médio mensal é de aproximadamente **R$ {mensal:,.2f}**."

    elif "faturamento médio por ano" in query.lower() or "fatura por ano" in query.lower():
        anual = faturamento_anual_medio(df)
        resposta_dados = f"O faturamento médio anual é de aproximadamente **R$ {anual:,.2f}**."

    elif "maior e menor faturamento" in query.lower():
        ano_max, val_max, ano_min, val_min = obter_anos_maior_menor_faturamento(df)
        resposta_dados = (
            f"O ano de **maior faturamento** foi {ano_max}, com **R$ {val_max:,.2f}**.\n\n"
            f"O ano de **menor faturamento** foi {ano_min}, com **R$ {val_min:,.2f}**."
        )

    elif "lucro total" in query.lower():
        total = df["lucro"].sum()
        resposta_dados = f"O lucro total do SAF durante os 15 anos foi de **R$ {total:,.2f}**."

    else:
        # 🤖 Deixa o agente consultar a planilha normalmente
        if any(p in query.lower() for p in ["lucro", "despesa", "faturamento", "ano", "valor"]):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"⚠️ Houve um erro ao consultar os dados: {e}"

    # Monta entrada para o modelo principal
    entrada_modelo = (
        "Você é o SAFBot 🐝, um assistente simpático e acessível. Responda com carinho e linguagem simples.\n"
        f"Pergunta: {query}\n"
        f"{resposta_dados}"
    )

    with st.spinner("O SAFBot está pensando..."):
        resposta = conversation.run(entrada_modelo)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
