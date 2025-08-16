import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# 🌿 Load env vars
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Page config
st.set_page_config(page_title="Sítio Cristal AI Agent 🌱", layout="wide")
st.title("🐝 Sítio Cristal — AI Assistant")
st.markdown("Chat with the assistant about the SAF data. Clear, simple answers — like a friendly porch conversation!")

# 📊 Load spreadsheet
df = pd.read_csv("dados/data.csv", sep=";")

# 🧠 Conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Visible history + welcome message
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown("""
Hello! 😊  
I’m **SAFBot**, a helper from **Sítio Cristal**. I’m here to chat with you and explain everything about our agroforestry system. 🌱💬  
Want to know which species we have? How much a certain year yielded? Or what exactly an SAF is? Ask away — I’ll keep it simple and direct, like we’re talking on the porch. 🐝💛

---
📌 Examples you can ask:
- Which species are in SAF Cristal?
- What was the profit in 2040?
- What is an SAF?
- How does this system help the environment?
        """)

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Models
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 📊 Agent with DataFrame access
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# Helper functions (keep column names as they are in the CSV)
def faturamento_total(df):
    return df["faturamento (R$)"].sum()

def lucro_total(df):
    return df["lucro (R$)"].sum()

def despesas_total(df):
    return df["despesas (R$)"].sum()

def anos_de_duracao(df):
    return len(df["anos"].unique())

def media_anual(df, coluna):
    return df.groupby("anos")[coluna].sum().mean()

def media_mensal(df, coluna):
    return media_anual(df, coluna) / 12

def maior_menor_faturamento(df):
    faturamento_ano = df.groupby("anos")["faturamento (R$)"].sum()
    maior = faturamento_ano.idxmax()
    menor = faturamento_ano.idxmin()
    return maior, menor

# 🔎 Decide whether to query the spreadsheet
def pergunta_envia_para_planilha(texto: str) -> bool:
    keywords_en_pt = [
        # EN
        "profit", "revenue", "income", "species", "producing", "production", "years",
        "how many", "which year", "in", "turnover", "how much", "values", "total",
        # PT (keep for bilingual robustness)
        "lucro", "renda", "espécies", "produzindo", "produção", "anos", "quantos",
        "qual foi", "faturamento", "quanto gerou", "valores", "total"
    ]
    t = texto.lower()
    return any(k in t for k in keywords_en_pt)

# User input
query = st.chat_input("Ask anything about SAF Cristal!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    if pergunta_envia_para_planilha(query):
        with st.spinner("Checking Sítio Cristal data... 📊"):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Oops! I couldn’t fetch the data right now: {str(e)}]"
    else:
        resposta_dados = ""

    input_completo = (
        "You are SAFBot 🐝, a helper from Sítio Cristal. "
        "Explain things in a warm, simple way without technical jargon — like talking with someone from the countryside. "
        "Be friendly and clear. Answer based on this context and the data below if available:\n\n"
        f"{resposta_dados}\n\n"
        f"User question: {query}"
    )

    resposta_obj = llm_chat.invoke(
        st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=input_completo)]
    )

    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
