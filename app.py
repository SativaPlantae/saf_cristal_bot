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

# 📊 Carrega e prepara os dados
df = pd.read_csv("dados/data.csv", sep=";")

def parse_currency(value):
    try:
        return float(str(value).replace("R$", "").replace(".", "").replace(",", "."))
    except:
        return 0.0

df["faturamento (R$)"] = df["faturamento (R$)"].apply(parse_currency)
df["despesas (R$)"] = df["despesas (R$)"].apply(parse_currency)
df["lucro (R$)"] = df["lucro (R$)"].apply(parse_currency)

# Calcula totais e médias com base em 15 anos
total_faturamento = df["faturamento (R$)"].sum()
numero_anos_saf = 15
faturamento_medio_anual = total_faturamento / numero_anos_saf
faturamento_medio_mensal = faturamento_medio_anual / 12

# Adiciona colunas auxiliares para perguntas específicas
df["faturamento_mensal"] = faturamento_medio_mensal

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visual
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
- Qual foi o faturamento total?
- Quanto gerou por mês, em média?
- Quais espécies estão produzindo?
- O que é um SAF?
        """)

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🧐 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 📊 Agente com acesso ao DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# Conversa informal com memória
conversation = ConversationChain(
    llm=llm_chat,
    memory=st.session_state.memory,
    verbose=False
)

# 🔍 Classifica a pergunta

def classificar_tipo_pergunta(texto):
    texto = texto.lower()
    if "por mês" in texto or "mensal" in texto:
        return "mensal"
    elif "total" in texto or "anual" in texto or "por ano" in texto:
        return "anual"
    else:
        return "geral"

query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    tipo = classificar_tipo_pergunta(query)

    if tipo == "mensal":
        resposta_dados = f"O faturamento médio mensal do SAF é de aproximadamente R$ {faturamento_medio_mensal:,.2f}."
    elif tipo == "anual":
        if "total" in query:
            resposta_dados = f"O faturamento total do SAF ao longo dos 15 anos é de R$ {total_faturamento:,.2f}."
        else:
            resposta_dados = f"O faturamento médio anual do SAF é de aproximadamente R$ {faturamento_medio_anual:,.2f}."
    else:
        try:
            resposta_dados = agent.run(query)
        except Exception as e:
            resposta_dados = f"[Erro ao consultar os dados: {str(e)}]"

    input_completo = (
        "Você é o SAFBot 🐝, um ajudante virtual do Sítio Cristal. "
        "Explique tudo com simplicidade, simpatia e linguagem acessível. "
        "Evite termos técnicos. Responda com base nas informações abaixo, se houver:\n\n"
        f"Pergunta: {query}\n"
        f"{resposta_dados}"
    )

    resposta_final = conversation.run(input_completo)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta_final)

    st.session_state.visible_history.append((query, resposta_final))
