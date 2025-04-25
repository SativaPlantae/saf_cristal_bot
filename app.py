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
df["despesas_mensal"] = df["despesas (R$)"] / 12
df["faturamento_mensal"] = df["faturamento (R$)"] / 12
df["lucro_mensal"] = df["lucro (R$)"] / 12

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
- Qual foi o faturamento mensal em 2028?
- Quanto foi o lucro anual em 2040?
- Quais espécies estão produzindo?
- Como esse sistema ajuda o meio ambiente?

Estou aqui pra conversar! 😄
        """)

# 💬 Mostrar histórico
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
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

# 💡 Classificação da pergunta
def classificar_tipo_pergunta(texto):
    texto = texto.lower()
    if any(p in texto for p in ["mensal", "por mês", "mensalmente"]):
        return "mensal"
    elif any(p in texto for p in [
        "lucro", "renda", "espécies", "produzindo", "produção", "anos", "quantos",
        "qual foi", "em", "faturamento", "quanto gerou", "valores",
        "maior", "menor", "despesas", "gastos"
    ]):
        return "geral"
    else:
        return "conversa"

# Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    tipo = classificar_tipo_pergunta(query)

    if tipo == "mensal":
        with st.spinner("Consultando valores mensais do SAF... 📊"):
            try:
                query_mensal = (
                    query.replace("lucro", "lucro_mensal")
                         .replace("faturamento", "faturamento_mensal")
                         .replace("despesas", "despesas_mensal")
                )
                resposta_dados = agent.run(query_mensal)
            except Exception as e:
                resposta_dados = f"[Erro ao consultar dados mensais: {str(e)}]"
    elif tipo == "geral":
        with st.spinner("Consultando dados do SAF... 📊"):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Erro ao consultar dados gerais: {str(e)}]"
    else:
        resposta_dados = ""

    # Resposta final
    input_completo = (
        "Você é o SAFBot 🐝, um ajudante virtual do Sítio Cristal. "
        "Explique tudo com simplicidade, simpatia e linguagem acessível. "
        "Evite termos técnicos. Responda com base nas informações abaixo, se houver:\n\n"
        f"Pergunta: {query}\n"
        f"{resposta_dados}"
    )

    resposta = conversation.run(input_completo)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
