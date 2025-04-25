import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 📊 Carrega os dados
df = pd.read_csv("dados/data.csv", sep=";")

# Corrige nomes das colunas
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# 💡 Correção: converter colunas monetárias
for col in ["faturamento_(r$)", "despesas_(r$)", "lucro_(r$)"]:
    df[col] = df[col].astype(str).str.replace("R$", "", regex=False).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 🧠 Memória da conversa
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
- Quais espécies existem no SAF Cristal?
- Qual foi o lucro total?
- Quanto rende por ano, em média?
- Como esse sistema ajuda o meio ambiente?

Estou aqui pra conversar! 😄
        """)

for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 🧮 Agente para analisar os dados
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# 🔎 Detecta se deve consultar a planilha
def pergunta_envia_para_planilha(texto):
    palavras_chave = [
        "lucro", "renda", "espécie", "produzindo", "produção", "anos", "quantos", "valores",
        "despesa", "faturamento", "quanto", "receita", "ganho", "foi lucrado", "média", "total"
    ]
    return any(p in texto.lower() for p in palavras_chave)

# 🧠 Cálculos automáticos
def calcular_metricas(df):
    anos = df["anos"].unique()
    duracao_em_anos = len(anos)

    total_faturamento = df["faturamento_(r$)"].sum()
    total_lucro = df["lucro_(r$)"].sum()
    total_despesas = df["despesas_(r$)"].sum()

    media_anual_faturamento = total_faturamento / duracao_em_anos
    media_mensal_faturamento = media_anual_faturamento / 12

    return {
        "anos": duracao_em_anos,
        "faturamento_total": total_faturamento,
        "lucro_total": total_lucro,
        "despesas_total": total_despesas,
        "faturamento_anual": media_anual_faturamento,
        "faturamento_mensal": media_mensal_faturamento
    }

metricas = calcular_metricas(df)

# Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    if pergunta_envia_para_planilha(query):
        try:
            resposta_dados = agent.run(query)
        except Exception as e:
            resposta_dados = f"(Desculpe, não consegui acessar os dados agora: {str(e)})"
    else:
        resposta_dados = ""

    resumo_dados = (
        f"O projeto SAF Cristal tem duração de aproximadamente {metricas['anos']} anos.\n"
        f"O faturamento total é de aproximadamente R$ {metricas['faturamento_total']:,.2f}, "
        f"com lucro total de R$ {metricas['lucro_total']:,.2f} e despesas de R$ {metricas['despesas_total']:,.2f}.\n"
        f"Em média, fatura R$ {metricas['faturamento_anual']:,.2f} por ano, "
        f"o que equivale a cerca de R$ {metricas['faturamento_mensal']:,.2f} por mês.\n"
    )

    prompt_inicial = (
        "Você é o SAFBot 🐝, um assistente simples, didático e amigável. "
        "Evite termos técnicos e responda como se estivesse conversando com alguém da comunidade rural. "
        "Use linguagem acessível e explique com carinho e bom humor. Se estiver usando dados, "
        "utilize o resumo abaixo como referência adicional:\n\n"
        f"{resumo_dados}\n\n"
        f"Pergunta do usuário: {query}\n"
        f"{resposta_dados}"
    )

    resposta_final = ChatOpenAI(
        temperature=0.3, model="gpt-4o", openai_api_key=openai_key
    ).invoke(prompt_inicial).content

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta_final)

    st.session_state.visible_history.append((query, resposta_final))
