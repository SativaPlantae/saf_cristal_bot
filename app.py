# app.py — Cristal Farm · Assistente IA (PT-BR)

import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
# Import robusto (algumas versões movem a função de lugar)
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
except ImportError:
    from langchain_experimental.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# 🌿 Variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Cristal Farm · Assistente IA 🌱", layout="wide")
st.title("🐝 Cristal Farm — Assistente IA")
st.markdown("Converse sobre os dados do SAF. Respostas claras e simples — como um papo de varanda!")

# 📊 Carrega a planilha (em PT no repositório)
df = pd.read_csv("dados/data_2.csv", sep=";")

# 🧠 Memória da conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico visível + mensagem de boas-vindas
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(
            """
Olá! 😊  
Eu sou a **SAFBot**, ajudante da **Cristal Farm**. Estou aqui para explicar tudo sobre o nosso sistema agroflorestal. 🌱💬  
Quer saber **quais espécies temos**, **quanto rendeu em determinado ano** ou **o que é um SAF**? Pergunte à vontade — falo simples e direto, como numa conversa na varanda. 🐝💛

---
📌 Exemplos do que você pode perguntar:
- Quais espécies existem no SAF da Cristal Farm?
- Qual foi o **lucro** em 2040?
- O que é um SAF?
- Como esse sistema ajuda o meio ambiente?
"""
        )

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

# =========================
# 🔁 TRADUTOR BIDIRECIONAL
# =========================

# Mapa de colunas (EN -> PT)
column_alias = {
    "type": "tipo",
    "years": "anos",
    "year": "anos",
    "species": "especies",
    "producing": "esta_produzindo",
    "expenses": "despesas",
    "revenue": "faturamento",
    "profit": "lucro",
    "individuals": "individuos",
    "price": "preco",
    "product": "produto",
}

# Valores (EN -> PT)
value_alias_en_to_pt = {
    # tipo
    "agricultural": "Agrícola",
    "forestry": "Florestal",
    "fruit-bearing": "Frutífera",
    # esta_produzindo
    "yes": "Sim",
    "no": "Não",
    # especies
    "açaí": "Açaí",
    "acai": "Açaí",
    "andiroba": "Andiroba",
    "banana": "Banana",
    "cacao": "Cacau",
    "cocoa": "Cacau",
    "coconut palm": "Coqueiro",
    "coconut": "Coqueiro",
    "cupuaçu": "Cupuçu",
    "cupuuçu": "Cupuçu",
    "cupuuco": "Cupuçu",
    "cupuaçu ": "Cupuçu",
    "papaya": "Mamão",
    "corn": "Milho",
    "mahogany": "Mogno",
    # produto
    "fruit": "Fruto",
    "wood": "Madeira",
    "corn cake": "Pamonha",
    "pulp": "Polpa",
    "juice": "Suco",
}

# Valores (PT -> EN) — útil se o agente responder em EN
value_alias_pt_to_en = {
    "Agrícola": "Agricultural",
    "Florestal": "Forestry",
    "Frutífera": "Fruit-bearing",
    "Sim": "Yes",
    "Não": "No",
    "Açaí": "Açaí",
    "Andiroba": "Andiroba",
    "Banana": "Banana",
    "Cacao": "Cacao",
    "Cacau": "Cacao",
    "Coqueiro": "Coconut Palm",
    "Cupuçu": "Cupuaçu",
    "Mamão": "Papaya",
    "Milho": "Corn",
    "Mogno": "Mahogany",
    "Fruto": "Fruit",
    "Madeira": "Wood",
    "Pamonha": "Corn Cake",
    "Polpa": "Pulp",
    "Suco": "Juice",
}

def _regex_replace_words(text: str, mapping: dict, case_insensitive=True):
    """Substitui palavras inteiras usando um dicionário (com fronteiras de palavra)."""
    flags = re.IGNORECASE if case_insensitive else 0
    for k in sorted(mapping.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(k) + r"\b"
        text = re.sub(pattern, mapping[k], text, flags=flags)
    return text

def translate_query_to_pt(query: str) -> str:
    """Mapeia nomes de colunas e valores (EN -> PT) antes de enviar ao agente."""
    q = _regex_replace_words(query, column_alias, case_insensitive=True)
    q = _regex_replace_words(q, value_alias_en_to_pt, case_insensitive=True)
    return q

def translate_text_en_to_pt(text: str) -> str:
    """Caso o agente retorne algum termo fixo em EN, converte para PT."""
    if not isinstance(text, str):
        return text
    return _regex_replace_words(text, value_alias_en_to_pt, case_insensitive=True)

# =========================
# Funções auxiliares (colunas PT)
# =========================
def faturamento_total(df_):
    return df_["faturamento"].sum() if "faturamento" in df_.columns else df_["faturamento (R$)"].sum()

def lucro_total(df_):
    return df_["lucro"].sum()

def despesas_total(df_):
    return df_["despesas"].sum()

def anos_de_duracao(df_):
    return len(df_["anos"].unique())

def media_anual(df_, coluna):
    return df_.groupby("anos")[coluna].sum().mean()

def media_mensal(df_, coluna):
    return media_anual(df_, coluna) / 12

def maior_menor_faturamento(df_):
    col = "faturamento" if "faturamento" in df_.columns else "faturamento (R$)"
    faturamento_ano = df_.groupby("anos")[col].sum()
    maior = faturamento_ano.idxmax()
    menor = faturamento_ano.idxmin()
    return maior, menor

# 🔎 Detecta se a pergunta deve ir para a planilha
def pergunta_envia_para_planilha(texto: str) -> bool:
    keywords_en_pt = [
        # EN
        "profit", "revenue", "income", "species", "producing", "production", "years",
        "how many", "which year", "turnover", "how much", "values", "total", "type",
        "individuals", "price", "product",
        # PT
        "lucro", "renda", "espécies", "especies", "produzindo", "produção", "anos",
        "quantos", "qual foi", "faturamento", "quanto gerou", "valores", "total",
        "tipo", "individuos", "preco", "produto"
    ]
    t = texto.lower()
    return any(k in t for k in keywords_en_pt)

# ===== ENTRADA DO USUÁRIO =====
query = st.chat_input("Pergunte algo sobre o SAF da Cristal Farm!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando os dados da Cristal Farm... 📊"):
            try:
                # Traduz a consulta (EN -> PT) antes de enviar ao agente
                query_pt = translate_query_to_pt(query)
                resposta_dados = agent.run(query_pt)
                # Garante termos em português se o agente devolver algo em EN
                resposta_dados = translate_text_en_to_pt(resposta_dados)
            except Exception as e:
                resposta_dados = f"[Ops! Não consegui acessar os dados agora: {str(e)}]"
    else:
        resposta_dados = ""

    # Instrução para o modelo de conversa
    input_completo = (
        "Você é a SAFBot 🐝, ajudante da Cristal Farm. "
        "Explique de forma acolhedora e simples, sem jargões técnicos — como quem conversa na varanda. "
        "Seja amigável e claro. Responda com base no contexto e, se houver, nos dados abaixo:\n\n"
        f"{resposta_dados}\n\n"
        f"Pergunta do usuário: {query}"
    )

    # Modelo de chat com memória
    resposta_obj = llm_chat.invoke(
        st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=input_completo)]
    )

    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)
    # Passo final: assegura termos de domínio em PT
    resposta = translate_text_en_to_pt(resposta)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
