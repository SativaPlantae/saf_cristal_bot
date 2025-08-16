import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# 🌿 Load env vars
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Page config
st.set_page_config(page_title="Cristal Farm AI Agent 🌱", layout="wide")
st.title("🐝 Cristal Farm — AI Assistant")
st.markdown("Chat with the assistant about the SAF data. Clear, simple answers — like a friendly porch conversation!")

# 📊 Load spreadsheet (kept in Portuguese on GitHub)
df = pd.read_csv("dados/data_2.csv", sep=";")

# 🧠 Conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Visible history + welcome message
if "visible_history" not in st.session_state:
    st.session_state.visible_history = []
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown("""
Hello! 😊  
I’m **SAFBot**, a helper from **Cristal Farm**. I’m here to chat with you and explain everything about our agroforestry system. 🌱💬  
Want to know which species we have? How much a certain year yielded? Or what exactly an SAF is? Ask away — I’ll keep it simple and direct, like we’re talking on the porch. 🐝💛

---
📌 Examples of what you can ask:
- Which species are in Cristal Farm’s SAF?
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

# =========================
# 🔁 TWO-WAY TRANSLATOR
# =========================

# Column mapping (EN -> PT)
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

# Value mapping per field (EN -> PT)
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

# PT -> EN (reverse)
value_alias_pt_to_en = {
    # tipo
    "Agrícola": "Agricultural",
    "Florestal": "Forestry",
    "Frutífera": "Fruit-bearing",
    # esta_produzindo
    "Sim": "Yes",
    "Não": "No",
    # especies
    "Açaí": "Açaí",
    "Andiroba": "Andiroba",
    "Banana": "Banana",
    "Cacau": "Cacao",
    "Coqueiro": "Coconut Palm",
    "Cupuçu": "Cupuaçu",
    "Mamão": "Papaya",
    "Milho": "Corn",
    "Mogno": "Mahogany",
    # produto
    "Fruto": "Fruit",
    "Madeira": "Wood",
    "Pamonha": "Corn Cake",
    "Polpa": "Pulp",
    "Suco": "Juice",
}

def _regex_replace_words(text: str, mapping: dict, case_insensitive=True):
    """Replace whole words using a mapping dict with regex boundaries."""
    flags = re.IGNORECASE if case_insensitive else 0
    # Sort by length to replace longer phrases first (avoid partial overlaps)
    for k in sorted(mapping.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(k) + r"\b"
        text = re.sub(pattern, mapping[k], text, flags=flags)
    return text

def translate_query_to_pt(query: str) -> str:
    """Map English column names and English value tokens to Portuguese before sending to the agent."""
    q = query
    # Columns
    q = _regex_replace_words(q, column_alias, case_insensitive=True)
    # Values
    q = _regex_replace_words(q, value_alias_en_to_pt, case_insensitive=True)
    return q

def translate_text_pt_to_en(text: str) -> str:
    """Translate common Portuguese values back to English in the agent's output."""
    if not isinstance(text, str):
        return text
    return _regex_replace_words(text, value_alias_pt_to_en, case_insensitive=False)

# =========================
# Helper functions (PT cols)
# =========================
def faturamento_total(df_):
    # If currency strings are present (e.g., 'R$ 66.360,00'), leave numeric parsing to the agent as needed
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

# 🔎 Decide whether to query the spreadsheet
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

# ===== USER INPUT =====
query = st.chat_input("Ask anything about Cristal Farm’s SAF!")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    if pergunta_envia_para_planilha(query):
        with st.spinner("Checking Cristal Farm data... 📊"):
            try:
                # Translate query EN -> PT before sending to the agent
                query_pt = translate_query_to_pt(query)
                resposta_dados = agent.run(query_pt)
                # Translate any PT tokens in the agent's raw data reply back to EN
                resposta_dados = translate_text_pt_to_en(resposta_dados)
            except Exception as e:
                resposta_dados = f"[Oops! I couldn’t fetch the data right now: {str(e)}]"
    else:
        resposta_dados = ""

    # Build the assistant instruction
    input_completo = (
        "You are SAFBot 🐝, a helper from Cristal Farm. "
        "Explain things in a warm, simple way without technical jargon — like talking with someone from the countryside. "
        "Be friendly and clear. Answer based on this context and the data below if available:\n\n"
        f"{resposta_dados}\n\n"
        f"User question: {query}"
    )

    # Run chat model with memory
    resposta_obj = llm_chat.invoke(
        st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=input_completo)]
    )

    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)
    # Final safety pass: translate any remaining PT tokens to EN in the final message
    resposta = translate_text_pt_to_en(resposta)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
