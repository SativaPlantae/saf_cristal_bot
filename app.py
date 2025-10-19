# app.py — Sítio Cristal · Assistente IA (PT-BR)

import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# Import robusto (algumas versões movem a função de lugar)
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

# Aviso se faltar a chave
if not openai_key:
    st.warning("Defina a variável OPENAI_API_KEY nos Secrets/ambiente para o chatbot funcionar.")

# Carrega a planilha (ajuste o caminho se necessário)
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

# Mensagem de boas-vindas + exemplos (só na 1ª renderização)
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

# Re-render do histórico visual
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# ============ Modelos ============
# Chat para respostas conversacionais
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)

# Chat para o agente (recomendado: temperatura 0.0)
llm_agent = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=openai_key)

# ============ Agente DataFrame ============
cols = ", ".join(df.columns.astype(str))
agent_prefix = (
    "Você é um analista de dados do Sítio Cristal e tem acesso a um DataFrame chamado df. "
    "Trabalhe em **português**. Use Python para consultar o df e calcule exatamente o que for pedido "
    "(contar, somar, filtrar, agrupar). "
    f"As colunas disponíveis são: {cols}. "
    "Quando houver campo de produção, considere `esta_produzindo == 'Sim'` como verdadeiro. "
    "Responda de forma direta, mostrando números e, quando fizer sentido, uma frase breve de conclusão."
)

agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    agent_type="openai-tools",          # essencial para tool calling
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    prefix=agent_prefix,
)

# ============ Utilidades (se quiser usar em respostas rápidas) ============
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

# Heurística simples: decidir se deve consultar a planilha
def pergunta_envia_para_planilha(texto: str) -> bool:
    palavras_chave = [
        "lucro", "renda", "espécies", "especies", "produzindo", "produção", "anos",
        "quantos", "qual foi", "faturamento", "quanto gerou", "valores", "total",
        "tipo", "individuos", "preco", "produto", "ano", "maior", "menor"
    ]
    t = texto.lower()
    return any(k in t for k in palavras_chave)

# ============ Entrada do usuário ============
query = st.chat_input("Pergunte algo sobre o SAF do Sítio Cristal!")

if query:
    # Exibe mensagem do usuário
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    # Consulta à planilha (quando fizer sentido)
    resposta_dados = ""
    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando os dados do Sítio Cristal... 📊"):
            try:
                if DEBUG:
                    st.info(f"[DEBUG] Enviando ao agente: {query}")
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Ops! Não consegui acessar os dados agora: {str(e)}]"

    # Constrói as mensagens incluindo a MEMÓRIA
    mensagens_anteriores = st.session_state.memory.load_memory_variables({})["history"]
    mensagens = mensagens_anteriores + [
        HumanMessage(
            content=(
                "Você é a SAFBot 🐝, ajudante do Sítio Cristal. "
                "Explique de forma acolhedora e simples, sem jargões técnicos — como quem conversa na varanda. "
                "Seja amigável e claro. Responda com base no contexto e, se houver, nos dados abaixo:\n\n"
                f"{resposta_dados}\n\n"
                f"Pergunta do usuário: {query}"
            )
        )
    ]

    # Gera a resposta com contexto + memória
    resposta_obj = llm_chat.invoke(mensagens)
    resposta = resposta_obj.content.strip() if hasattr(resposta_obj, "content") else str(resposta_obj)

    # Exibe a resposta
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    # Atualiza histórico/memória
    st.session_state.visible_history.append((query, resposta))
    st.session_state.memory.save_context({"input": query}, {"output": resposta})
