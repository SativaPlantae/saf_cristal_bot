import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 🌿 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🌱 Configuração da página
st.set_page_config(page_title="Agente SAF Cristal 🌱", layout="wide")
st.title("🐝 Agente Inteligente do Sítio Cristal")
st.markdown("Converse com o agente sobre os dados do SAF. Ele fala fácil, como quem troca ideia na varanda!")

# 📊 Carrega a planilha
df = pd.read_csv("dados/data.csv")

# 🧠 Memória de conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🧾 Histórico exibido
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
- O que é um SAF?
- Como esse sistema ajuda o meio ambiente?

Estou aqui pra conversar! 😄
""")

# Exibe o histórico
for user_msg, bot_msg in st.session_state.visible_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(bot_msg)

# 🤖 Modelos
llm_chat = ChatOpenAI(temperature=0.3, model="gpt-4o", openai_api_key=openai_key)
llm_agent = OpenAI(temperature=0.3, openai_api_key=openai_key)

# 📊 Agente para consulta ao DataFrame
agent = create_pandas_dataframe_agent(
    llm=llm_agent,
    df=df,
    verbose=False,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# 🎯 Prompt do SAFBot
prompt_template = PromptTemplate.from_template("""
Você é o SAFBot 🐝, um ajudante virtual do Sítio Cristal. Seu papel é conversar com simplicidade, simpatia e bom humor,
explicando conceitos de SAF e ajudando com dúvidas baseadas nos dados reais do projeto.

Use linguagem acessível e evite termos técnicos. Responda de forma acolhedora e clara.
""")

conversation = ConversationChain(
    llm=llm_chat,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False
)

# 🔎 Detecta se deve consultar a planilha
def pergunta_envia_para_planilha(texto):
    palavras_chave = [
        "lucro", "renda", "espécies", "produzindo", "produção", "anos", "quantos",
        "qual foi", "em", "faturamento", "quanto gerou", "valores"
    ]
    return any(p in texto.lower() for p in palavras_chave)

# 🧑‍🌾 Entrada do usuário
query = st.chat_input("Digite aqui sua pergunta sobre o SAF:")

if query:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(query)

    # Verifica se é uma pergunta que precisa consultar dados
    if pergunta_envia_para_planilha(query):
        with st.spinner("Consultando a planilha do SAF Cristal... 📊"):
            try:
                resposta_dados = agent.run(query)
            except Exception as e:
                resposta_dados = f"[Erro ao consultar os dados: {str(e)}]"
    else:
        resposta_dados = ""

    # Combina a pergunta com a resposta dos dados
    input_completo = query
    if resposta_dados:
        input_completo += f"\n\n[Informações da planilha]: {resposta_dados}"

    # Gera resposta
    resposta = conversation.run(input_completo)

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)

    st.session_state.visible_history.append((query, resposta))
