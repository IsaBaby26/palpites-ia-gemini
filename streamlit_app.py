import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Palpite com IA", layout="centered")
st.title("🔮 Agente de Palpites com Google Gemini")
st.markdown("Escolha os times e receba uma análise automática da IA com base em dados atuais.")

# Campo para a chave da API Gemini
gemini_api_key = st.text_input("🔑 Cole sua Gemini API Key", type="password")

# Lista de times do Brasileirão
times = [
    "Flamengo", "Palmeiras", "São Paulo", "Corinthians", "Grêmio", "Atlético-MG", 
    "Cruzeiro", "Internacional", "Botafogo", "Fortaleza", "Bragantino", "Fluminense", 
    "Bahia", "Cuiabá", "Vasco", "Santos", "Atlético-GO", "Juventude", "Criciúma", "Vitória"
]

home_team = st.selectbox("🏠 Time mandante", times)
away_team = st.selectbox("🚩 Time visitante", [t for t in times if t != home_team])

# Geração de análise com IA Gemini
if gemini_api_key and st.button("🎯 Gerar Palpite com IA"):
    try:
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel(model_name="models/gemini-pro")

        prompt = f'''
Você é um analista profissional do futebol brasileiro. 
Analise o confronto entre {home_team} (mandante) e {away_team} (visitante), considerando:

- Situação atual de ambos os times no Brasileirão
- Últimos 5 resultados de cada um
- Confrontos diretos recentes
- Quem é favorito e por quê
- Se é provável que ambos marquem (sim ou não)
- Se haverá mais ou menos de 2.5 gols
- Um placar estimado
- E um comentário técnico completo como se fosse dito por um comentarista de TV

Forneça uma resposta clara, completa e confiável.
'''

        with st.spinner("Consultando a IA..."):
            resposta = model.generate_content(prompt)
            st.success("✅ Análise gerada pela IA:")
            st.markdown(resposta.text)

    except Exception as e:
        st.error(f"❌ Erro ao acessar a API Gemini:

{e}")
