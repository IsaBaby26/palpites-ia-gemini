import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("IA de Palpites - Brasileirão 2023")

def obter_partidas_brasileirao(api_key):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {
    "league": 71,
    "season": 2023,
    "from": "2023-01-01",
    "to": "2023-12-31"
}
    response = requests.get(url, headers=headers, params=params)

    st.write("🔍 Status da resposta da API:", response.status_code)

    if response.status_code != 200:
        st.error(f"Erro ao consultar API: código {response.status_code}")
        return pd.DataFrame()

    dados = response.json()
    st.write("📦 Resposta bruta da API:", dados)

    if not dados.get('response'):
        st.warning("⚠️ Nenhum jogo encontrado na resposta da API.")
        return pd.DataFrame()

    partidas = []
    for jogo in dados['response']:
        goals = jogo['goals']
        home = jogo['teams']['home']['name']
        away = jogo['teams']['away']['name']
        partidas.append({
            "home_team": home,
            "away_team": away,
            "home_goals": goals['home'],
            "away_goals": goals['away'],
            "result": "Home Win" if goals['home'] > goals['away']
                      else "Away Win" if goals['away'] > goals['home'] else "Draw",
            "home_corners": np.nan,
            "away_corners": np.nan,
            "home_cards": np.nan,
            "away_cards": np.nan,
            "home_possession": np.nan,
            "away_possession": np.nan
        })
    return pd.DataFrame(partidas)

def treinar_modelo(df):
    df = df.dropna()
    label_encoder = LabelEncoder()
    df['result_encoded'] = label_encoder.fit_transform(df['result'])
    X = df[['home_goals', 'away_goals', 'home_corners', 'away_corners',
            'home_cards', 'away_cards', 'home_possession', 'away_possession']]
    y = df['result_encoded']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, label_encoder

def gerar_palpite_completo(home, away, df, model, label_encoder):
    last_home = df[df['home_team'] == home].tail(5)
    last_away = df[df['away_team'] == away].tail(5)
    avg = lambda col: col.mean() if not col.empty else 0.0
    stats = {f"{side}_{metric}": avg(last_home[metric]) if side == "home" else avg(last_away[metric])
             for side in ["home", "away"]
             for metric in ["goals", "corners", "cards", "possession"]}
    input_data = pd.DataFrame([stats])
    probs = model.predict_proba(input_data)[0]
    labels = label_encoder.classes_
    result = sorted(zip(labels, probs), key=lambda x: -x[1])
    best, prob = result[0]
    st.subheader(f"{home} x {away}")
    st.markdown(f"**Palpite principal:** {best} ({prob*100:.1f}%)")
    st.markdown(" | ".join([f"{r}: {p*100:.1f}%" for r, p in result]))

api_key = st.text_input("Cole sua API Key da API-FOOTBALL", type="password")

if st.button("Carregar dados"):
    df = obter_partidas_brasileirao(api_key)
    if not df.empty:
        model, encoder = treinar_modelo(df)
        times = sorted(set(df['home_team']).union(df['away_team']))
        home = st.selectbox("Time mandante", times)
        away = st.selectbox("Time visitante", [t for t in times if t != home])
        if st.button("Gerar Palpite"):
            gerar_palpite_completo(home, away, df, model, encoder)
