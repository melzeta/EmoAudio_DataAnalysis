import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

emotion_colors = {
    "amusement": "#F9E264",
    "anger": "#D20101",
    "sadness": "#2A3CBD",
    "contentment": "#A8E6A3",
    "disgust": "#DF1FDB",
    "awe": "#9DBCF5",
    "fear": "#08811E",
    "excitement": "#F88F68"
}

st.set_page_config(page_title="Music Emotion Dashboard", layout="wide")

# --- CARICAMENTO DATI ---
@st.cache_data
def load_and_process_data():
    # Carichiamo il file (assicurati che si chiami data.json nel Codespace)
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    user_rows = []
    response_rows = []
    
    for user_id, user_info in data.get('userData', {}).items():
        # Dati demografici/Generali utente
        user_rows.append({
            "user_id": user_id,
            "gender": user_info.get("gender", "N/A"),
            "age": user_info.get("age", "N/A"),
            "num_responses": len(user_info.get("emotionResponses", []))
        })
        
        # Dati delle singole risposte
        for resp in user_info.get('emotionResponses', []):
            if 'emotionValues' in resp:
                path_parts = resp['song'].split('/')
                intended = path_parts[1] if len(path_parts) > 1 else "unknown"
                
                # Uniamo i dati delle emozioni con i metadati della risposta
                row = resp['emotionValues'].copy()
                row.update({
                    "user_id": user_id,
                    "song_path": resp['song'],
                    "intended_emotion": intended,
                    "time_spent": resp.get("timeSpentSeconds", 0)
                })
                response_rows.append(row)
    
    return pd.DataFrame(user_rows), pd.DataFrame(response_rows)

df_users, df_responses = load_and_process_data()

@st.cache_data
def load_original_emotions():
    """Carica i valori emozionali originali delle canzoni"""
    try:
        df_original = pd.read_csv('song_emotions.csv')
    except FileNotFoundError:
        st.warning("⚠️ File song_emotions.csv non trovato. Gli spider charts mostreranno solo i dati degli utenti.")
        return {}
    
    # Crea un dizionario per accesso rapido: {filename: {emozione: valore}}
    original_dict = {}
    for _, row in df_original.iterrows():
        # Il filename è già nel formato corretto: "amusement\amusement_19692.mp3"
        filename = row['filename']
        
        original_dict[filename] = {
            'amusement': row['amusement'],
            'anger': row['anger'],
            'awe': row['awe'],
            'contentment': row['contentment'],
            'disgust': row['disgust'],
            'excitement': row['excitement'],
            'fear': row['fear'],
            'sadness': row['sadness']
        }
    
    return original_dict

# Carica i dati originali
original_emotions = load_original_emotions()

# --- SIDEBAR ---
st.sidebar.title("🎵 Analisi Emozioni")
menu = st.sidebar.radio("Sezioni:", [
    "📊 Panoramica Dataset", 
    "🕷️ Spider Charts"
])
# --- SEZIONE 1: PANORAMICA DATASET ---
if menu == "📊 Panoramica Dataset":
    st.header("Esplorazione Generale dei Dati")

    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)

    # Conta solo i valori unici nella colonna user_id
    utenti_unici = df_users['user_id'].nunique() 
    tot_risposte = len(df_responses)
    canzoni_uniche = df_responses['song_path'].nunique()
    media_risposte = round(tot_risposte / utenti_unici, 1) if utenti_unici > 0 else 0

    col1.metric("Utenti Unici", utenti_unici)
    col2.metric("Totale Risposte", tot_risposte)
    col3.metric("Canzoni Uniche", canzoni_uniche)
    col4.metric("Media Risposte/Utente", media_risposte)
    st.divider()

    # Layout a due colonne per i grafici
    col_left, col_right = st.columns(2)
    
    # Stile comune per i grafici
    chart_layout = dict(
        plot_bgcolor="rgba(0,0,0,0)", # Sfondo trasparente
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    with col_left:
        st.subheader("Distribuzione Risposte")
        
        counts = df_responses['intended_emotion'].value_counts().reset_index()
        counts.columns = ['Emozione', 'Conteggio']
        
        fig_dist = px.bar(
            counts, 
            x='Emozione', 
            y='Conteggio',
            color='Emozione',
            color_discrete_map=emotion_colors,
            template="plotly_white", # Tema pulito
            labels={'Conteggio': 'Risposte', 'Emozione': ''}
        )
        
        fig_dist.update_layout(chart_layout)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_right:
        st.subheader("Impegno Utenti")
        
        fig_hist = px.histogram(
            df_users, 
            x="num_responses", 
            nbins=20,
            template="plotly_white",
            color_discrete_sequence=['#4A90E2'], # Un blu neutro professionale
            labels={'num_responses': 'Canzoni ascoltate', 'count': 'Frequenza Utenti'}
        )
        
        fig_hist.update_layout(chart_layout)
        fig_hist.update_yaxes(title_text="Numero di Utenti") # Etichetta asse Y chiara
        st.plotly_chart(fig_hist, use_container_width=True)
        utenti_attivi = len(df_users[df_users['num_responses'] > 0])
        
        st.write(f"""
            Su un totale di **{utenti_unici}** utenti registrati, 
            i dati analizzati derivano dai **{utenti_attivi}** utenti che hanno effettivamente 
            interagito con le canzoni.
        """)

# --- SEZIONE 2: SPIDER CHARTS ---
elif menu == "🕷️ Spider Charts":
    st.header("Analisi del Trasferimento Emotivo")
    st.write("Per individuare 5 canzoni per categoria, ho presupposto che una canzone possa rappresentare più di un'emozione")
    st.write("Per individuare le canzoni che rappresentano al meglio un'emozione ho preso l'average")
    # Assicurati che questa lista coincida con le chiavi di emotion_colors
    emotions_list = ["amusement", "anger", "sadness", "contentment", "disgust", "awe", "fear", "excitement"]
    
    col_img, col_tab = st.columns([1, 1.2]) # La tabella ha un po' più di spazio

    with col_img:
        st.subheader("Modello Emotivo")
        # Assicurati di avere l'immagine nella cartella del progetto
        # Se il file si chiama 'plutchik.png', usa:
        st.image("plutchik.png", caption="Ruota delle Emozioni di Plutchik", use_container_width=True)

    with col_tab:
        st.subheader("Top 5 per Categoria")
        
        # Calcoliamo le medie
        summary_stats = df_responses.groupby('song_path').agg({
            e: 'mean' for e in emotions_list
        }).reset_index()

        summary_rows = []
        for emotion in emotions_list:
            top_5_songs = summary_stats.nlargest(5, emotion)
            
            # Stringa semplice: solo i nomi delle canzoni separati da virgola
            names_list = ", ".join([row['song_path'].split('/')[-1] for _, row in top_5_songs.iterrows()])
            
            summary_rows.append({
                "EMOZIONE": emotion.upper(),
                "TOP 5 BRANI (Ranked)": names_list
            })
        
        # Visualizziamo la tabella semplificata
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.divider()

    for emotion in emotions_list:
        st.markdown(f"## {emotion.upper()}")
        # Usiamo il colore dedicato per la linea di separazione
        st.markdown(f"<div style='background-color: {emotion_colors.get(emotion, '#ccc')}; height: 4px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        
        # 1. Raggruppamento dati: calcoliamo la media di TUTTE le emozioni per ogni canzone
        song_stats = df_responses.groupby('song_path').agg({
            e: 'mean' for e in emotions_list
        }).reset_index()
        
        # Aggiungiamo il conteggio degli utenti separatamente
        user_counts = df_responses.groupby('song_path')['user_id'].nunique().reset_index()
        song_stats = song_stats.merge(user_counts, on='song_path')
        song_stats.rename(columns={'user_id': 'num_users'}, inplace=True)
        
        # 2. Ordina per l'emozione corrente e prendi top 5
        top_5 = song_stats.nlargest(5, emotion)
        
        # --- 5 SPIDER CHARTS IN COLONNE ---
        cols = st.columns(5)
        
        for idx, (index, song_row) in enumerate(top_5.iterrows()):
            with cols[idx]:
                song_full_name = song_row['song_path'].split('/')[-1]
                # Converti il path per matchare il formato del CSV (con backslash)
                # es: "amusement/amusement_19692.mp3" -> "amusement\amusement_19692.mp3"
                song_path_for_match = song_row['song_path'].replace('/', '\\')
                # Accorciamo il nome se troppo lungo per non rompere il layout
                song_name = (song_full_name[:15] + '..') if len(song_full_name) > 17 else song_full_name
                
                # Valori degli utenti
                user_values = [song_row[e] for e in emotions_list]
                
                # Cerca i valori originali usando il path convertito
                original_values = None
                if song_path_for_match in original_emotions:
                    original_values = [original_emotions[song_path_for_match][e] for e in emotions_list]
                
                # Crea spider chart con Plotly Graph Objects
                fig = go.Figure()
                
                # Valori utenti (linea colorata solida) - PRIMA per stare sopra
                fig.add_trace(go.Scatterpolar(
                    r=user_values,
                    theta=emotions_list,
                    fill='toself',
                    line=dict(color=emotion_colors.get(emotion, "#1f77b4"), width=2),
                    name='Utenti',
                    showlegend=False
                ))
                
                # Valori originali (linea grigia tratteggiata) - DOPO per stare sotto
                if original_values:
                    fig.add_trace(go.Scatterpolar(
                        r=original_values,
                        theta=emotions_list,
                        fill='toself',
                        line=dict(color='gray', dash='dash', width=2),
                        fillcolor='rgba(128, 128, 128, 0.15)',
                        name='Originale',
                        opacity=0.7,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                        angularaxis=dict(tickfont=dict(size=8))
                    ),
                    showlegend=False,
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=280,
                    title=dict(text=f"<b>{song_name}</b>", font=dict(size=11), y=0.92, x=0.5, xanchor='center')
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"{emotion}_{idx}")
                
                # Info sotto il grafico
                st.markdown(f"<p style='text-align: center; font-size: 10px; color: gray;'>Users: {int(song_row['num_users'])} | Score: {song_row[emotion]:.2f}</p>", unsafe_allow_html=True)
        
        st.markdown("---")