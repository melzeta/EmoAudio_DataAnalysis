import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_process_data():
    """Load and process user emotion response data from JSON file"""
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    user_rows = []
    response_rows = []
    
    for user_id, user_info in data.get('userData', {}).items():
        user_rows.append({
            "user_id": user_id,
            "gender": user_info.get("gender", "N/A"),
            "age": user_info.get("age", "N/A"),
            "num_responses": len(user_info.get("emotionResponses", []))
        })
        
        for resp in user_info.get('emotionResponses', []):
            if 'emotionValues' in resp:
                path_parts = resp['song'].split('/')
                intended = path_parts[1] if len(path_parts) > 1 else "unknown"
                
                row = resp['emotionValues'].copy()
                row.update({
                    "user_id": user_id,
                    "song_path": resp['song'],
                    "intended_emotion": intended,
                    "time_spent": resp.get("timeSpentSeconds", 0)
                })
                response_rows.append(row)
    
    return pd.DataFrame(user_rows), pd.DataFrame(response_rows)

@st.cache_data
def load_original_emotions():
    """Load original emotion values for songs from CSV file"""
    try:
        df_original = pd.read_csv('song_emotions.csv')
    except FileNotFoundError:
        st.warning("⚠️ File song_emotions.csv non trovato. Gli spider charts mostreranno solo i dati degli utenti.")
        return {}
    
    original_dict = {}
    for _, row in df_original.iterrows():
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

df_users, df_responses = load_and_process_data()
original_emotions = load_original_emotions()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎵 Analisi Emozioni")
menu = st.sidebar.radio("Sezioni:", [
    "📊 Panoramica Dataset", 
    "🕷️ Spider Charts"
])

# ============================================================================
# SECTION 1: DATASET OVERVIEW
# ============================================================================

if menu == "📊 Panoramica Dataset":
    st.header("Esplorazione Generale dei Dati")

    col1, col2, col3, col4 = st.columns(4)
    
    utenti_unici = df_users['user_id'].nunique() 
    tot_risposte = len(df_responses)
    canzoni_uniche = df_responses['song_path'].nunique()
    media_risposte = round(tot_risposte / utenti_unici, 1) if utenti_unici > 0 else 0

    col1.metric("Utenti Unici", utenti_unici)
    col2.metric("Totale Risposte", tot_risposte)
    col3.metric("Canzoni Uniche", canzoni_uniche)
    col4.metric("Media Risposte/Utente", media_risposte)
    
    st.divider()

    col_left, col_right = st.columns(2)
    
    chart_layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    with col_left:
        st.subheader("Distribuzione per Genere")
        gender_counts = df_users['gender'].value_counts().reset_index()
        gender_counts.columns = ['Genere', 'Conteggio']
        
        fig_gender = px.pie(
            gender_counts, 
            names='Genere', 
            values='Conteggio',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_gender.update_layout(**chart_layout, height=300)
        st.plotly_chart(fig_gender, use_container_width=True)

    with col_right:
        st.subheader("Distribuzione per Età")
        age_counts = df_users['age'].value_counts().sort_index().reset_index()
        age_counts.columns = ['Età', 'Conteggio']
        
        fig_age = px.bar(
            age_counts,
            x='Età',
            y='Conteggio',
            color_discrete_sequence=['#3498db']
        )
        fig_age.update_layout(**chart_layout, height=300, xaxis_title="Età", yaxis_title="Numero Utenti")
        st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    st.subheader("Numero di Risposte per Utente")
    responses_per_user = df_users['num_responses'].value_counts().sort_index().reset_index()
    responses_per_user.columns = ['Numero Risposte', 'Numero Utenti']
    
    fig_responses = px.bar(
        responses_per_user,
        x='Numero Risposte',
        y='Numero Utenti',
        color_discrete_sequence=['#e74c3c']
    )
    fig_responses.update_layout(**chart_layout, height=350)
    st.plotly_chart(fig_responses, use_container_width=True)

    st.divider()

    st.subheader("Informazioni Aggiuntive")
    utenti_attivi = df_users[df_users['num_responses'] > 0].shape[0]
    st.markdown(f"""
        Su un totale di **{utenti_unici}** utenti registrati, 
        i dati analizzati derivano dai **{utenti_attivi}** utenti che hanno effettivamente 
        interagito con le canzoni.
    """)

# ============================================================================
# SECTION 2: SPIDER CHARTS
# ============================================================================

elif menu == "🕷️ Spider Charts":
    st.header("Highest AVG per Song")
    st.write("Top 5 canzoni per categoria emotiva basate sui punteggi medi degli utenti")
    
    emotions_list = ["amusement", "anger", "sadness", "contentment", "disgust", "awe", "fear", "excitement"]
    
    col_img, col_tab = st.columns([1, 1.2])

    with col_img:
        st.subheader("Modello Emotivo")
        st.image("plutchik.png", caption="Ruota delle Emozioni di Plutchik", use_container_width=True)

    with col_tab:
        st.subheader("Top 5 per Categoria")
        
        summary_stats = df_responses.groupby('song_path').agg({
            e: 'mean' for e in emotions_list
        }).reset_index()

        summary_rows = []
        for emotion in emotions_list:
            top_5_songs = summary_stats.nlargest(5, emotion)
            names_list = ", ".join([row['song_path'].split('/')[-1] for _, row in top_5_songs.iterrows()])
            
            summary_rows.append({
                "EMOZIONE": emotion.upper(),
                "TOP 5 BRANI (Ranked)": names_list
            })
        
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.divider()

    for emotion in emotions_list:
        st.markdown(f"## {emotion.upper()}")
        st.markdown(f"<div style='background-color: {emotion_colors.get(emotion, '#ccc')}; height: 4px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        
        song_stats = df_responses.groupby('song_path').agg({
            e: 'mean' for e in emotions_list
        }).reset_index()
        
        user_counts = df_responses.groupby('song_path')['user_id'].nunique().reset_index()
        song_stats = song_stats.merge(user_counts, on='song_path')
        song_stats.rename(columns={'user_id': 'num_users'}, inplace=True)
        
        top_5 = song_stats.nlargest(5, emotion)
        
        cols = st.columns(5)
        
        for idx, (index, song_row) in enumerate(top_5.iterrows()):
            with cols[idx]:
                song_full_name = song_row['song_path'].split('/')[-1]
                song_path_clean = song_row['song_path'].replace('songs/', '')
                song_path_for_match = song_path_clean.replace('/', '\\')
                song_name = (song_full_name[:15] + '..') if len(song_full_name) > 17 else song_full_name
                
                user_values = [song_row[e] for e in emotions_list]
                
                original_values = None
                if song_path_for_match in original_emotions:
                    original_values = [original_emotions[song_path_for_match][e] for e in emotions_list]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=user_values,
                    theta=emotions_list,
                    fill='toself',
                    line=dict(color=emotion_colors.get(emotion, "#1f77b4"), width=2),
                    name='Utenti',
                    showlegend=False
                ))
                
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
                st.markdown(f"<p style='text-align: center; font-size: 10px; color: gray;'>Users: {int(song_row['num_users'])} | Score: {song_row[emotion]:.2f}</p>", unsafe_allow_html=True)
        
        st.markdown("---")

    # ========================================================================
    # INTER-RATER AGREEMENT ANALYSIS
    # ========================================================================
    
    st.header("Inter-Rater Agreement Analysis")
    st.write("Analisi delle canzoni ascoltate da più utenti, confrontando le risposte individuali con i valori emozionali originali")
    
    song_user_counts = df_responses.groupby('song_path')['user_id'].nunique().reset_index()
    song_user_counts.rename(columns={'user_id': 'num_users'}, inplace=True)
    songs_multi_users = song_user_counts[song_user_counts['num_users'] >= 2].sort_values('num_users', ascending=False)
    
    for idx, song_row in songs_multi_users.iterrows():
        song_path = song_row['song_path']
        song_name = song_path.split('/')[-1]
        num_users = int(song_row['num_users'])
        
        user_responses = df_responses[df_responses['song_path'] == song_path]
        
        emotion_stats = {}
        for e in emotions_list:
            emotion_stats[e] = {
                'mean': user_responses[e].mean(),
                'std': user_responses[e].std()
            }
        
        song_path_clean = song_path.replace('songs/', '')
        song_path_for_match = song_path_clean.replace('/', '\\')
        original_values = original_emotions.get(song_path_for_match, {})
        
        with st.expander(f"**{song_name}** - {num_users} utenti", expanded=(idx == 0)):
            
            plot_data = []
            for user_idx, (_, user_row) in enumerate(user_responses.iterrows()):
                for e in emotions_list:
                    plot_data.append({
                        'Emotion': e,
                        'Score': user_row[e],
                        'Source': f'User {user_idx + 1}',
                        'Type': 'User Response'
                    })
            
            df_plot = pd.DataFrame(plot_data)
            fig = go.Figure()
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            for user_idx in range(num_users):
                user_data = df_plot[df_plot['Source'] == f'User {user_idx + 1}']
                fig.add_trace(go.Bar(
                    name=f'Utente {user_idx + 1}',
                    x=user_data['Emotion'],
                    y=user_data['Score'],
                    marker_color=colors[user_idx % len(colors)]
                ))
            
            if original_values:
                original_scores = [original_values.get(e, 0) for e in emotions_list]
                fig.add_trace(go.Scatter(
                    name='Valori Originali',
                    x=emotions_list,
                    y=original_scores,
                    mode='lines+markers',
                    line=dict(color='gray', width=3, dash='dash'),
                    marker=dict(size=8, color='gray')
                ))
            
            fig.update_layout(
                title=f"Risposte Individuali vs Valori Originali - {song_name}",
                xaxis_title="Emozione",
                yaxis_title="Punteggio",
                yaxis=dict(range=[0, 1]),
                barmode='group',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"agreement_{idx}")
            
            st.subheader("Statistiche")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Media ± Deviazione Standard per Emozione:**")
                stats_rows = []
                for e in emotions_list:
                    stats_rows.append({
                        'Emozione': e.capitalize(),
                        'Media': f"{emotion_stats[e]['mean']:.3f}",
                        'Dev Std': f"{emotion_stats[e]['std']:.3f}"
                    })
                st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)
            
            with col2:
                if original_values:
                    st.markdown("**Confronto con Originale:**")
                    comparison_rows = []
                    for e in emotions_list:
                        diff = abs(emotion_stats[e]['mean'] - original_values.get(e, 0))
                        comparison_rows.append({
                            'Emozione': e.capitalize(),
                            'Originale': f"{original_values.get(e, 0):.3f}",
                            'Differenza': f"{diff:.3f}"
                        })
                    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("Dati emozionali originali non disponibili per questa canzone")
