import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import EMOTIONS_LIST, EMOTION_COLORS
from data.loaders import load_original_emotions_by_filename, calculate_similarity_top3


def show(df_responses=None):
    st.header("Analisi Similarity: Input vs Output")
    
    st.markdown("""
    **Metodo:** Prodotto interno (dot product) tra valori originali e risposte medie utenti,
    calcolato sulle **top 3 emozioni** dei valori originali.
    
    **Interpretazione:**
    - Similarity >= 0.6: Buon match
    - Similarity 0.4-0.6: Match medio
    - Similarity < 0.4: Scarso match
    """)
    
    # Carica dati originali
    original_data = load_original_emotions_by_filename()
    
    if not original_data:
        st.error("Impossibile caricare i dati originali dal CSV.")
        return
    
    st.divider()
    
    # Ottieni df_responses dal session_state se non passato
    if df_responses is None:
        df_responses = st.session_state.get("df_responses")
    if df_responses is None:
        st.error("Dati delle risposte utenti non disponibili.")
        return
    
    # Aggrega per canzone: calcola medie delle emozioni e conta utenti
    song_aggregates = df_responses.groupby('song_path').agg({
        **{e: 'mean' for e in EMOTIONS_LIST},
        'user_id': 'nunique'
    }).reset_index()
    song_aggregates.rename(columns={'user_id': 'num_users'}, inplace=True)
    
    # Calcola similarity per ogni canzone
    similarity_results = []
    
    for _, song in song_aggregates.iterrows():
        song_path = song['song_path']
        filename = song_path.split('/')[-1]
        
        # Verifica se abbiamo i dati originali
        if filename not in original_data:
            continue
        
        original_emotions = original_data[filename]
        user_emotions_avg = {e: song[e] for e in EMOTIONS_LIST}
        
        # Calcola similarity
        similarity, top3 = calculate_similarity_top3(original_emotions, user_emotions_avg)
        
        if similarity is not None:
            similarity_results.append({
                'song_name': filename,
                'song_path': song_path,
                'num_users': int(song['num_users']),
                'similarity_score': similarity,
                'top3_emotions': ', '.join(top3),
                'match_quality': 'Buon Match' if similarity >= 0.6 else ('Medio' if similarity >= 0.4 else 'Scarso')
            })
    
    df_similarity = pd.DataFrame(similarity_results)
    
    if len(df_similarity) == 0:
        st.warning("Nessuna canzone trovata con dati originali corrispondenti.")
        return
    
    # Statistiche complessive
    st.subheader("Risultati Complessivi")
    
    total = len(df_similarity)
    good = len(df_similarity[df_similarity['similarity_score'] >= 0.6])
    medium = len(df_similarity[(df_similarity['similarity_score'] >= 0.4) & (df_similarity['similarity_score'] < 0.6)])
    poor = len(df_similarity[df_similarity['similarity_score'] < 0.4])
    success_rate = (good / total) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Totale Canzoni", total)
    col2.metric("Buon Match", f"{good} ({success_rate:.1f}%)")
    col3.metric("Match Medio", medium)
    col4.metric("Scarso Match", poor)
    
    # Messaggio valutazione
    if success_rate >= 70:
        st.success(f"Ottimo! {good}/{total} canzoni ({success_rate:.1f}%) mostrano buon match.")
    elif success_rate >= 50:
        st.info(f"Discreto. {good}/{total} canzoni ({success_rate:.1f}%) mostrano buon match.")
    else:
        st.warning(f"Solo {good}/{total} canzoni ({success_rate:.1f}%) mostrano buon match.")
    
    st.divider()
    
    # Istogramma distribuzione
    st.subheader("Distribuzione Similarity Scores")
    
    fig_hist = px.histogram(
        df_similarity,
        x='similarity_score',
        nbins=20,
        labels={'similarity_score': 'Similarity Score', 'count': 'Numero Canzoni'},
        color_discrete_sequence=['#4A90E2']
    )
    fig_hist.add_vline(x=0.6, line_dash="dash", line_color="green", annotation_text="Buon Match (>=0.6)")
    fig_hist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medio (>=0.4)")
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()
    
    # Tabella dettagliata
    st.subheader("Dettaglio per Canzone")
    
    df_sorted = df_similarity.sort_values('similarity_score', ascending=False)
    
    st.dataframe(
        df_sorted[['song_name', 'num_users', 'similarity_score', 'top3_emotions', 'match_quality']],
        hide_index=True,
        use_container_width=True,
        column_config={
            'song_name': 'Canzone',
            'num_users': 'Utenti',
            'similarity_score': st.column_config.NumberColumn('Similarity', format="%.3f"),
            'top3_emotions': 'Top 3 Emozioni Input',
            'match_quality': 'Valutazione'
        }
    )
    
    st.divider()
    
    # Best e Worst matches
    col_best, col_worst = st.columns(2)
    
    with col_best:
        st.subheader("Top 5 Best Matches")
        top_5 = df_sorted.head(5)
        for _, row in top_5.iterrows():
            st.markdown(f"""
            **{row['song_name']}**  
            Similarity: `{row['similarity_score']:.3f}` | Users: {row['num_users']} | Top 3: {row['top3_emotions']}
            """)
    
    with col_worst:
        st.subheader("Top 5 Worst Matches")
        bottom_5 = df_sorted.tail(5)
        for _, row in bottom_5.iterrows():
            st.markdown(f"""
            **{row['song_name']}**  
            Similarity: `{row['similarity_score']:.3f}` | Users: {row['num_users']} | Top 3: {row['top3_emotions']}
            """)