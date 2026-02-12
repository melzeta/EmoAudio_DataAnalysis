import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================================
# SECTION 1: DATASET OVERVIEW
# ============================================================================

def render(df_users: pd.DataFrame, df_responses: pd.DataFrame, chart_layout: dict):
    st.header("Esplorazione Generale dei Dati")

    col_left, col_right = st.columns([1, 1.5])

    utenti_unici = df_users["user_id"].nunique()
    tot_risposte = len(df_responses)
    canzoni_uniche = df_responses["song_path"].nunique()
    media_risposte = round(tot_risposte / utenti_unici, 1) if utenti_unici > 0 else 0
    utenti_attivi = df_users[df_users["num_responses"] > 0].shape[0]

    with col_left:
        st.subheader("Metriche Principali")
        st.metric("Utenti Unici", utenti_unici)
        st.metric("Totale Risposte", tot_risposte)
        st.metric("Canzoni Uniche", canzoni_uniche)
        st.metric("Media Risposte/Utente", media_risposte)

        st.divider()

        st.subheader("Informazioni Aggiuntive")
        st.markdown(f"""
            Su un totale di **{utenti_unici}** utenti registrati, 
            i dati analizzati derivano dai **{utenti_attivi}** utenti che hanno effettivamente 
            interagito con le canzoni. Non ci sono informazioni di genere o et0 per gli utenti.
        """)

    with col_right:
        st.subheader("Numero di Risposte per Utente")
        responses_per_user = df_users["num_responses"].value_counts().sort_index().reset_index()
        responses_per_user.columns = ["Numero Risposte", "Numero Utenti"]

        fig_responses = px.bar(
            responses_per_user,
            x="Numero Risposte",
            y="Numero Utenti",
            color_discrete_sequence=["#e74c3c"],
        )
        fig_responses.update_layout(**chart_layout, height=450)
        st.plotly_chart(fig_responses, use_container_width=True)
