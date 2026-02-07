import plotly.graph_objects as go


def build_spider_chart(
    emotions_list,
    user_values,
    emotion_color,
    song_name,
    original_values=None,
):
    """
    Utility opzionale per creare uno spider chart.
    Non viene usata nella versione attuale dell'app (serve solo come componente riutilizzabile).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=emotions_list,
        fill="toself",
        line=dict(color=emotion_color, width=2),
        name="Utenti",
        showlegend=False
    ))

    if original_values:
        fig.add_trace(go.Scatterpolar(
            r=original_values,
            theta=emotions_list,
            fill="toself",
            line=dict(color="gray", dash="dash", width=2),
            fillcolor="rgba(128, 128, 128, 0.15)",
            name="Originale",
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
        title=dict(
            text=f"<b>{song_name}</b>",
            font=dict(size=11),
            y=0.92,
            x=0.5,
            xanchor="center"
        )
    )

    return fig