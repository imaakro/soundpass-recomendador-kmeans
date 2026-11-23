"""SoundPass · Recomendador de conciertos según tu perfil musical (K-means).

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.predict import cargar_artefactos, recomendar_conciertos  # noqa: E402
from src.preprocessing import (FEATURES, NOMBRES_FEATURES, validar_controles,  # noqa: E402
                               validar_seleccion, vector_desde_canciones, vector_desde_controles)

st.set_page_config(page_title="SoundPass · Tu perfil musical", page_icon="🎟️", layout="centered")

COLORES = {
    "Bailable y alegre": "#d69e2e",
    "Tranquilo y acústico": "#2f855a",
    "Intenso y enérgico": "#c53030",
}
N_RECOMENDADOS = 5


@st.cache_resource
def obtener_artefactos():
    return cargar_artefactos(RAIZ)


@st.cache_data
def obtener_canciones():
    canciones = pd.read_csv(RAIZ / "data" / "processed" / "canciones_app.csv")
    canciones = canciones.sort_values("track_popularity", ascending=False)
    canciones["etiqueta"] = canciones["track_name"] + " — " + canciones["track_artist"]
    return canciones.drop_duplicates("etiqueta").set_index("etiqueta")


def formato_pesos(valor):
    return "$" + f"{int(valor):,}".replace(",", ".")


def mostrar_resultado(vector, artefactos, canciones):
    perfil, recomendados = recomendar_conciertos(artefactos, vector, n=N_RECOMENDADOS)
    color = COLORES.get(perfil["nombre"], "#2b6cb0")
    st.markdown(
        f"<div style='border-left:8px solid {color};padding:0.8rem 1rem;background:rgba(0,0,0,0.03);"
        f"border-radius:6px;margin-top:0.5rem'><div style='font-size:0.9rem;opacity:0.7'>Tu perfil musical</div>"
        f"<div style='font-size:1.7rem;font-weight:700;color:{color}'>{perfil['nombre']}</div>"
        f"<div>{perfil['descripcion']}</div></div>",
        unsafe_allow_html=True,
    )

    st.subheader("🎟️ Conciertos recomendados para ti")
    for _, c in recomendados.iterrows():
        with st.container(border=True):
            izq, der = st.columns([3, 1])
            izq.markdown(f"**{c['artista']}** · {c['nombre_evento']}")
            izq.caption(f"📍 {c['lugar']}, {c['ciudad']} · 📅 {c['fecha']} · desde {formato_pesos(c['precio_desde_cop'])} COP")
            der.metric("Afinidad", f"{c['afinidad']:.0f} %")
            if not c["mismo_perfil"]:
                izq.caption(f"De otro perfil ({c['perfil']}), pero suena cercano a tu gusto.")

    with st.expander("¿Cómo lo calculó el modelo?"):
        tipico = canciones.groupby("perfil")[FEATURES].mean().loc[perfil["nombre"]]
        tabla = pd.DataFrame({"Tu gusto": vector.iloc[0][FEATURES], f"Típico de «{perfil['nombre']}»": tipico})
        st.dataframe(tabla.rename(index=NOMBRES_FEATURES).round(2), use_container_width=True)
        st.caption("Tu gusto se ubica en el plano donde se entrenó K-means y queda en el perfil del centro más cercano "
                   "(la barra más corta). Los conciertos se ordenan por qué tan cerca suena cada artista de tu gusto.")
        st.bar_chart(pd.Series(perfil["distancias"]).sort_values().rename("distancia al centro"), horizontal=True)


try:
    artefactos = obtener_artefactos()
    canciones = obtener_canciones()
except Exception as error:
    st.error(f"No se pudo cargar el modelo o los datos: {error}")
    st.stop()

metricas = json.loads((RAIZ / "models" / "metricas.json").read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Sobre el modelo")
    st.write("**K-means** (no supervisado) entrenado con 28.352 canciones reales de Spotify "
             "usando sus características de audio: bailable, energía, acústico y alegría.")
    st.metric("Perfiles musicales (k)", metricas["k"])
    st.metric("Silueta en datos de prueba", f"{metricas['silueta_test']:.3f}")
    st.caption("La silueta va de -1 a 1; más de 0.4 indica grupos bien separados.")
    st.divider()
    st.caption("Módulo de recomendación de **SoundPass**\n\nGrupo 4 · Inteligencia Artificial I\n\n"
               "Camilo Montenegro · Esteban Parra")

st.title("🎟️ SoundPass")
st.markdown("#### Descubre tu perfil musical y los conciertos que te van a gustar")

tab_canciones, tab_controles, tab_perfiles = st.tabs(["Elegir canciones", "Ajustar mi gusto", "Los perfiles"])

with tab_canciones:
    st.write("Escoge **al menos 3 canciones** que te gusten. Puedes escribir para buscar por canción o artista.")
    elegidas = st.multiselect("Canciones que te gustan", options=canciones.index.tolist(), max_selections=10,
                              placeholder="Ej.: Tusa, Blinding Lights, bad guy…")
    if st.button("Ver mis conciertos", type="primary", use_container_width=True, key="btn_canciones"):
        errores = validar_seleccion(elegidas)
        if errores:
            for e in errores:
                st.error(e)
        else:
            mostrar_resultado(vector_desde_canciones(canciones.loc[elegidas]), artefactos, canciones)

with tab_controles:
    st.write("¿No encuentras tus canciones? Dinos cómo te gusta la música (0 = nada, 1 = mucho).")
    c1, c2 = st.columns(2)
    bailable = c1.slider("💃 Bailable", 0.0, 1.0, 0.7, 0.05)
    energia = c2.slider("⚡ Energía", 0.0, 1.0, 0.7, 0.05)
    acustico = c1.slider("🎸 Acústico", 0.0, 1.0, 0.1, 0.05)
    alegria = c2.slider("😄 Alegría", 0.0, 1.0, 0.7, 0.05)
    if st.button("Ver mis conciertos", type="primary", use_container_width=True, key="btn_controles"):
        errores = validar_controles(bailable, energia, acustico, alegria)
        if errores:
            for e in errores:
                st.error(e)
        else:
            mostrar_resultado(vector_desde_controles(bailable, energia, acustico, alegria), artefactos, canciones)

with tab_perfiles:
    st.write("El modelo encontró estos tres perfiles en las canciones de Spotify. Así quedaron los conciertos del catálogo:")
    conciertos = artefactos["conciertos"]
    for info in artefactos["perfiles"].values():
        nombre = info["nombre"]
        with st.container(border=True):
            st.markdown(f"<span style='color:{COLORES.get(nombre)};font-weight:700;font-size:1.15rem'>{nombre}</span>",
                        unsafe_allow_html=True)
            st.write(info["descripcion"])
            artistas = conciertos.loc[conciertos["perfil"] == nombre, "artista"].tolist()
            st.caption("Conciertos: " + ", ".join(artistas))
    st.caption("Catálogo de prueba de SoundPass: los eventos, fechas y precios son de ejemplo para el proyecto.")
