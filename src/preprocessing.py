"""Funciones de preprocesamiento: canciones de Spotify -> vector de gusto musical.

Todo son funciones (sin clases) para poder leerlo de arriba hacia abajo.
"""

from pathlib import Path

import pandas as pd

# Características de audio de Spotify que usa el modelo (todas van de 0 a 1).
FEATURES = [
    "danceability",  # qué tan bailable es
    "energy",        # qué tan intensa / enérgica
    "acousticness",  # qué tan acústica (guitarra, piano) vs. electrónica
    "valence",       # qué tan alegre (1) o triste (0) suena
]

NOMBRES_FEATURES = {
    "danceability": "Bailable",
    "energy": "Energía",
    "acousticness": "Acústico",
    "valence": "Alegría",
}

COLUMNAS_INFO = ["track_id", "track_name", "track_artist", "track_popularity", "playlist_genre"]


def cargar_canciones(ruta):
    """Lee el CSV original de canciones de Spotify."""
    return pd.read_csv(Path(ruta))


def limpiar_canciones(df):
    """Deja una fila por canción y quita filas sin nombre o sin características.

    El dataset repite canciones que aparecen en varias playlists (mismo track_id).
    """
    limpio = df.dropna(subset=["track_name", "track_artist"] + FEATURES)
    limpio = limpio.drop_duplicates(subset="track_id")
    limpio = limpio[(limpio[FEATURES] >= 0).all(axis=1) & (limpio[FEATURES] <= 1).all(axis=1)]
    return limpio.reset_index(drop=True)


def vector_desde_canciones(canciones):
    """El gusto de una persona = el promedio de las características de las canciones
    que dijo que le gustan. Devuelve un DataFrame de una fila con FEATURES."""
    return canciones[FEATURES].mean().to_frame().T


def vector_desde_controles(bailable, energia, acustico, alegria):
    """Arma el vector de gusto a partir de los controles deslizantes de la app (0 a 1)."""
    return pd.DataFrame([[bailable, energia, acustico, alegria]], columns=FEATURES)


def validar_controles(bailable, energia, acustico, alegria):
    """Revisa que los valores estén entre 0 y 1. Devuelve una lista de errores."""
    errores = []
    for nombre, valor in [("Bailable", bailable), ("Energía", energia),
                          ("Acústico", acustico), ("Alegría", alegria)]:
        if valor is None or not 0 <= valor <= 1:
            errores.append(f"«{nombre}» debe estar entre 0 y 1.")
    return errores


def validar_seleccion(canciones_elegidas, minimo=3):
    """Pide al menos `minimo` canciones para que el promedio represente un gusto."""
    if len(canciones_elegidas) < minimo:
        return [f"Escoge al menos {minimo} canciones que te gusten (llevas {len(canciones_elegidas)})."]
    return []


def perfil_por_artista(canciones):
    """Promedio de las características de audio de cada artista."""
    agrupado = canciones.groupby("track_artist")
    perfil = agrupado[FEATURES].mean()
    perfil["canciones"] = agrupado.size()
    perfil["genero_principal"] = agrupado["playlist_genre"].agg(lambda s: s.value_counts().idxmax())
    return perfil
