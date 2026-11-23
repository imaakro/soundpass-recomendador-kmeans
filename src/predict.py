"""Cargar el modelo entrenado, asignar el perfil musical y recomendar conciertos."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURES

RAIZ = Path(__file__).resolve().parent.parent


def cargar_artefactos(raiz=RAIZ):
    """Carga lo que dejó el notebook: K-means, PCA, nombres de perfiles y artistas de los conciertos."""
    raiz = Path(raiz)
    with open(raiz / "models" / "perfiles.json", encoding="utf-8") as f:
        perfiles = json.load(f)
    return {
        "modelo": joblib.load(raiz / "models" / "kmeans_model.pkl"),
        "pca": joblib.load(raiz / "models" / "pca.pkl"),
        "perfiles": perfiles,
        "conciertos": pd.read_csv(raiz / "data" / "processed" / "conciertos_con_perfil.csv"),
    }


def a_plano(artefactos, X):
    """Lleva vectores de FEATURES al plano de 2 componentes donde se entrenó K-means."""
    return artefactos["pca"].transform(np.asarray(X[FEATURES], dtype=float))


def perfil_usuario(artefactos, vector):
    """Devuelve el perfil (cluster) del usuario y su distancia a cada centro."""
    punto = a_plano(artefactos, vector)
    modelo = artefactos["modelo"]
    cluster = int(modelo.predict(punto)[0])
    distancias = np.linalg.norm(modelo.cluster_centers_ - punto, axis=1)
    info = artefactos["perfiles"][str(cluster)]
    return {
        "cluster": cluster,
        "nombre": info["nombre"],
        "descripcion": info["descripcion"],
        "punto": punto[0],
        "distancias": {artefactos["perfiles"][str(i)]["nombre"]: float(d) for i, d in enumerate(distancias)},
    }


def recomendar_conciertos(artefactos, vector, n=5):
    """Ordena los conciertos de SoundPass por qué tan cerca está el sonido del artista
    del gusto del usuario. Primero los del mismo perfil, luego por distancia."""
    perfil = perfil_usuario(artefactos, vector)
    conciertos = artefactos["conciertos"].copy()
    puntos = conciertos[["pc1", "pc2"]].to_numpy()
    conciertos["distancia"] = np.linalg.norm(puntos - perfil["punto"], axis=1)
    conciertos["mismo_perfil"] = conciertos["cluster"] == perfil["cluster"]
    # Afinidad de 0 a 100 %: 100 si el artista suena exactamente como tu gusto.
    conciertos["afinidad"] = (100 * np.clip(1 - conciertos["distancia"] / 0.8, 0, 1)).round(0)
    ordenados = conciertos.sort_values(["mismo_perfil", "distancia"], ascending=[False, True])
    return perfil, ordenados.head(n).reset_index(drop=True)
