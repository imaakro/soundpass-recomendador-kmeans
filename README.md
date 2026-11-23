# SoundPass · Recomendador de conciertos con K-means

Módulo de inteligencia artificial para **SoundPass**, la aplicación de venta de boletas para conciertos que desarrollamos
en la materia de Desarrollo de Aplicaciones en la Nube. Este módulo descubre el **perfil de gusto musical** de una
persona a partir de canciones que le gustan y le recomienda **conciertos** cuyos artistas suenan parecido.
Proyecto de la **Actividad 3 de Inteligencia Artificial I** (Grupo 4): llevar un modelo de Machine Learning desde
el notebook hasta una aplicación desplegada en la nube.

## Descripción

Una plataforma de boletería tiene muchos conciertos y cada persona solo ve unos pocos. Si le mostramos a un fan del
reguetón un concierto de metal, lo más probable es que no compre. El problema es saber qué le gusta a cada persona sin
pedirle que llene un formulario largo.

La solución usa **aprendizaje no supervisado**. Spotify describe cada canción con números que miden cómo suena: qué
tan bailable es, cuánta energía tiene, qué tan acústica es y qué tan alegre suena. Con esos números, el algoritmo
**K-means** agrupa 28.352 canciones reales en **perfiles musicales**, sin que nadie le diga de antemano qué género es
cada una. Después ubicamos a cada artista del catálogo de conciertos en esos mismos perfiles.

En la app, la persona escoge tres o más canciones que le gustan (o mueve cuatro controles si no encuentra sus
canciones). El sistema promedia cómo suenan esas canciones, le dice a qué perfil pertenece y le muestra los conciertos
más afines, con un porcentaje de afinidad.

## Demostración

🔗 **App desplegada:** _pendiente (Streamlit Community Cloud)_ <!-- reemplazar por la URL https://....streamlit.app -->

Capturas de la prueba local en [`docs/evidencias/`](docs/evidencias/).

## Algoritmo utilizado

- **K-means** (`sklearn.cluster.KMeans`, `k = 3`, `n_init = 10`, `random_state = 42`) sobre los 2 componentes
  principales (**PCA**) de cuatro características de audio.
- **Por qué es apropiado:** no tenemos una etiqueta de "gusto correcto" para cada canción; queremos descubrir grupos
  de canciones que suenan parecido. K-means es simple, rápido con decenas de miles de canciones y cada centro se
  interpreta como "la canción típica" del perfil. Para un usuario nuevo basta con buscar el centro más cercano, lo que
  hace que la recomendación sea instantánea en la app.
- **Por qué PCA antes de K-means:** con las cuatro características directas la silueta se queda alrededor de 0,3.
  PCA resume las cuatro en dos ejes que conservan el 78 % de la información (eje 1: intenso ↔ acústico; eje 2:
  oscuro ↔ alegre) y quita ruido, con lo que los grupos quedan mejor separados.
- **Elección de k:** método del codo y coeficiente de silueta para k entre 2 y 8. Con k = 3 la silueta es la más alta
  (0,424) y los tres perfiles se interpretan fácilmente; con k ≥ 4 baja de 0,4.

### Métricas de desempeño

| Métrica | Entrenamiento (80 %) | Prueba (20 %) | Interpretación |
|---|---|---|---|
| Silueta | 0.423 | **0.423** | > 0.4 (mínimo pedido); grupos bien separados |
| Davies-Bouldin | 0.800 | 0.799 | menor es mejor |
| Calinski-Harabasz | 20 405 | 5 217 | mayor es mejor (depende del tamaño de la muestra) |

### Perfiles encontrados

| Perfil | Canciones | Cómo suena (promedio) | Artistas del catálogo |
|---|---|---|---|
| Bailable y alegre | 12 221 | bailable 0,71 · energía 0,74 · alegría 0,72 | Karol G, J Balvin, Shakira, Carlos Vives, Bruno Mars |
| Intenso y enérgico | 11 295 | energía 0,75 · alegría 0,32 · acústico 0,06 | Metallica, Guns N' Roses, Martin Garrix, Coldplay |
| Tranquilo y acústico | 4 836 | acústico 0,57 · energía 0,46 | Billie Eilish, Daniel Caesar, Frank Ocean, Lewis Capaldi |

El modelo nunca vio el género, pero el 67 % de las canciones de EDM cayó en *Intenso y enérgico* y el 61 % de las de
música latina en *Bailable y alegre*.

### Cómo se recomienda un concierto

1. El gusto del usuario es el promedio de las características de las canciones que escogió.
2. Ese gusto se transforma con el mismo PCA y K-means le asigna el perfil del centro más cercano.
3. Cada artista del catálogo está ubicado en el mismo plano (promedio de sus canciones en el dataset).
4. Se muestran primero los conciertos del mismo perfil, ordenados por distancia; la afinidad es
   `100 × (1 − distancia / 0,8)`.

## Dataset

- **Fuente:** canciones de Spotify obtenidas con la Spotify Web API (paquete `spotifyr`) y publicadas por el proyecto
  [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21) (2020-01-21).
  Archivo original en `data/raw/spotify_songs.csv`.
- **Tamaño:** 32.833 filas y 23 columnas; 28.352 canciones únicas después de quitar 5 filas sin nombre y 4.477
  repetidas (la misma canción en varias playlists). Seis géneros: pop, rap, rock, latin, r&b y edm.
- **Features usadas:** `danceability` (bailable), `energy` (energía), `acousticness` (acústico), `valence` (alegría),
  todas entre 0 y 1. Se descartaron `speechiness`, `instrumentalness` y `liveness` porque casi todas las canciones
  están cerca de 0, y `loudness` porque está muy correlacionada con `energy`.
- **Catálogo de conciertos:** `data/raw/conciertos_soundpass.csv`, 24 conciertos **de ejemplo** (artistas reales;
  eventos, fechas y precios de prueba para el proyecto).

## Instalación local

### Requisitos

- Python 3.11 o superior
- pip

### Pasos

```bash
git clone https://github.com/USUARIO/soundpass-recomendador-kmeans.git
cd soundpass-recomendador-kmeans
python -m venv venv
# Windows: venv\Scripts\activate   ·   Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### Ejecutar la aplicación

```bash
streamlit run app/app.py
```

Se abre en `http://localhost:8501`.

### Volver a entrenar el modelo

Ejecutar de principio a fin `notebooks/01_training.ipynb` (Jupyter, VS Code o Google Colab después de clonar el
repositorio). Regenera `models/`, `data/processed/` y las figuras de `docs/figuras/`.

## Uso

1. **Elegir canciones:** escribe el nombre de una canción o artista, escoge al menos 3 y pulsa *Ver mis conciertos*.
   La app muestra tu perfil musical y los 5 conciertos más afines, con lugar, fecha, precio y porcentaje de afinidad.
2. **Ajustar mi gusto:** si no encuentras tus canciones, mueve los controles *Bailable*, *Energía*, *Acústico* y
   *Alegría* (de 0 a 1).
3. **¿Cómo lo calculó el modelo?:** compara tu gusto con el típico de tu perfil y muestra la distancia a cada centro.
4. **Los perfiles:** explica los tres perfiles y qué conciertos del catálogo hay en cada uno.
5. Si escoges menos de 3 canciones, la app muestra un mensaje de error y no recomienda.

## Estructura del proyecto

```
soundpass-recomendador-kmeans/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .streamlit/config.toml
├── data/
│   ├── raw/
│   │   ├── spotify_songs.csv          # dataset original de Spotify
│   │   └── conciertos_soundpass.csv   # catálogo de conciertos de prueba
│   └── processed/
│       ├── canciones_app.csv          # canciones populares para el buscador de la app
│       └── conciertos_con_perfil.csv  # cada concierto con su perfil y su posición
├── notebooks/
│   └── 01_training.ipynb              # EDA, preprocesamiento, entrenamiento, evaluación y serialización
├── models/
│   ├── kmeans_model.pkl               # K-means entrenado (3 centros)
│   ├── pca.pkl                        # PCA ajustado con train
│   ├── perfiles.json                  # nombre y descripción de cada perfil
│   └── metricas.json
├── src/
│   ├── preprocessing.py               # limpieza, vector de gusto y validaciones
│   └── predict.py                     # perfil del usuario y recomendación de conciertos
├── app/
│   └── app.py                         # aplicación Streamlit
└── docs/
    ├── evidencias/                    # capturas de la app funcionando
    └── figuras/                       # gráficas del notebook
```

El código está separado en **entrenamiento** (notebook), **lógica reutilizable** (`src/`, usada por el notebook y por
la app sin duplicar código) y **presentación** (`app/`).

## Autores

- Camilo Montenegro
- Esteban Parra

Inteligencia Artificial I · Fundación Universitaria Los Libertadores · Docente: Gonzalo Novoa

## Licencia

MIT License. Ver [`LICENSE`](LICENSE).

## Referencias

[1] rfordatascience, "Spotify Songs," *TidyTuesday*, 21 ene. 2020. [En línea]. Disponible:
https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21

[2] Spotify, "Web API Reference: Get Track's Audio Features." [En línea]. Disponible:
https://developer.spotify.com/documentation/web-api/reference/get-audio-features

[3] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825-2830, 2011.

[4] Streamlit Inc., "Streamlit Community Cloud documentation." [En línea]. Disponible:
https://docs.streamlit.io/deploy/streamlit-community-cloud
