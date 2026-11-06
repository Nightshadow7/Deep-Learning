import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# Cargar UrbanSound8K
us8k = pd.read_csv("UrbanSound8K.csv")
us8k["dataset_origen"] = "UrbanSound8K"
us8k["duracion_segundos"] = us8k["end"] - us8k["start"]
us8k["salience_str"] = us8k["salience"].map({1: "Primer Plano (1)", 2: "Fondo (2)"})

# Cargar ESC-50
esc50 = pd.read_csv("esc50.csv") # Asumiendo que descargaste el meta/esc50.csv del repo
# Filtrar clases tácticas de ESC-50 (Ej: 42 = siren, 43 = glass_breaking)
clases_criticas = ["siren", "glass_breaking", "car_crash"] 
esc50 = esc50[esc50["category"].isin(clases_criticas)].copy()

# 3. Imputación y Estandarización de ESC-50
esc50["class"] = esc50["category"]
esc50["dataset_origen"] = "Aporte ESC-50"
esc50["duracion_segundos"] = 5.0 # ESC-50 garantiza audios de 5 segundos exactos
esc50["salience"] = 1 # Imputación: Asumimos que los audios de estudio son primer plano
esc50["salience_str"] = "Primer Plano (1)"

# Fusión Final (Data Fusion)
df_fusion = pd.concat([us8k, esc50], ignore_index=True)

# GRÁFICO 1: Cantidad de Audios y Aporte ESC-50

plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_fusion, y="class", hue="dataset_origen", 
    order=df_fusion["class"].value_counts().index,
    palette={"UrbanSound8K": "#4C72B0", "Aporte ESC-50": "#C44E52"}
)
plt.title("Gráfico 1: Distribución de Clases y Aporte Táctico de ESC-50", fontweight="bold")
plt.xlabel("Cantidad de Muestras")
plt.ylabel("Clase Acústica")
plt.legend(title="Fuente de Datos")
plt.tight_layout()
plt.savefig("grafico_1_cantidad_sonidos.png")
plt.show()

# GRÁFICO 2: Variabilidad de Duración Combinada

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_fusion, x="duracion_segundos", y="class", 
    hue="dataset_origen", palette="muted"
)
plt.title("Gráfico 2: Variabilidad de Duración Temporal (Segundos)", fontweight="bold")
plt.xlabel("Duración del Audio (s)")
plt.legend(title="Fuente de Datos")
plt.tight_layout()
plt.savefig("grafico_2_duracion_sonidos.png")
plt.show()

# GRÁFICO 3: Prominencia (Cerca vs Lejos) Estandarizada

plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_fusion, x="class", hue="salience_str", 
    order=df_fusion["class"].value_counts().index,
    palette="viridis"
)
plt.title("Gráfico 3: Prominencia del Sonido (Salience Imputado)", fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Cantidad de Muestras")
plt.legend(title="Nivel de Prominencia")
plt.tight_layout()
plt.savefig("grafico_3_cerca_vs_lejos.png")
plt.show()