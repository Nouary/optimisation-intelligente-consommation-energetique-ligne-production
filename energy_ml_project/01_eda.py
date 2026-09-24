import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1) Charger les données (chemin relatif au script)
csv_path = Path(__file__).resolve().parent / "data" / "steel_line_energy_dataset.csv"
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nInfo:\n")
print(df.info())

# 2) Convertir timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 3) Aperçu
print("\nHead:\n", df.head())

# 4) Valeurs manquantes
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))

# 5) Statistiques descriptives
print("\nDescribe:\n", df.describe(include="all"))

# ---------- VISUALISATIONS ----------

# A) Distribution de la cible
plt.figure(figsize=(7,4))
sns.histplot(df["energy_consumption_kwh_15min"], bins=30, kde=True)
plt.title("Distribution - Energy Consumption (kWh/15min)")
plt.xlabel("Energy consumption (kWh/15min)")
plt.tight_layout()
plt.show()

# B) Charge vs Consommation
plt.figure(figsize=(7,4))
sns.scatterplot(
    data=df,
    x="operational_load_pct",
    y="energy_consumption_kwh_15min",
    alpha=0.4
)
plt.title("Operational Load vs Energy Consumption")
plt.xlabel("Operational load (%)")
plt.ylabel("Energy consumption (kWh/15min)")
plt.tight_layout()
plt.show()

# C) Température four vs Consommation
plt.figure(figsize=(7,4))
sns.scatterplot(
    data=df,
    x="furnace_temp_c",
    y="energy_consumption_kwh_15min",
    alpha=0.4
)
plt.title("Furnace Temperature vs Energy Consumption")
plt.xlabel("Furnace temperature (°C)")
plt.ylabel("Energy consumption (kWh/15min)")
plt.tight_layout()
plt.show()

# D) Corrélation (numériques uniquement)
num_df = df.select_dtypes(include=np.number)
corr = num_df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.show()

# E) Série temporelle de la consommation
df_ts = df.set_index("timestamp")["energy_consumption_kwh_15min"]

plt.figure(figsize=(12,4))
df_ts.plot()
plt.title("Energy consumption over time")
plt.ylabel("kWh/15min")
plt.tight_layout()
plt.show()

# F) Week-end vs Semaine
plt.figure(figsize=(7,4))
sns.boxplot(data=df, x="weekend", y="energy_consumption_kwh_15min")
plt.title("Weekend vs Weekday Energy Consumption")
plt.xlabel("Weekend")
plt.ylabel("Energy consumption (kWh/15min)")
plt.tight_layout()
plt.show()

print("\nEDA finished successfully.")
