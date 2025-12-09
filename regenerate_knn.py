from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pickle
import pandas as pd

# Charger tes données d'entraînement
df = pd.read_excel("data.xlsx")  # ou ton dataset original
X = df[["Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]]
y = df["Temperature (C)"]

# Créer et entraîner le scaler
knn_scaler = StandardScaler()
X_scaled = knn_scaler.fit_transform(X)

# Créer et entraîner le modèle KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_scaled, y)

# Sauvegarder le scaler
with open("scaler_knn.pkl", "wb") as f:
    pickle.dump(knn_scaler, f)

# Sauvegarder le modèle KNN
with open("model_knn.pkl", "wb") as f:
    pickle.dump(knn_model, f)

print("✅ Modèle KNN et scaler régénérés avec succès !")
