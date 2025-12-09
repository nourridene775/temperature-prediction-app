import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# ------------------ CSS pour navbar ------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #4169E1;
    }
    [data-testid="stSidebar"] div {
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Fonction pour charger un pickle en toute sécurité ------------------
def load_pickle(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ Fichier manquant : {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {file_path} : {e}")
        return None

# ------------------ Charger les modèles ------------------
lr_model = load_pickle("model_linear_regression.pkl")
knn_model = load_pickle("model_knn.pkl")
knn_scaler = load_pickle("scaler_knn.pkl")

# ------------------ Sidebar (Navbar) ------------------
st.sidebar.title("Dashboard prédiction température")
menu = st.sidebar.radio("Aller à", ["Accueil", "Régression Linéaire", "KNN"])

# ------------------ Page Accueil ------------------
if menu == "Accueil":
    st.title("📊 Dashboard de Prédiction de Température")
    st.write("Bienvenue sur le dashboard de prédiction de température. Voici une analyse statistique basée sur vos données météo.")

    # Charger les données
    try:
        df = pd.read_excel("data.xlsx")
    except Exception as e:
        st.error(f"❌ Impossible de charger data.xlsx : {e}")
        df = pd.DataFrame()

    # Vérifier que le DataFrame contient les colonnes nécessaires
    if not df.empty and all(col in df.columns for col in ["Date", "Temperature (C)", "Humidity"]):
        st.subheader("📌 Aperçu des données")
        st.dataframe(df)

        # ------------------ Conversion sécurisée en datetime ------------------
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df[df['Date'].notna()]  # Supprimer lignes sans date valide

        if df.empty:
            st.error("❌ Aucune date valide trouvée dans la colonne 'Date'. Vérifiez le format de vos dates dans Excel.")
        else:
            # Extraire l'année de manière sûre
            df['Année'] = df['Date'].apply(lambda x: x.year if pd.notnull(x) else None)

            # Grouper par année et calculer les moyennes
            df_yearly = df.groupby('Année')[['Temperature (C)', 'Humidity']].mean().reset_index()

            # ------------------ Tracer Température et Humidité ------------------
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=df_yearly['Année'],
                y=df_yearly['Temperature (C)'],
                mode="lines+markers",
                name="Température",
                line=dict(color='red')
            ))
            fig_temp.add_trace(go.Scatter(
                x=df_yearly['Année'],
                y=df_yearly['Humidity'],
                mode="lines+markers",
                name="Humidité",
                line=dict(color='blue')
            ))
            fig_temp.update_layout(
                xaxis_title="Année",
                yaxis_title="Valeur",
                legend_title="Variables",
                template="plotly_white"
            )
            st.plotly_chart(fig_temp)

            # Nombre de mesures et température moyenne
            nb_mesures = len(df)
            temp_moyenne = df['Temperature (C)'].mean()
            st.info(f"Nombre total de mesures : {nb_mesures}  |  Température moyenne : {temp_moyenne:.2f} °C")

            # Histogramme Humidité
            st.subheader("💧 Répartition de l'humidité")
            fig_humidity = go.Figure(go.Histogram(
                x=df["Humidity"],
                nbinsx=20,
                marker_color='blue'
            ))
            fig_humidity.update_layout(
                xaxis_title="Humidité",
                yaxis_title="Fréquence",
                template="plotly_white"
            )
            st.plotly_chart(fig_humidity)
    else:
        st.warning("⚠ Les colonnes 'Date', 'Temperature (C)' ou 'Humidity' sont manquantes dans data.xlsx.")

# ------------------ Page Régression Linéaire ------------------
elif menu == "Régression Linéaire":
    if lr_model is None:
        st.error("❌ Modèle de régression linéaire non chargé.")
    else:
        st.title("Prédiction avec Régression Linéaire")
        st.subheader("Saisie des paramètres")

        humidity = st.number_input("Humidity", min_value=0.0, max_value=1.0, value=0.5)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0)
        pressure = st.number_input("Pressure (millibars)", min_value=900.0, max_value=1100.0, value=1013.0)

        input_data = pd.DataFrame({
            "Humidity": [humidity],
            "Wind Speed (km/h)": [wind_speed],
            "Pressure (millibars)": [pressure]
        })

        prediction = lr_model.predict(input_data)[0]
        st.success(f"Température prédite : {prediction:.2f} °C")

        # Visualisation
        categories = list(input_data.columns) + ["Température prédite"]
        values = list(input_data.iloc[0]) + [prediction]
        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            marker_color=['yellow', 'green', 'blue', 'red']
        ))
        fig.update_layout(yaxis_title="Valeur", xaxis_title="Paramètres", showlegend=False)
        st.plotly_chart(fig)

# ------------------ Page KNN ------------------
elif menu == "KNN":
    if knn_model is None or knn_scaler is None:
        st.error("❌ Modèle KNN ou scaler non chargé.")
    else:
        st.title("Prédiction avec K-Nearest Neighbors (KNN)")
        st.subheader("Saisie des paramètres")

        humidity = st.number_input("Humidity", min_value=0.0, max_value=1.0, value=0.5, key="humidity_knn")
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0, key="wind_knn")
        pressure = st.number_input("Pressure (millibars)", min_value=900.0, max_value=1100.0, value=1013.0, key="pressure_knn")

        input_data_knn = pd.DataFrame({
            "Humidity": [humidity],
            "Wind Speed (km/h)": [wind_speed],
            "Pressure (millibars)": [pressure]
        })

        input_scaled = knn_scaler.transform(input_data_knn)
        prediction_knn = knn_model.predict(input_scaled)[0]
        st.success(f"Température prédite (KNN) : {prediction_knn:.2f} °C")

        # Visualisation
        categories = list(input_data_knn.columns) + ["Température prédite"]
        values = list(input_data_knn.iloc[0]) + [prediction_knn]
        fig_knn = go.Figure(go.Bar(
            x=categories,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            marker_color=['yellow', 'green', 'blue', 'red']
        ))
        fig_knn.update_layout(yaxis_title="Valeur", xaxis_title="Paramètres", showlegend=False)
        st.plotly_chart(fig_knn)
