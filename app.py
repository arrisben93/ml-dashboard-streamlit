import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, mean_absolute_error, 
    mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

# Titre principal
st.markdown("""
    <h1 style='text-align: center; color: orange;'>Projet Final de IA</h1>
""", unsafe_allow_html=True)

# Barre latérale pour la navigation
st.sidebar.title("MENU")
menu = st.sidebar.radio("Aller vers", [
    "Chargement CSV", 
    "Statistiques", 
    "Apprentissage automatique", 
    "Prédiction"
])

models_trained = {}
df = None

# 1. Chargement du fichier CSV
if menu == "Chargement CSV":
    file = st.file_uploader("Téléversez un fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state['data'] = df
        st.success("Fichier chargé avec succès !")
        st.dataframe(df.head())

# 2. Statistiques descriptives
elif menu == "Statistiques":
    if 'data' not in st.session_state:
        st.warning("Veuillez d'abord charger un fichier CSV")
    else:
        df = st.session_state['data']

        st.subheader("Statistiques Descriptives")
        st.write(df.describe())

        st.subheader("Matrice de Corrélation")
        numeric_data = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# 3. Module d'apprentissage automatique
elif menu == "Apprentissage automatique":
    file = st.file_uploader("Retéléversez un fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state['data'] = df
        st.success("Fichier chargé !")

    if 'data' in st.session_state:
        df = st.session_state['data']
        task_type = st.radio("Type de tâche ML :", ["Classification", "Régression"])
        target_col = st.selectbox("Colonne cible :", df.columns)

        X = df.drop(columns=target_col)
        y = df[target_col]

        # Auto correction du type de tâche
        if task_type == "Classification" and y.dtype in ['float64', 'float32']:
            st.warning("La cible semble continue, passage en tâche de Régression.")
            task_type = "Régression"

        # Encodage des variables catégorielles
        X = pd.get_dummies(X)

        # Normalisation
        normalizer = st.selectbox("Normalisation :", ["Aucune", "StandardScaler", "MinMaxScaler", "Normalizer"])
        if normalizer == "StandardScaler":
            X = StandardScaler().fit_transform(X)
        elif normalizer == "MinMaxScaler":
            X = MinMaxScaler().fit_transform(X)
        elif normalizer == "Normalizer":
            X = Normalizer().fit_transform(X)
        else:
            X = X.values

        # Division du dataset
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.info("Entraînement des modèles...")

        # Choix des modèles
        if task_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "SVR": SVR()
            }

        # Entraînement et évaluation
        results = []
        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            models_trained[name] = model

            if task_type == "Classification":
                results.append([
                    name,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ])
            else:
                results.append([
                    name,
                    mean_absolute_error(y_test, y_pred),
                    mean_squared_error(y_test, y_pred),
                    r2_score(y_test, y_pred)
                ])

        # Sauvegarde des modèles dans la session
        st.session_state['models_trained'] = models_trained

        # Affichage des résultats
        st.subheader("Résultats des modèles")
        if task_type == "Classification":
            st.dataframe(pd.DataFrame(results, columns=["Modèle", "Accuracy", "Precision", "Recall", "F1-score"]))
        else:
            st.dataframe(pd.DataFrame(results, columns=["Modèle", "MAE", "MSE", "R2"]))

# 4. Interface de prédiction
elif menu == "Prédiction":
    if 'data' not in st.session_state or 'models_trained' not in st.session_state:
        st.warning("Veuillez entraîner un modèle dans l'onglet Apprentissage automatique")
    else:
        df = st.session_state['data']
        models_trained = st.session_state['models_trained']
        model_name = st.selectbox("Sélectionner un modèle :", list(models_trained.keys()))
        model = models_trained[model_name]

        target_col = st.selectbox("Redonner la colonne cible :", df.columns)
        X = df.drop(columns=target_col)
        X = pd.get_dummies(X)

        inputs = []
        st.subheader("Entrer les nouvelles valeurs à prédire :")
        for col in X.columns:
            val = st.number_input(f"{col}", step=1.0)
            inputs.append(val)

        if st.button("Lancer la prédiction"):
            prediction = model.predict([inputs])
            st.success(f"Résultat prédit : {prediction[0]}")
