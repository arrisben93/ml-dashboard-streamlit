# ml-dashboard-streamlit

Ce projet est une application web développée avec **Streamlit**, visant à permettre l'exploration de données, la création de modèles de machine learning (classification et régression), et la prédiction à partir de données utilisateurs.

## 📌 Fonctionnalités

- 📁 Chargement de fichiers CSV
- 📊 Statistiques descriptives + heatmap de corrélation
- 🧠 Apprentissage automatique avec :
  - Classification : Logistic Regression, Random Forest, KNN, SVM
  - Régression : Linear Regression, Random Forest Regressor, KNN Regressor, SVR
- ⚙️ Prétraitement avec encodage automatique + normalisation (`StandardScaler`, `MinMaxScaler`, `Normalizer`)
- 🧮 Évaluation des modèles avec métriques :
  - Classification : Accuracy, Precision, Recall, F1-score
  - Régression : MAE, MSE, R²
- 🔮 Interface de prédiction en direct avec les modèles entraînés

## 🛠️ Technologies utilisées

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)

## 📂 Structure du projet

```bash
.
├── app.py              # Application principale Streamlit
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
├── data/               # Dossier pour stocker vos CSV (à ajouter localement)
````

## ▶️ Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📚 Cours liés

Ce projet réutilise les compétences vues dans les cours :

* Python de base (variables, boucles, conditions, fonctions)
* Pandas, NumPy
* Statistiques descriptives et corrélations
* Visualisation (Matplotlib, Seaborn)
* Machine Learning (modèles, normalisation, métriques)
* Streamlit pour l’interface utilisateur

## 🏁 Auteur

* **Nom :** *\ARISS*
* **Formation :** IA1 – 420-IAA-TT
* **Encadrant :** *\BENFRIHA*

```
