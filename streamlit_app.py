import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

st.set_page_config(page_title="Dashboard - Modeles cardiaques", layout="wide")

st.title("Dashboard de comparaison des modeles")
st.caption("Prediction du risque de maladie cardiaque - dataset combine (1190 patients)")

# ---------------------------------------------------------
# Chargement des donnees
# ---------------------------------------------------------
DATA_PATH = "heart_statlog_cleveland_hungary_final.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

# ---------------------------------------------------------
# Description des colonnes
# ---------------------------------------------------------
COLUMN_DESCRIPTIONS = {
    "age": "Age du patient",
    "sex": "Sexe du patient (1 = homme, 0 = femme)",
    "chest pain type": "Type de douleur thoracique",
    "resting bp s": "Pression arterielle au repos",
    "cholesterol": "Taux de cholesterol",
    "fasting blood sugar": "Glycemie a jeun",
    "resting ecg": "Resultat ECG au repos",
    "max heart rate": "Frequence cardiaque maximale",
    "exercise angina": "Angine provoquee par l'effort",
    "oldpeak": "Depression du segment ST",
    "ST slope": "Pente du segment ST",
    "target": "Maladie cardiovasculaire (1 = malade, 0 = sain)",
}

# ---------------------------------------------------------
# Section 1 : Donnees - lecture et traitement (vue resumee)
# ---------------------------------------------------------
st.header("1. Donnees : lecture et traitement")

nb_zero_chol = (data["cholesterol"] == 0).sum()

# --- Indicateurs cles ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Patients", len(data))
kpi2.metric("Variables", data.shape[1] - 1)
kpi3.metric("Malades", f"{(data['target'].mean() * 100):.0f} %")
kpi4.metric("Cholesterol a 0 corrige", int(nb_zero_chol))

# --- Visuels : repartition de la cible + effet du nettoyage du cholesterol ---
viz1, viz2 = st.columns(2)

with viz1:
    target_counts = data["target"].map({0: "Sain", 1: "Malade"}).value_counts().reset_index()
    target_counts.columns = ["Statut", "Nombre"]
    fig_target = px.pie(
        target_counts, names="Statut", values="Nombre", hole=0.5,
        title="Repartition des patients", color="Statut",
        color_discrete_map={"Sain": "#4C78A8", "Malade": "#E45756"},
    )
    fig_target.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_target, use_container_width=True)

data_clean = data.copy()
mediane_chol = data_clean.loc[data_clean["cholesterol"] > 0, "cholesterol"].median()
if nb_zero_chol > 0:
    data_clean["cholesterol"] = data_clean["cholesterol"].replace(0, mediane_chol)

with viz2:
    fig_chol = px.histogram(
        data_clean, x="cholesterol", nbins=30,
        title="Distribution du cholesterol (apres nettoyage)",
        color_discrete_sequence=["#4C78A8"],
    )
    fig_chol.add_vline(
        x=mediane_chol, line_dash="dash", line_color="#E45756", line_width=2,
        annotation_text=f"Mediane = {mediane_chol:.0f}", annotation_position="top right",
    )
    fig_chol.update_layout(
        height=320, margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Cholesterol", yaxis_title="Nombre de patients",
        bargap=0.05,
    )
    st.plotly_chart(fig_chol, use_container_width=True)
    if nb_zero_chol > 0:
        st.caption(f"{nb_zero_chol} valeurs a 0 remplacees par la mediane avant tracage.")

with st.expander("Description des colonnes du dataset"):
    desc_df = pd.DataFrame([
        {"Colonne": col, "Description": COLUMN_DESCRIPTIONS.get(col, "Description non disponible")}
        for col in data.columns
    ])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

st.caption(
    "Encodage : les colonnes categorielles (chest pain type, resting ecg, ST slope) sont "
    "transformees en variables 0/1 (one-hot encoding)."
)

categorical_cols = ["chest pain type", "resting ecg", "ST slope"]
data_clean = pd.get_dummies(data_clean, columns=categorical_cols, drop_first=True)

X = data_clean.drop(columns=["target"])
y = data_clean["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Selection des variables (SelectKBest) ---
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)
scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
top_features = scores.head(10).index.tolist()

fig_scores = px.bar(
    x=scores.values, y=scores.index, orientation="h",
    color=[c in top_features for c in scores.index],
    color_discrete_map={True: "#4C78A8", False: "#D3D3D3"},
    title="Pertinence des variables (SelectKBest - score ANOVA F) - les 10 retenues en bleu",
    labels={"x": "Score F (ANOVA)", "y": "Variable"},
)
fig_scores.update_layout(yaxis={"categoryorder": "total ascending"}, height=400, showlegend=False)
st.plotly_chart(fig_scores, use_container_width=True)

st.caption(
    f"Split train/test : {X_train.shape[0]} patients entrainement / {X_test.shape[0]} test. "
    f"Variables retenues : {', '.join(top_features)}"
)

X_train = X_train[top_features]
X_test = X_test[top_features]

st.divider()

# ---------------------------------------------------------
# Section 2 : Entrainement et evaluation des modeles
# ---------------------------------------------------------
st.header("2. Entrainement et evaluation des modeles")

MODEL_NAMES = [
    "Regression Logistique",
    "SVM",
    "Random Forest",
    "XGBoost",
    "Random Forest Optimise",
    "Voting Classifier",
]
st.markdown(" | ".join(f"`{name}`" for name in MODEL_NAMES))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_all_models(X_train, X_test, y_train, X_train_scaled, X_test_scaled):
    results = {}

    # 5.1 Regression Logistique
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    results["Regression Logistique"] = {"model": model_lr, "y_pred": model_lr.predict(X_test)}

    # 5.2 Support Vector Classifier (donnees standardisees)
    model_svc = SVC(kernel="rbf", probability=True, random_state=42)
    model_svc.fit(X_train_scaled, y_train)
    results["SVM"] = {"model": model_svc, "y_pred": model_svc.predict(X_test_scaled)}

    # 5.3 Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    results["Random Forest"] = {"model": model_rf, "y_pred": model_rf.predict(X_test)}

    # 5.4 XGBoost
    model_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    model_xgb.fit(X_train, y_train)
    results["XGBoost"] = {"model": model_xgb, "y_pred": model_xgb.predict(X_test)}

    # 5.5 Random Forest optimise (GridSearchCV)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    model_rf_optimized = grid_search.best_estimator_
    results["Random Forest Optimise"] = {
        "model": model_rf_optimized,
        "y_pred": model_rf_optimized.predict(X_test),
        "best_params": grid_search.best_params_,
    }

    # 5.6 Voting Classifier (ensemble)
    voting_model = VotingClassifier(
        estimators=[
            ("lr", model_lr),
            ("rf", model_rf_optimized),
            ("xgb", model_xgb),
        ],
        voting="soft",
    )
    voting_model.fit(X_train, y_train)
    results["Voting Classifier"] = {"model": voting_model, "y_pred": voting_model.predict(X_test)}

    return results

with st.spinner("Entrainement des modeles en cours (incluant GridSearchCV)..."):
    model_results = train_all_models(X_train, X_test, y_train, X_train_scaled, X_test_scaled)

with st.expander("Voir les meilleurs parametres du Random Forest optimise (GridSearchCV)"):
    st.write(model_results["Random Forest Optimise"]["best_params"])

st.divider()

# ---------------------------------------------------------
# Section 6 : Metriques de performance
# ---------------------------------------------------------
st.header("3. Metriques de performance")

metrics_rows = []
for name, res in model_results.items():
    y_pred = res["y_pred"]
    metrics_rows.append({
        "Modele": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Rappel (recall)": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    })

metrics_df = pd.DataFrame(metrics_rows).sort_values(by="Accuracy", ascending=False)
st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Rappel (recall)": "{:.2%}",
        "F1-score": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

fig_bar = px.bar(
    metrics_df.melt(id_vars="Modele", var_name="Metrique", value_name="Score"),
    x="Modele",
    y="Score",
    color="Metrique",
    barmode="group",
    title="Comparaison des metriques par modele",
)
st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("Voir le classification_report detaille de chaque modele"):
    for name, res in model_results.items():
        st.text(f"--- {name} ---")
        st.text(classification_report(y_test, res["y_pred"]))

st.divider()

# ---------------------------------------------------------
# Section 7 : Matrices de confusion de tous les modeles
# ---------------------------------------------------------
st.header("4. Matrices de confusion")

model_names = list(model_results.keys())
n_cols = 2
rows_needed = (len(model_names) + n_cols - 1) // n_cols

for row_idx in range(rows_needed):
    cols = st.columns(n_cols)
    for col_idx in range(n_cols):
        idx = row_idx * n_cols + col_idx
        if idx >= len(model_names):
            continue
        name = model_names[idx]
        y_pred = model_results[name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pas de maladie", "Maladie"],
                y=["Pas de maladie", "Maladie"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 18},
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=f"Matrice de confusion - {name}",
            xaxis_title="Prediction du modele",
            yaxis_title="Vraie valeur",
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        with cols[col_idx]:
            st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Section 8 : Comparaison des modeles
# ---------------------------------------------------------
st.header("5. Comparaison des modeles")

comparison_display = metrics_df[["Modele", "Accuracy", "Rappel (recall)"]].copy()
comparison_display["Accuracy"] = (comparison_display["Accuracy"] * 100).round(0).astype(int).astype(str) + "%"
comparison_display["Rappel (recall)"] = (comparison_display["Rappel (recall)"] * 100).round(0).astype(int).astype(str) + "%"
st.table(comparison_display.set_index("Modele"))

st.markdown("### Pourquoi on choisit le Voting Classifier")
st.markdown(
    "- Il obtient la meilleure accuracy globale parmi les modeles testes.\n"
    "- Il obtient egalement un excellent rappel sur la classe Maladie, ce qui mesure la "
    "capacite du modele a detecter les vrais malades, minimisant les faux negatifs.\n"
    "- En combinant les predictions de la Regression Logistique, du Random Forest optimise "
    "et de XGBoost, il beneficie des forces de chaque approche tout en compensant leurs "
    "faiblesses individuelles."
)

st.divider()

# ---------------------------------------------------------
# Section 9 : Interpretation - importance des variables
# ---------------------------------------------------------
st.header("6. Interpretation : quelles variables comptent le plus ?")

model_rf_optimized = model_results["Random Forest Optimise"]["model"]
model_xgb = model_results["XGBoost"]["model"]

importance_rf = pd.Series(model_rf_optimized.feature_importances_, index=X_train.columns)
importance_xgb = pd.Series(model_xgb.feature_importances_, index=X_train.columns)
importances = ((importance_rf + importance_xgb) / 2).sort_values(ascending=False)

fig_imp = px.bar(
    x=importances.values,
    y=importances.index,
    orientation="h",
    title="Importance moyenne des variables (Random Forest optimise + XGBoost)",
    labels={"x": "Importance", "y": "Variable"},
)
fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_imp, use_container_width=True)