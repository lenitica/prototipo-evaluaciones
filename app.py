import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# Configuración general
# -------------------------
st.set_page_config(page_title="Prototipo Evaluaciones", layout="wide")

st.title("Prototipo de análisis y predicción de evaluaciones")

# -------------------------
# Cargar datos
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("CONSOLIDADO_PER1_2025_.xlsx", sheet_name="Datos_Consolidados")
    df.columns = [c.strip().lower() for c in df.columns]
    df["observacion"] = df["observacion"].str.strip().str.title()
    df["y_aprueba"] = df["observacion"].map({"Aprobado": 1, "No Aprobado": 0})
    df["estudiante_id"] = (df["apellidos"] + " " + df["nombres"]).str.upper()
    return df

df = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Filtros")
areas = ["(Todos)"] + sorted(df["area"].unique())
area_sel = st.sidebar.selectbox("Área", areas)
modo = st.sidebar.radio("Módulo", ["Análisis psicométrico", "Modelos predictivos"])

# -------------------------
# Aplicar filtro
# -------------------------
df_f = df.copy()
if area_sel != "(Todos)":
    df_f = df_f[df_f["area"] == area_sel]

# -------------------------
# Vista psicométrica
# -------------------------
if modo == "Análisis psicométrico":
    st.subheader("Análisis psicométrico")
    total = len(df_f)
    aprob = df_f["y_aprueba"].sum()
    st.metric("Estudiantes", total)
    st.metric("Aprobados", int(aprob))
    if total > 0:
        st.metric("Tasa de aprobación", f"{aprob/total:.1%}")

    por_area = df.groupby("area")["y_aprueba"].mean().reset_index()
    fig = px.bar(por_area, x="area", y="y_aprueba", title="Tasa de aprobación por área")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Vista modelos predictivos
# -------------------------
else:
    st.subheader("Modelos predictivos")

    # Variables de entrada
    X = df[["area", "grupo", "periodo"]].astype(str)
    y = df["y_aprueba"]

    # Preprocesamiento
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["area", "grupo", "periodo"])],
        remainder="drop"
    )

    # Modelos
    lr = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    dt = Pipeline([("prep", pre), ("clf", DecisionTreeClassifier(max_depth=5))])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Función de métricas
    def metricas(model, X_t, y_t):
        y_pred = model.predict(X_t)
        acc = accuracy_score(y_t, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_pred, average="binary", zero_division=0)
        return acc, f1

    # Evaluación
    acc_lr, f1_lr = metricas(lr, X_test, y_test)
    acc_dt, f1_dt = metricas(dt, X_test, y_test)

    st.write("**Regresión logística** → Accuracy:", round(acc_lr, 3), "F1:", round(f1_lr, 3))
    st.write("**Árbol de decisión** → Accuracy:", round(acc_dt, 3), "F1:", round(f1_dt, 3))

    # Probabilidades en todo el dataset
    probs = lr.predict_proba(X)[:, 1]
    df_pred = df.copy()
    df_pred["prob_aprobacion"] = probs

    # Mostrar primeras 20 filas
    st.dataframe(df_pred[["estudiante_id", "area", "grupo", "prob_aprobacion"]].head(20))

    # Botón de descarga
    st.download_button(
        label="⬇️ Descargar todas las predicciones",
        data=df_pred.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_completas.csv",
        mime="text/csv"
    )

    # -------------------------
    # Gráfico: Histograma de probabilidades
    # -------------------------
    st.markdown("### Distribución de probabilidades de aprobación")
    fig_hist = px.histogram(
        df_pred,
        x="prob_aprobacion",
        nbins=20,
        title="Distribución de probabilidades (Regresión Logística)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------
    # Gráfico: Importancia de variables del Árbol de Decisión
    # -------------------------
    st.markdown("### Importancia de variables (Árbol de decisión)")
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_ohe = ohe.fit_transform(X)
    dt_raw = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_ohe, y)

    importancias = pd.DataFrame({
        "feature": ohe.get_feature_names_out(["area", "grupo", "periodo"]),
        "importance": dt_raw.feature_importances_
    }).sort_values("importance", ascending=False)

    fig_imp = px.bar(importancias.head(10), x="feature", y="importance", title="Top 10 características")
    st.plotly_chart(fig_imp, use_container_width=True)

