import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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
        # 🔧 Normalizar tipos en columnas categóricas
    for col in ["area", "grupo", "periodo", "apellidos", "nombres"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

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

    # Métricas principales (sobre el dataframe filtrado por sidebar)
    total_eval = len(df_f)                   # número de registros = evaluaciones
    aprob = int(df_f["y_aprueba"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Evaluaciones", total_eval)
    # 🔒 Siempre mostramos tasa (si total_eval > 0)
    c2.metric("Tasa de aprobación", "—" if total_eval == 0 else f"{aprob/total_eval:.1%}")

    # -------------------------
    # Comparativa por área
    # -------------------------
    st.markdown("### Tasa de aprobación por área (comparativa)")
    por_area_global = df.groupby("area")["y_aprueba"].mean().reset_index()
    por_area_global["tasa"] = por_area_global["y_aprueba"]
    por_area_global["seleccionada"] = por_area_global["area"].eq(area_sel) if area_sel != "(Todos)" else False

    fig_area = px.bar(
        por_area_global,
        x="area", y="tasa",
        color="seleccionada",
        color_discrete_map={True: "#636EFA", False: "#B0BEC5"},
        title="Tasa de aprobación por área",
        labels={"tasa": "Tasa de aprobación"}
    )
    fig_area.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_area, use_container_width=True)

    # -------------------------
    # Detalle por grupo si el usuario filtra un área
    # -------------------------
    if area_sel != "(Todos)":
        st.markdown(f"### Tasa de aprobación por grupo — Área: **{area_sel}**")
        if len(df_f) == 0:
            st.info("No hay datos para esta área con los filtros actuales.")
        else:
            por_grupo = df_f.groupby("grupo")["y_aprueba"].mean().reset_index()
            fig_grupo = px.bar(
                por_grupo.sort_values("y_aprueba", ascending=False),
                x="grupo", y="y_aprueba",
                title=f"Tasa de aprobación por grupo en {area_sel}",
                labels={"y_aprueba": "Tasa de aprobación"}
            )
            fig_grupo.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_grupo, use_container_width=True)




# -------------------------
# Vista modelos predictivos
# -------------------------
else:
    # ====== BEGIN: Vista modelos predictivos (sin matrices) ======
    st.subheader("Modelos predictivos")

    # Variables de entrada
    X = df[["area", "grupo", "periodo"]].astype(str)
    y = df["y_aprueba"]

    # Preprocesamiento
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["area", "grupo", "periodo"])],
        remainder="drop"
    )

    # Modelos (pipelines) con balanceo de clases
    lr = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    dt = Pipeline([("prep", pre), ("clf", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # =========================
    # PROBABILIDADES RL + UMBRAL ÓPTIMO (max F1)
    # =========================
    p_test = lr.predict_proba(X_test)[:, 1]
    ths = np.linspace(0, 1, 101)
    f1s = [f1_score(y_test, (p_test >= t).astype(int), zero_division=0) for t in ths]
    t_opt = float(ths[int(np.argmax(f1s))])
    f1_opt = max(f1s)

    usar_topt = st.checkbox(
        "Usar umbral óptimo (t*) para la Regresión Logística",
        value=True,
        help="Si está activado, usamos t* (max F1) en lugar del umbral fijo 0.5 para calcular las métricas de RL."
    )
    umbral_rl = t_opt if usar_topt else 0.5

    # Predicciones con el umbral elegido por el usuario (RL) y con 0.5 para Árbol
    y_pred_rl_ui = (p_test >= umbral_rl).astype(int)
    y_pred_dt = dt.predict(X_test)

    # =========================
    # MÉTRICAS (RL con umbral elegido) + (Árbol con 0.5)
    # =========================
    acc_rl = accuracy_score(y_test, y_pred_rl_ui)
    f1_rl  = f1_score(y_test, y_pred_rl_ui, pos_label=1, average="binary", zero_division=0)
    prec_rl = precision_score(y_test, y_pred_rl_ui, pos_label=1, zero_division=0)
    rec_rl  = recall_score(y_test,  y_pred_rl_ui, pos_label=1, zero_division=0)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt  = f1_score(y_test, y_pred_dt, pos_label=1, average="binary", zero_division=0)
    prec_dt = precision_score(y_test, y_pred_dt, pos_label=1, zero_division=0)
    rec_dt  = recall_score(y_test,  y_pred_dt, pos_label=1, zero_division=0)

    st.markdown(f"### Métricas de validación — RL (umbral = {umbral_rl:.2f}) vs Árbol (0.5)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (RL)", f"{acc_rl:.3f}")
    c2.metric("F1 (RL)",       f"{f1_rl:.3f}")
    c3.metric("Precision (RL)",f"{prec_rl:.3f}")
    c4.metric("Recall (RL)",   f"{rec_rl:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Accuracy (Árbol)", f"{acc_dt:.3f}")
    c6.metric("F1 (Árbol)",       f"{f1_dt:.3f}")
    c7.metric("Precision (Árbol)",f"{prec_dt:.3f}")
    c8.metric("Recall (Árbol)",   f"{rec_dt:.3f}")

    # Barra comparativa (sin use_container_width; ahora width="stretch")
    etiqueta_rl = f"Regresión logística (umbral {umbral_rl:.2f})"
    metrics_df = pd.DataFrame({
        "Modelo": [etiqueta_rl]*4 + ["Árbol de decisión"]*4,
        "Métrica": ["Accuracy", "F1", "Precision", "Recall"]*2,
        "Valor": [acc_rl, f1_rl, prec_rl, rec_rl, acc_dt, f1_dt, prec_dt, rec_dt]
    })
    fig_metrics = px.bar(
        metrics_df, x="Modelo", y="Valor", color="Métrica", barmode="group",
        text="Valor", range_y=[0, 1], title="Comparación de métricas por modelo"
    )
    fig_metrics.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig_metrics, width="stretch")
    st.caption(f"🧪 Umbral óptimo estimado para RL: t* = {t_opt:.2f} (F1 en test con t* = {f1_opt:.3f}).")

    # ================================
    # Explicación sencilla de métricas (para usuarios no técnicos)
    # ================================
    with st.expander("¿Qué significan Accuracy, F1, Precision y Recall?"):
        st.markdown(
            """
    ### Accuracy (Exactitud)
    De cada 100 predicciones, ¿cuántas acierta el modelo en promedio?  
    Ejemplo: si acierta 80 y falla 20 → **80% de Accuracy**.  
    ⚠️ Puede ser engañosa si una clase es mucho más grande que la otra.

    ---

    ### Precision (Precisión)
    De todos los que el modelo predijo como *Aprobados*, ¿qué porcentaje realmente aprobaron?  
    👉 Responde: *¿qué tan “finos” son mis positivos?*

    ---

    ### Recall (Cobertura)
    De todos los que realmente aprobaron, ¿qué porcentaje detectó el modelo?  
    👉 Responde: *¿cuántos casos positivos reales encontré?*

    ---

    ### F1-score
    Es una combinación equilibrada entre *Precision* y *Recall*.  
    Sirve mucho cuando los datos están desequilibrados (más aprobados que no aprobados).  
    👉 Cuanto más cerca de **1.0**, mejor el balance entre ambos.

    ---

    💡 **Resumen práctico:**
    - *Accuracy* = aciertos globales.  
    - *Precision* = de los que marqué como aprobados, cuántos lo eran.  
    - *Recall* = de los aprobados reales, cuántos logré encontrar.  
    - *F1* = equilibrio entre precisión y cobertura.
                """
        )

    # ================================
    # PREDICCIONES INDIVIDUALES (RL)
    # ================================
    st.markdown("### Predicciones individuales (probabilidad de aprobación, RL)")
    probs_all = lr.predict_proba(X)[:, 1]
    df_pred = df.copy()
    df_pred["prob_aprobacion"] = probs_all

    # Evitar problemas de tipado al mostrar/descargar
    for col in ["estudiante_id", "area", "grupo", "periodo"]:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].astype(str)

    st.dataframe(
        df_pred[["estudiante_id", "area", "grupo", "prob_aprobacion"]]
        .sort_values("prob_aprobacion", ascending=False)
        .head(20),
        width="stretch"
    )

    st.download_button(
        label="Descargar todas las predicciones",
        data=df_pred.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_completas.csv",
        mime="text/csv"
    )

    # Histograma (sin use_container_width)
    st.markdown("### Distribución de probabilidades de aprobación (RL)")
    fig_hist = px.histogram(
        df_pred, x="prob_aprobacion", nbins=20,
        title="Distribución de probabilidades estimadas"
    )
    st.plotly_chart(fig_hist, width="stretch")

    # Importancia de variables (Árbol)
    st.markdown("### Importancia de variables (Árbol de decisión)")
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_ohe = ohe.fit_transform(X)
    dt_raw = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42).fit(X_ohe, y)

    importancias = pd.DataFrame({
        "feature": ohe.get_feature_names_out(["area", "grupo", "periodo"]),
        "importance": dt_raw.feature_importances_
    }).sort_values("importance", ascending=False)

    fig_imp = px.bar(importancias.head(10), x="feature", y="importance",
                     title="Top 10 características por importancia")
    st.plotly_chart(fig_imp, width="stretch")
    # ====== END: Vista modelos predictivos (sin matrices) ======

