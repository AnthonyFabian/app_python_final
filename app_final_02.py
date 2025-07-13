import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Dashboard Farmacéutico")

# ==============================
# Paletas de colores
# ==============================
PALETTE_TASKS = sns.color_palette("Blues_d", 8)
PALETTE_OPS = sns.color_palette("crest", 8)

# ==============================
# Diccionario de subprocesos
# ==============================
sub_procesos_fp = {
    'REAB_BAJO_FLPN_ESTAN': 'Reabastecimiento Caja',
    'REAB_ALTO_FLPN_ESTAN': 'Reabastecimiento Caja',
    'REAB_BAJO_CASES': 'Reabastecimiento Caja',
    'REAB_ALTO_FLPN': 'Reabastecimiento Caja',
    'REAB_BAJO_FLPN': 'Reabastecimiento Caja',
    'REAB_ALTO_CASES': 'Reabastecimiento Caja',
    'REAB_BAJO_PALLET': 'Reabastecimiento Pallet',
    'REAB_ALTO_PALLET': 'Reabastecimiento Pallet',
    'PANAL_REAB_BAJO_PALLET': 'Reabastecimiento Pallet',
    'PANAL_REAB_ALTO_PALLET': 'Reabastecimiento Pallet',
    'REAB_BAJO_FLPN_MM': 'Reabastecimiento Caja',
    'Cases Replenishement': 'Reabastecimiento Caja',
    'REAB_BAJO_FLPN_ESTAN_P2': 'Reabastecimiento Caja',
    'REAB_ALTO_FLPN_ESTAN_P2': 'Reabastecimiento Caja',
    'Picking Cubicado CD11': 'Picking Unidades',
    'PICK_CUB_ACTIV_GENERAL': 'Picking Unidades',
    'PICK_CUB_RESERVA_GENERAL': 'Picking Unidades',
    'PICK_MZ_CASE_ACTIVO': 'Picking Cajas',
    'PICK_CUB_ACTIV_GENERAL_X': 'Picking Unidades',
    'PICK_CUB_ACTIV_GENERAL_MFA': 'Picking Unidades',
    'PICK_ACTIVE_UN_REFRIG': 'Picking Refrigerado',
    'CC Location LPN Scan': 'Otros',
    'FECHA_CORTA_ESTAN_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_ESTAN_SP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_ALTO_SP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_BAJO_SP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_ALTO_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_BAJO_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_ALTO_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_BAJO_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_KMNL_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_MZ_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_ESTAN_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_SDA_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_KMNL_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_MZ_SP': 'Gestion Vencimiento Corto',
    'Pick Cart': 'Picking Unidades',
    'FECHA_CORTA_KMNL_SP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_SDA_CPE': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_MZ_CP': 'Gestion Vencimiento Corto',
    'FECHA_CORTA_SDA_SP': 'Gestion Vencimiento Corto',
    'PLT_NA_TRASLADO_CD12': 'Traslados Internos',
    'NA_TRASLADO_CD12': 'Traslados Internos',
    'PLT_NB_TRASLADO_CD12': 'Traslados Internos',
    'NB_TRASLADO_CD12': 'Traslados Internos',
    'PLT_NB_TRASLADO_CD06': 'Traslados Internos',
    'PLT_NA_TRASLADO_CD06': 'Traslados Internos',
    'NA_TRASLADO_CD06': 'Traslados Internos',
    'NB_TRASLADO_CD06': 'Traslados Internos',
    'LIMPIEZA VIDRIO': 'Otros',
    'EXTR_ALTO_PALL': 'Otros'
}

procesos_fp = {
    'Reabastecimiento Caja': 'Reabastecimiento',
    'Reabastecimiento Pallet': 'Reabastecimiento',
    'Picking Unidades': 'Picking',
    "Gestion Vencimiento Corto": "Otros Procesos",
    'Picking Cajas': 'Picking',
    'Picking Refrigerado': 'Picking',
    'Traslados Internos': 'Traslados',
    'Otros': 'Otros Procesos',
}

# ==============================
# Carga de datos
# ==============================
@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "D:/python_02/data_pyhton_pf.csv",  # Cambia aquí tu ruta
        encoding='latin1', sep=';'
    )
    # Fechas
    df["Fe y Hr Crea"] = pd.to_datetime(df["Fe y Hr Crea"], dayfirst=True)
    df["Fe y Hr Modif"] = pd.to_datetime(df["Fe y Hr Modif"], dayfirst=True)
    # Duración
    df["Duracion_min"] = (df["Fe y Hr Modif"] - df["Fe y Hr Crea"]).dt.total_seconds() / 60
    df["Anio_Mes"] = df["Fe y Hr Modif"].dt.to_period("M").astype(str)
    # Mapear subprocesos y procesos
    df["Subproceso"] = df["Tipo de tarea"].map(sub_procesos_fp).fillna("Otros")
    df["Proceso"] = df["Subproceso"].map(procesos_fp).fillna("Otros Procesos")
    return df

df = cargar_datos()

# ==============================
# Sidebar y Filtros
# ==============================
st.sidebar.title("Filtros generales")

# Periodo
min_date = df["Fe y Hr Crea"].min().date()
max_date = df["Fe y Hr Modif"].max().date()
periodo = st.sidebar.date_input("Periodo", [min_date, max_date])
if len(periodo) != 2 or periodo[0] is None or periodo[1] is None:
    st.warning("Seleccione el periodo inicial y final para filtrar los datos.")
    st.stop()
start_date, end_date = periodo
if start_date > end_date:
    st.warning("La fecha inicial no puede ser mayor que la final.")
    st.stop()

# Filtro de periodo aplicado a los datos
df = df[(df["Fe y Hr Crea"].dt.date >= start_date) & (df["Fe y Hr Modif"].dt.date <= end_date)]

# Sección: Área de análisis
area = st.sidebar.radio("Área de análisis", ["Tareas", "Operarios"])

# ==============================
# DASHBOARD - TAREAS
# ==============================
if area == "Tareas":
    st.markdown("## 🗃️ Análisis de Tareas Operativas")
    subtareas = [
        "Evolución mensual",
        "Distribución por tipo",
        "Duración promedio por tipo",
        "Backlog (tareas no finalizadas)",
        "Tareas por estado",
        "Tareas por ubicación actual"
    ]
    sub_tarea = st.sidebar.radio("Análisis de Tareas", subtareas)

    # Datos filtrados
    df_finalizadas = df[df["Estado"] == "Finalizada"]

    # KPIs
    total_tareas = df.shape[0]
    tareas_finalizadas = df_finalizadas.shape[0]
    duracion_promedio = df_finalizadas["Duracion_min"].mean()
    porc_finalizadas = 100 * tareas_finalizadas / total_tareas if total_tareas else 0
    st.write(f"Total Tareas (todas): **{total_tareas:,}** &nbsp;&nbsp; | &nbsp;&nbsp; Tareas Finalizadas: **{tareas_finalizadas:,}** &nbsp;&nbsp; | &nbsp;&nbsp; Duración Prom. (min): **{duracion_promedio:.1f}** &nbsp;&nbsp; | &nbsp;&nbsp; % Finalizadas: **{porc_finalizadas:.1f}%**")

    if sub_tarea == "Evolución mensual":
        conteo = (
            df_finalizadas.groupby("Anio_Mes")["Nro Tarea"].count().reset_index()
        )
        plt.figure(figsize=(6, 3))
        bars = plt.bar(conteo["Anio_Mes"], conteo["Nro Tarea"], color=PALETTE_TASKS)
        plt.title("Cantidad de tareas finalizadas por mes")
        plt.ylabel("Cantidad")
        plt.xlabel("Mes")
        plt.xticks(rotation=45)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height()):,}", ha='center', va='bottom', fontsize=6)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_tarea == "Distribución por tipo":
        conteo = df_finalizadas["Subproceso"].value_counts()
        top_n = 6
        nombres_top = conteo.index[:top_n].tolist()
        valores_top = conteo.values[:top_n]
        otros = conteo[top_n:].sum()

        nombres_final = nombres_top.copy()
        valores_final = list(valores_top)
        if "Otros" not in nombres_final and otros > 0:
            nombres_final.append("Otros")
            valores_final.append(otros)
        elif otros > 0:
            idx = nombres_final.index("Otros")
            valores_final[idx] += otros

        if len(nombres_final) <= 1:
            st.warning("No hay suficientes tipos distintos de tarea para mostrar el gráfico.")
        else:
            plt.figure(figsize=(5, 4))
            wedges, texts, autotexts = plt.pie(
                valores_final,
                autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
                startangle=90,
                wedgeprops=dict(width=0.4),
                labels=None
            )
            plt.legend(nombres_final, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
            plt.title("Distribución de Tipos de Tarea Finalizadas")
            plt.tight_layout()
            st.pyplot(plt.gcf())

    elif sub_tarea == "Duración promedio por tipo":
        prom_dur = (
            df_finalizadas.groupby("Subproceso")["Duracion_min"].mean()
            .sort_values(ascending=False).reset_index()
        )
        prom_dur_sin_otros = prom_dur[prom_dur["Subproceso"] != "Otros"]

        if len(prom_dur_sin_otros) == 0:
            st.warning("No hay suficientes subprocesos distintos para mostrar el ranking. Todos los registros actuales corresponden a 'Otros'.")
        else:
            top_n = min(20, len(prom_dur_sin_otros))
            top_prom = prom_dur_sin_otros.head(top_n)
            plt.figure(figsize=(7, 4))
            ax = sns.barplot(x="Duracion_min", y="Subproceso", data=top_prom, palette=PALETTE_TASKS)
            plt.xlabel("Duración promedio (min)")
            plt.ylabel("Tipo de tarea")
            plt.title(f"Duración Promedio por Tipo de Tarea (Top {top_n})")
            for i, v in enumerate(top_prom["Duracion_min"]):
                ax.text(v + max(top_prom["Duracion_min"])*0.01, i, f"{v:.1f}", va="center", fontsize=6)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            if len(prom_dur_sin_otros) > top_n:
                st.info(f"Se muestran los {top_n} subprocesos con mayor duración promedio.")

    elif sub_tarea == "Backlog (tareas no finalizadas)":
        backlog = df[df["Estado"] != "Finalizada"]
        st.write(f"Total de tareas en backlog: **{backlog.shape[0]:,}**")
        conteo = backlog["Subproceso"].value_counts()
        plt.figure(figsize=(5, 3))
        bars = plt.barh(conteo.index, conteo.values, color=PALETTE_TASKS)
        plt.xlabel("Cantidad")
        plt.title("Backlog por subproceso")
        for i, v in enumerate(conteo.values):
            plt.text(v, i, f"{v:,}", va="center", fontsize=6)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_tarea == "Tareas por estado":
        conteo = df["Estado"].value_counts()
        plt.figure(figsize=(5, 4))
        wedges, texts, autotexts = plt.pie(
            conteo.values,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
            startangle=90,
            wedgeprops=dict(width=0.4),
            labels=None
        )
        plt.legend(conteo.index, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
        plt.title("Distribución por Estado de Tareas")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_tarea == "Tareas por ubicación actual":
        conteo = df_finalizadas["Ubicación actual"].value_counts().head(10)
        plt.figure(figsize=(7, 3))
        bars = plt.bar(conteo.index, conteo.values, color=PALETTE_TASKS)
        plt.title("Top 10 Ubicaciones actuales con más tareas finalizadas")
        plt.ylabel("Cantidad")
        plt.xlabel("Ubicación")
        plt.xticks(rotation=45)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height()):,}", ha='center', va='bottom', fontsize=6)
        plt.tight_layout()
        st.pyplot(plt.gcf())

# ==============================
# DASHBOARD - OPERARIOS
# ==============================
else:
    st.markdown("## 👷‍♂️ Análisis de Operarios")
    subops = [
        "Productividad (tareas realizadas)",
        "Cumplimiento por operario",
        "Distribución de tareas entre operarios",
        "Ranking gerencial de operarios"
    ]
    sub_ops = st.sidebar.radio("Análisis de Operarios", subops)
    slider_default = 10

    # KPIs Operarios
    df_ops = df[df["Estado"] == "Finalizada"]
    n_ops = df_ops["Usuario Modificac"].nunique()
    prom_tareas = df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().mean()
    max_tareas = df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().max()
    min_tareas = df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().min()
    nombre_max = df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().idxmax()
    nombre_min = df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().idxmin()
    st.write(f"Operarios activos: **{n_ops:,}** &nbsp;&nbsp; | &nbsp;&nbsp; Prom. tareas por operario: **{prom_tareas:.1f}** &nbsp;&nbsp; | &nbsp;&nbsp; Máx tareas un operario: **{max_tareas:,} ({nombre_max})** &nbsp;&nbsp; | &nbsp;&nbsp; Min tareas un operario: **{min_tareas:,} ({nombre_min})**")

    if sub_ops == "Productividad (tareas realizadas)":
        top_n = st.sidebar.slider("¿Cuántos operarios mostrar?", min_value=5, max_value=50, value=slider_default, key="prod")
        tareas_ops = (
            df_ops.groupby("Usuario Modificac")["Nro Tarea"].count().sort_values(ascending=False)
        ).head(top_n)
        plt.figure(figsize=(7, 4))
        bars = plt.barh(tareas_ops.index, tareas_ops.values, color=PALETTE_OPS)
        plt.xlabel("Tareas realizadas")
        plt.ylabel("Operario")
        plt.title(f"Productividad: Tareas realizadas por operario (Top {top_n})")
        for i, v in enumerate(tareas_ops.values):
            plt.text(v, i, f"{v:,}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_ops == "Cumplimiento por operario":
        prom_tipo = (
            df_ops.groupby("Subproceso")["Duracion_min"].mean().reset_index()
        )
        df_merged = df_ops.merge(
            prom_tipo, on="Subproceso", suffixes=("", "_prom")
        )
        df_merged["Cumple"] = df_merged["Duracion_min"] <= df_merged["Duracion_min_prom"]
        cumplimiento = (
            df_merged.groupby(["Usuario Modificac", "Cumple"]).size().unstack().fillna(0)
        )
        top_n = st.sidebar.slider("¿Cuántos operarios mostrar?", min_value=5, max_value=50, value=slider_default, key="cumple")
        top_users = cumplimiento.sum(axis=1).sort_values(ascending=False).head(top_n).index
        cumplimiento_top = cumplimiento.loc[top_users]
        ax = cumplimiento_top.plot(
            kind="barh", stacked=True, color=["salmon", "mediumseagreen"], figsize=(8, 4)
        )
        ax.set_title(f"Tareas cumplidas vs no cumplidas por operario (Top {top_n})")
        ax.set_xlabel("N° de tareas")
        ax.set_ylabel("Operario")
        ax.legend(title="Cumple", labels=["No", "Sí"])
        for bars in ax.containers:
            ax.bar_label(bars, label_type="center", fontsize=8, color="black")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_ops == "Distribución de tareas entre operarios":
        conteo = df_ops["Usuario Modificac"].value_counts()
        plt.figure(figsize=(6, 4))
        sns.histplot(conteo, bins=20, color=PALETTE_OPS[4])
        plt.xlabel("Tareas finalizadas por operario")
        plt.title("Distribución de tareas entre operarios")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif sub_ops == "Ranking gerencial de operarios":
        # Scatter: Top N operarios por tareas vs tiempo total
        df_kpi = (
            df_ops.groupby("Usuario Modificac")
            .agg(
                Tareas=("Nro Tarea", "count"),
                Tiempo_total=("Duracion_min", "sum")
            )
            .reset_index()
        )
        top_n = st.sidebar.slider("¿Cuántos operarios mostrar?", min_value=5, max_value=50, value=slider_default, key="ger")
        top_kpi = df_kpi.sort_values("Tareas", ascending=False).head(top_n)
        plt.figure(figsize=(7, 5))
        plt.scatter(top_kpi["Tareas"], top_kpi["Tiempo_total"], s=180, alpha=0.8)
        for i, row in top_kpi.iterrows():
            plt.text(row["Tareas"], row["Tiempo_total"], row["Usuario Modificac"], fontsize=9, alpha=0.8)
        plt.xlabel("Tareas realizadas")
        plt.ylabel("Tiempo total (min)")
        plt.title(f"Dispersión: Tareas realizadas vs. Tiempo total (Top {top_n} operarios)")
        plt.tight_layout()
        st.pyplot(plt.gcf())