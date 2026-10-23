import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.cluster import AgglomerativeClustering

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (PADRÃO ENTREGA 1)
# ==============================================================================
st.set_page_config(
    page_title="Spot-Checking de Index Tracking",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta oficial extraída da Apresentação 1 e do Notebook do Grupo
CORES = {
    "sp100": "#0F4C81",  # Azul Marinho Principal (S&P 100)
    "sp100_sec": "#2B6999",  # Azul Secundário (Apresentação 1)
    "ibov": "#1B4D3E",  # Verde Esmeralda Fechado (IBOVESPA)
    "ibov_sec": "#2E7D67",  # Verde Secundário
    "alerta": "#C53030",  # Vermelho Destaque / Alerta
    "laranja": "#D97706",  # Laranja Sóbrio
    "roxo": "#4C51BF",  # Indigo Sóbrio
    "neutro": "#88837D",  # Cinza dos gráficos da Apresentação 1
    "grafite": "#1A202C",  # Preto/Grafite (Benchmark)
}

# CSS Limpo que sobrescreve variáveis nativas do Dark Mode do Streamlit
st.markdown(
    """
<style>
    /* 1. Sobrescreve as variáveis globais de tema do Streamlit para forçar Light Mode real */
    :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        --text-color: #1E293B !important;
        --background-color: #FFFFFF !important;
        --secondary-background-color: #F8FAFC !important;
        --primary-color: #0F4C81 !important;
        color-scheme: light !important;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* 2. Sidebar e Inputs (Data de Corte e Selectbox com fundo claro e letra escura) */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1E293B !important;
    }

    /* Caixa do DateInput (Data de Corte) e Selectbox (Período de Avaliação) */
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] div[data-baseweb="base-input"],
    [data-testid="stDateInput"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 600 !important;
        border-color: #CBD5E1 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] input {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    /* Menu suspenso (Dropdown Popover) quando aberto */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[role="listbox"] ul,
    div[role="listbox"] li {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    /* Tags dentro do Multiselect */
    span[data-baseweb="tag"] {
        background-color: #0F4C81 !important;
        color: #FFFFFF !important;
    }
    span[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* 3. Abas (st.tabs) com contraste nítido em todas as abas (ativas e inativas) */
    div[data-testid="stTabs"] button[data-baseweb="tab"],
    div[data-baseweb="tab-list"] button {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-bottom: 2px solid #CBD5E1 !important;
        border-radius: 6px 6px 0 0 !important;
        margin-right: 6px !important;
        padding: 10px 18px !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] *,
    div[data-baseweb="tab-list"] button * {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        border-bottom: 3px solid #0F4C81 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] *,
    div[data-baseweb="tab-list"] button[aria-selected="true"] * {
        color: #0F4C81 !important;
        -webkit-text-fill-color: #0F4C81 !important;
        font-weight: 700 !important;
    }

    /* 4. Cabeçalho no estilo da Apresentação 1 */
    .finor-tag {
        font-size: 0.82rem;
        font-weight: 700;
        color: #2B6999;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .finor-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1A202C;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .finor-subtitle {
        font-size: 0.98rem;
        color: #555555;
        margin-bottom: 1.1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #EDF2F7;
    }
    .sec-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #1A202C;
        margin-top: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .sec-sub {
        font-size: 0.88rem;
        color: #666666;
        margin-bottom: 0.75rem;
    }

    /* 5. Cards de KPI customizados em HTML (100% legíveis em qualquer tema) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 1.25rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0F4C81;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569 !important;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.50rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 6px;
    }
    .kpi-delta {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
    }
    .kpi-delta.good {
        background-color: #DCFCE7;
        color: #166534 !important;
    }
    .kpi-delta.neutral {
        background-color: #E2E8F0;
        color: #1E293B !important;
    }

    /* 6. Tabela Comparativa HTML limpa */
    .finor-table-container {
        overflow-x: auto;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
    table.finor-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-size: 0.88rem;
    }
    table.finor-table thead tr {
        background-color: #0F4C81 !important;
        color: #FFFFFF !important;
        text-align: right;
    }
    table.finor-table thead th:first-child {
        text-align: left;
    }
    table.finor-table th {
        padding: 10px 12px;
        font-weight: 700;
        color: #FFFFFF !important;
        white-space: nowrap;
    }
    table.finor-table td {
        padding: 9px 12px;
        border-bottom: 1px solid #EDF2F7;
        text-align: right;
        color: #1E293B !important;
        white-space: nowrap;
    }
    table.finor-table td:first-child {
        text-align: left;
        font-weight: 600;
        color: #0F172A !important;
    }
    table.finor-table tbody tr:nth-child(even) {
        background-color: #F8FAFC !important;
    }
    table.finor-table tbody tr.best-row {
        background-color: #ECFDF5 !important;
        font-weight: 700 !important;
    }
    .badge-best {
        background-color: #1B4D3E;
        color: #FFFFFF !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS E LIMPEZA SEM VAZAMENTO TEMPORAL (COM CACHE)
# ==============================================================================
TRADUCAO_SP100 = {
    "Information Technology": "Tecnologia da Informação",
    "Health Care": "Saúde",
    "Financials": "Financeiro",
    "Industrials": "Industrial",
    "Communication Services": "Serviços de Comunicação",
    "Consumer Discretionary": "Consumo Discricionário",
    "Consumer Staples": "Consumo Básico",
    "Energy": "Energia",
    "Utilities": "Utilidades Públicas",
    "Real Estate": "Imobiliário",
    "Materials": "Materiais",
}


@st.cache_data
def carregar_mercado(mercado: str, data_corte_treino: str):
    if mercado == "S&P 100 (EUA)":
        precos = pd.read_csv("sp100_precos.csv", index_col="Date", parse_dates=True)
        indice = pd.read_csv(
            "sp100_indice.csv", index_col="Date", parse_dates=True
        ).squeeze()
        try:
            df_set = pd.read_csv("SP100_wikipedia_21set2026.csv")
            df_set["Sector"] = df_set["Sector"].map(lambda x: TRADUCAO_SP100.get(x, x))
            mapa_set = dict(zip(df_set["Symbol"].str.strip(), df_set["Sector"]))
        except Exception:
            mapa_set = {}
    else:
        precos = pd.read_csv("ibov_precos.csv", index_col="Date", parse_dates=True)
        indice = pd.read_csv(
            "ibov_indice.csv", index_col="Date", parse_dates=True
        ).squeeze()
        try:
            df_set = pd.read_csv("IBOVESPA_B3_24set2026.csv")
            mapa_set = dict(zip(df_set["Codigo"].str.strip() + ".SA", df_set["Setor"]))
        except Exception:
            mapa_set = {}

    if isinstance(indice, pd.DataFrame):
        indice = indice.iloc[:, 0]

    # Filtro de missing values (<= 5%) estritamente no período de treino
    pct_nulos = precos.loc[:data_corte_treino].isna().mean() * 100
    validas = pct_nulos[pct_nulos <= 5.0].index.tolist()

    precos_clean = precos[validas].ffill().dropna(how="any")
    indice_clean = indice.ffill().dropna()

    ret_acoes = np.log(precos_clean / precos_clean.shift(1)).dropna(how="any")
    ret_idx = np.log(indice_clean / indice_clean.shift(1)).dropna()

    datas_comuns = ret_acoes.index.intersection(ret_idx.index)
    ret_acoes = ret_acoes.loc[datas_comuns]
    ret_idx = ret_idx.loc[datas_comuns]

    setores_map = {col: mapa_set.get(col, "Outros") for col in ret_acoes.columns}
    return ret_acoes, ret_idx, setores_map


# ==============================================================================
# 3. ENGENHARIA DE FEATURES E FAMÍLIAS DE MODELOS (SPOT-CHECKING)
# ==============================================================================
def calcular_features_treino(treino_X, treino_y, setores_map):
    var_idx = treino_y.var()
    vol_idx = treino_y.std() * np.sqrt(252)
    feats = pd.DataFrame(index=treino_X.columns)
    feats["corr_idx"] = treino_X.corrwith(treino_y)
    feats["beta"] = treino_X.apply(lambda s: s.cov(treino_y) / var_idx)
    feats["vol_anual"] = treino_X.std() * np.sqrt(252)
    feats["vol_gap_abs"] = (feats["vol_anual"] - vol_idx).abs()
    feats["te_ind"] = treino_X.sub(treino_y, axis=0).std() * np.sqrt(252)
    feats["setor"] = feats.index.map(setores_map).fillna("Outros")
    return feats.sort_values("corr_idx", ascending=False)


def filtrar_colinearidade(treino_X, feats, max_cand=55, lim_corr=0.90):
    cands = feats.sort_values("corr_idx", ascending=False).head(max_cand).index.tolist()
    corr_m = treino_X[cands].corr()
    sel = []
    for a in cands:
        if all(abs(corr_m.loc[a, s]) < lim_corr for s in sel):
            sel.append(a)
    return sel


def resolver_slsqp(treino_X, treino_y, acoes, peso_max=0.35, lambda_vol=0.0):
    R = treino_X[acoes].values
    y = treino_y.values
    n = len(acoes)
    cov_anual = np.cov(R, rowvar=False) * 252.0
    vol_idx = np.std(y, ddof=1) * np.sqrt(252.0)

    def func_obj(w):
        mse = np.mean((y - R @ w) ** 2)
        if lambda_vol > 0:
            v_c = np.sqrt(max(w.T @ cov_anual @ w, 1e-12))
            return mse + lambda_vol * (((v_c - vol_idx) ** 2) / 252.0)
        return mse

    w0 = np.ones(n) / n
    ub = min(1.0, max(peso_max, 1.0 / n + 1e-3))
    res = minimize(
        func_obj,
        x0=w0,
        method="SLSQP",
        bounds=[(0.0, ub)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-9, "maxiter": 400},
    )
    w = pd.Series(np.maximum(res.x, 0.0), index=acoes)
    w = w[w > 1e-5]
    return w / w.sum()


@st.cache_data
def executar_familia_modelos(
    treino_X, treino_y, setores_map, k: int, peso_max: float, lambda_vol: float
):
    feats = calcular_features_treino(treino_X, treino_y, setores_map)
    sem_red = filtrar_colinearidade(
        treino_X, feats, max_cand=max(55, k * 2), lim_corr=0.90
    )
    top_k_corr = feats.head(k).index.tolist()
    top_k_sem_red = sem_red[:k]

    # 1. Baseline 1: Top K Correlação (Pesos Iguais 1/K)
    w_b1 = pd.Series(1.0 / k, index=top_k_corr)

    # 2. Baseline 2: Top K Correlação (Pesos Inverso da Volatilidade 1/σ)
    inv_v = 1.0 / feats.loc[top_k_corr, "vol_anual"]
    w_b2 = inv_v / inv_v.sum()

    # 3. Otimização Restrita SLSQP Pura (Sem Redundância, Foco Retorno)
    w_slsqp = resolver_slsqp(
        treino_X, treino_y, top_k_sem_red, peso_max=peso_max, lambda_vol=0.0
    )

    # 4. SLSQP + Ajuste de Volatilidade (Retorno + Volatilidade)
    w_vol = resolver_slsqp(
        treino_X, treino_y, top_k_sem_red, peso_max=peso_max, lambda_vol=lambda_vol
    )

    # 5. Lasso Positivo (Seleção L1 Automática via Bisseção + Reotimização SLSQP)
    X_v, y_v = treino_X.values, treino_y.values
    low, high = 1e-7, 1e-2
    sup_lasso = None
    for _ in range(25):
        mid = np.sqrt(low * high)
        m = Lasso(
            alpha=mid,
            positive=True,
            fit_intercept=False,
            max_iter=20000,
            random_state=42,
        )
        m.fit(X_v, y_v)
        pos = treino_X.columns[m.coef_ > 1e-6].tolist()
        if 1 <= len(pos) <= k:
            sup_lasso = pos
            if len(pos) == k:
                break
            high = mid
        elif len(pos) > k:
            low = mid
        else:
            high = mid
    if not sup_lasso or len(sup_lasso) < min(3, k):
        m = Lasso(
            alpha=2e-5,
            positive=True,
            fit_intercept=False,
            max_iter=20000,
            random_state=42,
        )
        m.fit(X_v, y_v)
        sup_lasso = (
            pd.Series(m.coef_, index=treino_X.columns)
            .sort_values(ascending=False)
            .head(k)
            .index.tolist()
        )
    w_lasso = resolver_slsqp(
        treino_X,
        treino_y,
        sup_lasso[:k],
        peso_max=peso_max,
        lambda_vol=lambda_vol * 0.5,
    )

    # 6. Clustering Hierárquico de Correlação + Representantes SLSQP (Com cópia gravável)
    corr = treino_X.corr().clip(-1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    dist_matrix = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_matrix, 0.0)
    try:
        cl = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
    except TypeError:
        cl = AgglomerativeClustering(
            n_clusters=k, affinity="precomputed", linkage="average"
        )
    lbls = cl.fit_predict(dist_matrix)
    rep_cluster = []
    for g in range(k):
        membros = treino_X.columns[lbls == g]
        score = feats.loc[membros, "corr_idx"] / feats.loc[membros, "te_ind"]
        rep_cluster.append(score.idxmax())
    w_cluster = resolver_slsqp(
        treino_X, treino_y, rep_cluster, peso_max=peso_max, lambda_vol=lambda_vol * 0.5
    )

    carteiras = {
        "1. Baseline: Top K Corr (1/K)": w_b1,
        "2. Baseline: Top K Inv-Vol (1/σ)": w_b2,
        "3. SLSQP Puro (Sem Redundância)": w_slsqp,
        "4. SLSQP + Ajuste de Volatilidade": w_vol,
        "5. Lasso Positivo (L1 + SLSQP)": w_lasso,
        "6. Clustering + SLSQP": w_cluster,
    }
    return carteiras, feats


def calcular_kpis(pesos, ret_X, ret_y, nome):
    r_c = ret_X[pesos.index] @ pesos
    dif = r_c - ret_y
    te = dif.std() * np.sqrt(252) * 100
    vol_c = r_c.std() * np.sqrt(252) * 100
    vol_i = ret_y.std() * np.sqrt(252) * 100
    gap_v = abs(vol_c - vol_i)
    acum_c = (np.exp(r_c.sum()) - 1.0) * 100
    acum_i = (np.exp(ret_y.sum()) - 1.0) * 100
    td = acum_c - acum_i
    corr = r_c.corr(ret_y)
    beta = r_c.cov(ret_y) / ret_y.var()
    ss_res = ((ret_y - r_c) ** 2).sum()
    ss_tot = ((ret_y - ret_y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    hhi = (pesos**2).sum()
    return (
        pd.Series(
            {
                "K Efetivo": int((pesos > 1e-5).sum()),
                "TE (% a.a.)": round(te, 2),
                "Vol. Carteira (%)": round(vol_c, 2),
                "Gap Vol. vs Índice (p.p.)": round(gap_v, 2),
                "Track. Diff. Acum. (%)": round(td, 2),
                "Correlação (r)": round(corr, 4),
                "R²": round(r2, 4),
                "Beta (β)": round(beta, 3),
                "HHI": round(hhi, 3),
                "Peso Máx. (%)": round(pesos.max() * 100, 1),
            },
            name=nome,
        ),
        r_c,
    )


def renderizar_tabela_html(df, col_destaque="TE (% a.a.)", cor_header="#0F4C81"):
    """Gera tabela HTML com fundo claro garantido e alto contraste."""
    melhor_idx = df[col_destaque].idxmin() if col_destaque in df.columns else None
    html = [f'<div class="finor-table-container"><table class="finor-table">']
    html.append(
        f'<thead style="background-color: {cor_header} !important;"><tr><th>Família de Modelo</th>'
    )
    for c in df.columns:
        html.append(
            f'<th style="background-color: {cor_header} !important; color: #FFFFFF !important;">{c}</th>'
        )
    html.append("</tr></thead><tbody>")

    for idx, row in df.iterrows():
        is_best = idx == melhor_idx
        tr_class = ' class="best-row"' if is_best else ""
        nome_exib = (
            f'{idx} <span class="badge-best">Melhor TE</span>' if is_best else idx
        )
        html.append(f"<tr{tr_class}><td>{nome_exib}</td>")
        for c in df.columns:
            val = row[c]
            if isinstance(val, (int, np.integer)) or c == "K Efetivo":
                txt = f"{int(val)}"
            elif c in ["Correlação (r)", "R²"]:
                txt = f"{val:.4f}"
            elif c in ["Beta (β)", "HHI"]:
                txt = f"{val:.3f}"
            elif isinstance(val, (float, np.floating)):
                txt = f"{val:.2f}"
            else:
                txt = str(val)
            html.append(f"<td>{txt}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


# ==============================================================================
# 4. SIDEBAR: CONTROLES DO EXPERIMENTO DE SPOT-CHECKING
# ==============================================================================
st.sidebar.markdown("### >>>>> PARÂMETROS DO MODELO")
mercado_sel = st.sidebar.radio(
    "Índice de Referência (Benchmark)", ["S&P 100 (EUA)", "IBOVESPA (Brasil)"], index=0
)

is_ibov = "IBOV" in mercado_sel
cor_primaria = CORES["ibov"] if is_ibov else CORES["sp100"]
cor_secundaria = CORES["ibov_sec"] if is_ibov else CORES["sp100_sec"]

k_val = st.sidebar.slider(
    "Cardinalidade da Carteira (K ações)", min_value=5, max_value=40, value=15, step=1
)
peso_max_val = (
    st.sidebar.slider(
        "Peso Máximo por Ação (%)", min_value=10, max_value=100, value=30, step=5
    )
    / 100.0
)
lambda_vol_val = st.sidebar.slider(
    "Ajuste de Volatilidade (λ_vol)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Controla o ajuste em dois estágios: aproxima a volatilidade anualizada da carteira da volatilidade do índice.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### >>>>> DIVISÃO TEMPORAL")
ret_acoes_full, ret_idx_full, setores_map = carregar_mercado(mercado_sel, "2024-12-31")
min_d, max_d = ret_acoes_full.index.min().date(), ret_acoes_full.index.max().date()

default_corte = (
    pd.to_datetime("2025-01-01").date()
    if min_d < pd.to_datetime("2024-06-01").date()
    else pd.to_datetime("2025-06-01").date()
)
data_corte = st.sidebar.date_input(
    "Data de Corte (Fim do Treino)",
    value=default_corte,
    min_value=min_d,
    max_value=max_d,
)

janela_aval = st.sidebar.selectbox(
    "Período de Avaliação Exibido",
    ["Fora da Amostra (Out-of-Sample)", "Dentro da Amostra (Treino In-Sample)"],
)

# Recarregar respeitando exatamente o corte de treino escolhido
ret_acoes, ret_idx, setores_map = carregar_mercado(mercado_sel, str(data_corte))
treino_X = ret_acoes.loc[: str(data_corte)]
treino_y = ret_idx.loc[: str(data_corte)]
oos_X = (
    ret_acoes.loc[str(data_corte) :].iloc[1:]
    if len(ret_acoes.loc[str(data_corte) :]) > 1
    else ret_acoes.loc[str(data_corte) :]
)
oos_y = ret_idx.loc[oos_X.index]

aval_X = oos_X if "Fora" in janela_aval else treino_X
aval_y = oos_y if "Fora" in janela_aval else treino_y

carteiras, feats_treino = executar_familia_modelos(
    treino_X, treino_y, setores_map, k_val, peso_max_val, lambda_vol_val
)

# Cálculo de KPIs na janela selecionada e no treino
lista_kpis_aval, series_ret_aval = [], {}
lista_kpis_treino = []

for nome_m, w_m in carteiras.items():
    kpi_a, r_a = calcular_kpis(w_m, aval_X, aval_y, nome_m)
    kpi_t, _ = calcular_kpis(w_m, treino_X, treino_y, nome_m)
    lista_kpis_aval.append(kpi_a)
    lista_kpis_treino.append(kpi_t)
    series_ret_aval[nome_m] = r_a

df_kpis = pd.DataFrame(lista_kpis_aval)
df_kpis_tr = pd.DataFrame(lista_kpis_treino)
df_kpis["TE Treino (% a.a.)"] = df_kpis_tr["TE (% a.a.)"]
df_kpis["Overfitting (p.p.)"] = (
    df_kpis["TE (% a.a.)"] - df_kpis["TE Treino (% a.a.)"]
).round(2)

melhor_modelo = df_kpis["TE (% a.a.)"].idxmin()
te_melhor = df_kpis.loc[melhor_modelo, "TE (% a.a.)"]
te_base1 = df_kpis.loc["1. Baseline: Top K Corr (1/K)", "TE (% a.a.)"]
ganho_te = te_base1 - te_melhor
vol_idx_atual = aval_y.std() * np.sqrt(252) * 100

# ==============================================================================
# 5. CABEÇALHO E KPIS (ESTÉTICA DA APRESENTAÇÃO 1 COM ALTO CONTRASTE)
# ==============================================================================
st.markdown(
    '<div class="finor-tag">>>>>> FINOR × BOOTCAMP DATA SCIENCE 2026 • PARTE 2</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="finor-title">Spot-Checking de Modelos de Index Tracking: {mercado_sel}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="finor-subtitle">Comparação de 6 famílias de algoritmos para replicar o índice com <b>K = {k_val} ações</b> '
    f'(de {ret_acoes.shape[1]} válidas). Período analisado ({janela_aval}): <b>{aval_X.index[0].strftime("%d/%m/%Y")} a '
    f'{aval_X.index[-1].strftime("%d/%m/%Y")}</b> ({len(aval_X)} pregões).</div>',
    unsafe_allow_html=True,
)

nome_curto_vencedor = (
    melhor_modelo.split(": ")[-1]
    if ": " in melhor_modelo
    else melhor_modelo.split(". ")[-1]
)
vol_mod_4 = df_kpis.loc["4. SLSQP + Ajuste de Volatilidade", "Vol. Carteira (%)"]
gap_mod_4 = df_kpis.loc[
    "4. SLSQP + Ajuste de Volatilidade", "Gap Vol. vs Índice (p.p.)"
]

st.markdown(
    f"""
<div class="kpi-grid">
    <div class="kpi-card" style="border-left-color: {cor_primaria};">
        <div class="kpi-label">MENOR TRACKING ERROR</div>
        <div class="kpi-value">{te_melhor:.2f}% a.a.</div>
        <span class="kpi-delta good">{-ganho_te:+.2f} p.p. vs. Baseline 1/K</span>
    </div>
    <div class="kpi-card" style="border-left-color: {cor_primaria};">
        <div class="kpi-label">FAMÍLIA VENCEDORA</div>
        <div class="kpi-value">{nome_curto_vencedor}</div>
        <span class="kpi-delta good">R² = {df_kpis.loc[melhor_modelo, 'R²']:.3f} | r = {df_kpis.loc[melhor_modelo, 'Correlação (r)']:.3f}</span>
    </div>
    <div class="kpi-card" style="border-left-color: {cor_primaria};">
        <div class="kpi-label">VOLATILIDADE (SLSQP + VOL)</div>
        <div class="kpi-value">{vol_mod_4:.2f}% a.a.</div>
        <span class="kpi-delta neutral">Índice: {vol_idx_atual:.2f}% (Gap: {gap_mod_4:.2f} p.p.)</span>
    </div>
    <div class="kpi-card" style="border-left-color: {cor_primaria};">
        <div class="kpi-label">DESVIO ACUMULADO (MELHOR)</div>
        <div class="kpi-value">{df_kpis.loc[melhor_modelo, 'Track. Diff. Acum. (%)']:+.2f}%</div>
        <span class="kpi-delta neutral">Concentração HHI: {df_kpis.loc[melhor_modelo, 'HHI']:.3f}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. FUNÇÃO PADRONIZADA PARA GRÁFICOS PLOTLY (EIXOS E LEGENDAS 100% VISÍVEIS)
# ==============================================================================
def configurar_grafico_padrao(fig, altura=470, y_title="", x_title="Data"):
    """Garante textos de eixos, ticks e legendas em cor escura #222222 visíveis em qualquer tema."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=altura,
        font=dict(family="sans-serif", size=12, color="#222222"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF", font_size=12, font_color="#1A202C", bordercolor="#CBD5E1"
        ),
        margin=dict(l=65, r=25, t=55, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11.5, color="#1A202C"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12.5, color="#1A202C")),
            tickfont=dict(size=11, color="#333333"),
            showgrid=False,
            showline=True,
            linecolor="#CCCCCC",
            linewidth=1,
            ticks="outside",
            tickcolor="#CCCCCC",
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12.5, color="#1A202C")),
            tickfont=dict(size=11, color="#333333"),
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=1,
            griddash="dash",
            showline=True,
            linecolor="#CCCCCC",
            linewidth=1,
            zeroline=False,
        ),
    )
    return fig


# ==============================================================================
# 7. ABAS DE ANÁLISE
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Desempenho & Spot-Checking",
        "2. Ajuste de Volatilidade",
        "3. Curva Erro × Nº de Ações (K)",
        "4. Pesos da Carteira & Setores",
    ]
)

# ------------------------------------------------------------------------------
# ABA 1: DESEMPENHO ACUMULADO E TABELA COMPARATIVA
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        '<div class="sec-title">>>>>> Performance ao longo do tempo: Índice vs. Famílias de Modelos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Evolução comparativa normalizada (Base 100 = R$/US$ 100 no início da janela de avaliação)</div>',
        unsafe_allow_html=True,
    )

    modelos_visiveis = st.multiselect(
        "Selecione as famílias de modelos para comparar no gráfico:",
        options=list(carteiras.keys()),
        default=[
            "1. Baseline: Top K Corr (1/K)",
            "2. Baseline: Top K Inv-Vol (1/σ)",
            "4. SLSQP + Ajuste de Volatilidade",
            "5. Lasso Positivo (L1 + SLSQP)",
            "6. Clustering + SLSQP",
        ],
    )

    idx_acum = np.exp(aval_y.cumsum()) * 100.0
    fig_acum = go.Figure()

    # Linha de referência Base 100 idêntica à Apresentação 1
    fig_acum.add_hline(y=100, line_dash="dot", line_color="#A0AEC0", line_width=1.2)

    # Benchmark
    fig_acum.add_trace(
        go.Scatter(
            x=idx_acum.index,
            y=idx_acum,
            mode="lines",
            name=f"Benchmark ({mercado_sel})",
            line=dict(color=CORES["grafite"], width=3.2),
        )
    )

    mapa_cores_mod = {
        "1. Baseline: Top K Corr (1/K)": (CORES["neutro"], "dot", 2.0),
        "2. Baseline: Top K Inv-Vol (1/σ)": ("#4A5568", "dash", 2.0),
        "3. SLSQP Puro (Sem Redundância)": (cor_secundaria, "dashdot", 2.2),
        "4. SLSQP + Ajuste de Volatilidade": (cor_primaria, "solid", 3.0),
        "5. Lasso Positivo (L1 + SLSQP)": (CORES["alerta"], "solid", 2.3),
        "6. Clustering + SLSQP": (CORES["laranja"], "solid", 2.4),
    }

    for m_nome in modelos_visiveis:
        s_acum = np.exp(series_ret_aval[m_nome].cumsum()) * 100.0
        c_m, d_m, w_m = mapa_cores_mod[m_nome]
        fig_acum.add_trace(
            go.Scatter(
                x=s_acum.index,
                y=s_acum,
                mode="lines",
                name=f"{m_nome} (TE: {df_kpis.loc[m_nome, 'TE (% a.a.)']:.2f}%)",
                line=dict(color=c_m, width=w_m, dash=d_m),
            )
        )

    # Anotação de destaque para o maior choque diário no período
    dia_choque = aval_y.abs().idxmax()
    val_choque = idx_acum.loc[dia_choque]
    fig_acum.add_annotation(
        x=dia_choque,
        y=val_choque,
        text=f"Maior oscilação diária ({aval_y.loc[dia_choque]*100:+.2f}%)<br>{dia_choque.strftime('%d/%m/%Y')}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-42,
        font=dict(size=10.5, color=CORES["alerta"]),
        bgcolor="#FFF5F5",
        bordercolor=CORES["alerta"],
    )

    fig_acum = configurar_grafico_padrao(
        fig_acum,
        altura=470,
        y_title="Evolução do Patrimônio (Base 100 = R$/US$ 100)",
        x_title="Data",
    )
    st.plotly_chart(fig_acum, use_container_width=True)

    st.markdown(
        '<div class="sec-title">>>>>> Tabela Comparativa de Desempenho (Spot-Checking)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Métricas calculadas fora da amostra ordenadas pelo menor Tracking Error anualizado</div>',
        unsafe_allow_html=True,
    )

    df_exib = df_kpis.sort_values("TE (% a.a.)")
    st.markdown(
        renderizar_tabela_html(
            df_exib, col_destaque="TE (% a.a.)", cor_header=cor_primaria
        ),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# ABA 2: AJUSTE DE VOLATILIDADE
# ------------------------------------------------------------------------------
with tab2:
    col_v1, col_v2 = st.columns([1.1, 1.0])
    with col_v1:
        st.markdown(
            '<div class="sec-title">>>>>> Aderência de Volatilidade: Carteiras vs. Índice</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sec-sub">Volatilidade Anualizada (% a.a.) de cada família comparada à volatilidade do índice (linha tracejada)</div>',
            unsafe_allow_html=True,
        )

        df_vol_plot = df_kpis.sort_values("Gap Vol. vs Índice (p.p.)", ascending=False)
        cores_bar_vol = [
            (
                cor_primaria
                if ("Ajuste de Volatilidade" in idx or "Clustering" in idx)
                else (CORES["alerta"] if "1/K" in idx else CORES["neutro"])
            )
            for idx in df_vol_plot.index
        ]

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(
                y=df_vol_plot.index,
                x=df_vol_plot["Vol. Carteira (%)"],
                orientation="h",
                marker_color=cores_bar_vol,
                text=[
                    f"{v:.2f}% (Gap: {g:.2f} p.p.)"
                    for v, g in zip(
                        df_vol_plot["Vol. Carteira (%)"],
                        df_vol_plot["Gap Vol. vs Índice (p.p.)"],
                    )
                ],
                textposition="outside",
                textfont=dict(color="#1A202C", size=11),
            )
        )
        fig_vol.add_vline(
            x=vol_idx_atual, line_dash="dash", line_color=CORES["alerta"], line_width=2
        )
        fig_vol.add_annotation(
            x=vol_idx_atual,
            y=len(df_vol_plot) - 0.4,
            text=f"Volatilidade do Índice: {vol_idx_atual:.2f}%",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=CORES["alerta"]),
        )
        fig_vol = configurar_grafico_padrao(
            fig_vol,
            altura=420,
            x_title="Volatilidade Anualizada (% a.a.)",
            y_title="Família de Modelo",
        )
        fig_vol.update_xaxes(
            range=[0, max(df_vol_plot["Vol. Carteira (%)"].max(), vol_idx_atual) * 1.28]
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_v2:
        st.markdown(
            '<div class="sec-title">>>>>> Controle de Exposição às Ações Mais Voláteis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sec-sub">Comparativo de alocação nos ativos de maior volatilidade anual (Baseline 1/K vs. SLSQP + Ajuste de Vol.)</div>',
            unsafe_allow_html=True,
        )

        w_igual = carteiras["1. Baseline: Top K Corr (1/K)"]
        w_vol_mod = carteiras["4. SLSQP + Ajuste de Volatilidade"]
        uniao_ativos = w_igual.index.union(w_vol_mod.index)

        df_comp_vol = pd.DataFrame(
            {
                "Setor": [setores_map.get(a, "Outros") for a in uniao_ativos],
                "Vol. Anual (%)": (
                    feats_treino.loc[uniao_ativos, "vol_anual"] * 100
                ).round(2),
                "Beta (β)": feats_treino.loc[uniao_ativos, "beta"].round(2),
                "Peso 1/K (%)": (
                    w_igual.reindex(uniao_ativos, fill_value=0.0) * 100
                ).round(2),
                "Peso SLSQP+Vol (%)": (
                    w_vol_mod.reindex(uniao_ativos, fill_value=0.0) * 100
                ).round(2),
            },
            index=[a.replace(".SA", "") for a in uniao_ativos],
        ).sort_values("Vol. Anual (%)", ascending=False)

        st.markdown(
            renderizar_tabela_html(
                df_comp_vol.head(10), col_destaque="", cor_header=cor_primaria
            ),
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------------------
# ABA 3: CURVA DE TRACKING ERROR X CARDINALIDADE (K)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(
        '<div class="sec-title">>>>>> Fronteira de Cardinalidade: Tracking Error × Número de Ações (K)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Identificação do ponto de retorno decrescente à medida que adicionamos mais ativos na carteira</div>',
        unsafe_allow_html=True,
    )

    ks_grid = [5, 8, 10, 12, 15, 20, 25, 30, 35]
    linhas_k = []
    for k_i in ks_grid:
        c_i, _ = executar_familia_modelos(
            treino_X, treino_y, setores_map, k_i, peso_max_val, lambda_vol_val
        )
        for m_n in [
            "1. Baseline: Top K Corr (1/K)",
            "2. Baseline: Top K Inv-Vol (1/σ)",
            "4. SLSQP + Ajuste de Volatilidade",
            "5. Lasso Positivo (L1 + SLSQP)",
            "6. Clustering + SLSQP",
        ]:
            kpi_i, _ = calcular_kpis(c_i[m_n], aval_X, aval_y, m_n)
            linhas_k.append({"K": k_i, "Modelo": m_n, "TE": kpi_i["TE (% a.a.)"]})

    df_curva_k = pd.DataFrame(linhas_k)
    fig_k = go.Figure()
    for m_n in df_curva_k["Modelo"].unique():
        sub = df_curva_k[df_curva_k["Modelo"] == m_n]
        c_m, d_m, w_m = mapa_cores_mod[m_n]
        fig_k.add_trace(
            go.Scatter(
                x=sub["K"],
                y=sub["TE"],
                mode="lines+markers",
                name=m_n,
                line=dict(color=c_m, width=w_m, dash=d_m),
                marker=dict(size=7),
            )
        )

    fig_k.add_vline(
        x=k_val, line_dash="dash", line_color=CORES["alerta"], line_width=1.5
    )
    fig_k.add_annotation(
        x=k_val,
        y=df_curva_k["TE"].max() * 0.94,
        text=f"Corte Atual: K = {k_val} ações",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font=dict(color=CORES["alerta"], size=11.5),
    )
    fig_k = configurar_grafico_padrao(
        fig_k,
        altura=450,
        x_title="Número Máximo de Ações na Carteira (K)",
        y_title="Tracking Error Anualizado (% a.a.)",
    )
    st.plotly_chart(fig_k, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 4: ALOCAÇÃO DE PESOS E DIVERSIFICAÇÃO SETORIAL
# ------------------------------------------------------------------------------
with tab4:
    mod_inspecao = st.selectbox(
        "Selecione a família de modelo para inspecionar os pesos e setores:",
        list(carteiras.keys()),
        index=3,
    )
    w_inspecao = carteiras[mod_inspecao].sort_values(ascending=True)

    col_p1, col_p2 = st.columns([1.1, 1.0])
    with col_p1:
        st.markdown(
            f'<div class="sec-title">>>>>> Composição da Carteira ({len(w_inspecao)} ações ativas)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sec-sub">{mod_inspecao} — Alocação percentual por ativo (Soma = 100%)</div>',
            unsafe_allow_html=True,
        )

        labels_y = [
            f"{t.replace('.SA', '')} ({setores_map.get(t, 'Outros')})"
            for t in w_inspecao.index
        ]
        fig_w = go.Figure(
            go.Bar(
                y=labels_y,
                x=w_inspecao.values * 100,
                orientation="h",
                marker_color=cor_primaria,
                text=[f"{v*100:.1f}%" for v in w_inspecao.values],
                textposition="outside",
                textfont=dict(color="#1A202C", size=11),
            )
        )
        fig_w = configurar_grafico_padrao(
            fig_w,
            altura=max(400, len(w_inspecao) * 28),
            x_title="Peso Alocado na Carteira (%)",
            y_title="Ativo (Setor)",
        )
        fig_w.update_xaxes(range=[0, max(w_inspecao.values * 100) * 1.25])
        st.plotly_chart(fig_w, use_container_width=True)

    with col_p2:
        st.markdown(
            '<div class="sec-title">>>>>> Equilíbrio Setorial: Carteira vs. Índice</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sec-sub">Peso total alocado por setor econômico comparado à distribuição do benchmark</div>',
            unsafe_allow_html=True,
        )

        s_set_cart = (
            pd.Series(
                w_inspecao.values * 100,
                index=[setores_map.get(t, "Outros") for t in w_inspecao.index],
            )
            .groupby(level=0)
            .sum()
        )
        s_set_univ = (
            pd.Series(
                1.0, index=[setores_map.get(t, "Outros") for t in ret_acoes.columns]
            )
            .groupby(level=0)
            .sum()
        )
        s_set_univ = (s_set_univ / s_set_univ.sum()) * 100

        df_set_comp = (
            pd.DataFrame(
                {"Carteira (%)": s_set_cart, "Universo do Índice (%)": s_set_univ}
            )
            .fillna(0.0)
            .sort_values("Carteira (%)", ascending=True)
        )

        fig_set = go.Figure()
        fig_set.add_trace(
            go.Bar(
                y=df_set_comp.index,
                x=df_set_comp["Universo do Índice (%)"],
                name="Proporção no Índice (%)",
                orientation="h",
                marker_color=CORES["neutro"],
            )
        )
        fig_set.add_trace(
            go.Bar(
                y=df_set_comp.index,
                x=df_set_comp["Carteira (%)"],
                name="Peso na Carteira (%)",
                orientation="h",
                marker_color=cor_primaria,
            )
        )
        fig_set.update_layout(barmode="group")
        fig_set = configurar_grafico_padrao(
            fig_set,
            altura=max(400, len(df_set_comp) * 38),
            x_title="Participação Setorial (%)",
            y_title="Setor Econômico",
        )
        st.plotly_chart(fig_set, use_container_width=True)
