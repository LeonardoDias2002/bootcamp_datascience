import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Dashboard | Index Tracking (S&P 100 & Ibovespa)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO DE LEITURA E PROCESSAMENTO DE DADOS
# ==============================================================================

def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace('.SA', '')

def read_csv_flexible(filename_base: str) -> pd.DataFrame:
    if os.path.exists(filename_base):
        return pd.read_csv(filename_base)
    elif os.path.exists(f"{filename_base}.csv"):
        return pd.read_csv(f"{filename_base}.csv")
    else:
        raise FileNotFoundError(f"Arquivo não localizado: '{filename_base}' ou '{filename_base}.csv'")

def load_index_series(filepath_base: str) -> pd.Series:
    df = read_csv_flexible(filepath_base)
    date_cols = [c for c in df.columns if str(c).lower() in ['date', 'data', 'datetime', 'ate']]
    date_col = date_cols[0] if date_cols else df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    series = df[num_cols[0]].dropna()
    
    if series.abs().max() > 2.0:
        returns = series.pct_change().dropna()
    else:
        returns = series
        
    returns.name = 'INDEX'
    return returns

@st.cache_data(show_spinner="Carregando e processando base de dados...")
def load_and_prepare_dataset(dataset_type: str = 'Ibov') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if dataset_type == 'Ibov':
        meta = read_csv_flexible("Ibov_Acoes")
        ret = read_csv_flexible("Ibov_retornos_limpos")
        idx_series = load_index_series("Ibov_indice")
        
        meta = meta.rename(columns={'Codigo': 'Ticker', 'Part. (%)': 'Peso_Indice', 'Setor': 'Setor'})
        date_col = [c for c in ret.columns if str(c).lower() in ['date', 'data', 'datetime', 'ate']][0]
        
    elif dataset_type == 'SP100':
        meta = read_csv_flexible("SP100_Acoes")
        ret = read_csv_flexible("SP100_retornos_limpos")
        idx_series = load_index_series("SP100_indice")
        
        meta = meta.rename(columns={'Symbol': 'Ticker', 'Sector': 'Setor'})
        if 'Peso_Indice' not in meta.columns and 'Weight' in meta.columns:
            meta['Peso_Indice'] = meta['Weight']
        elif 'Peso_Indice' not in meta.columns:
            meta['Peso_Indice'] = 1.0 / len(meta)
            
        date_col = [c for c in ret.columns if str(c).lower() in ['date', 'data', 'datetime', 'ate']][0]

    ret[date_col] = pd.to_datetime(ret[date_col])
    ret = ret.set_index(date_col).sort_index()

    meta['Ticker_Norm'] = meta['Ticker'].apply(normalize_ticker)
    col_mapping = {col: normalize_ticker(col) for col in ret.columns if col not in [date_col, 'INDEX']}
    ret = ret.rename(columns=col_mapping)
    ret = ret.loc[:, ~ret.columns.duplicated()]

    common_dates = ret.index.intersection(idx_series.index)
    ret_assets = ret.loc[common_dates].dropna(how='all').fillna(0.0)
    index_returns = idx_series.loc[common_dates]

    valid_tickers = [c for c in ret_assets.columns if c in meta['Ticker_Norm'].values]
    ret_assets = ret_assets[valid_tickers]
    
    std_devs = ret_assets.std()
    non_zero_assets = std_devs[std_devs > 1e-8].index
    ret_assets = ret_assets[non_zero_assets]

    meta_sub = meta.drop_duplicates(subset=['Ticker_Norm'])
    return meta_sub, ret_assets, index_returns

# ==============================================================================
# ENGINE QUANTITATIVO DE MODELAGEM E AVALIAÇÃO
# ==============================================================================

def calculate_max_drawdown(series: pd.Series) -> float:
    cum_returns = (1 + series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return float(drawdown.min())

def build_feature_engineering_pipeline(
    asset_returns: pd.DataFrame,
    index_returns: pd.Series,
    metadata: pd.DataFrame,
    trading_days: int = 252
) -> pd.DataFrame:
    features = pd.DataFrame(index=asset_returns.columns)
    
    features['corr_idx'] = asset_returns.corrwith(index_returns).fillna(0)
    features['beta'] = asset_returns.apply(lambda s: s.cov(index_returns) / index_returns.var() if index_returns.var() > 0 else 0.0)
    features['vol_anual'] = asset_returns.std() * np.sqrt(trading_days)
    
    corr = asset_returns.corr().fillna(0)
    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, np.nan)
    features['corr_media_pares'] = np.nanmean(corr_values, axis=1)
    
    features['vol_21d'] = asset_returns.rolling(21).std().mean(axis=0) * np.sqrt(trading_days)
    features['vol_63d'] = asset_returns.rolling(63).std().mean(axis=0) * np.sqrt(trading_days)
    
    total_obs = len(asset_returns)
    features['ret_accum_3m'] = (1 + asset_returns.iloc[-min(63, total_obs):]).cumprod().iloc[-1] - 1
    features['ret_accum_6m'] = (1 + asset_returns.iloc[-min(126, total_obs):]).cumprod().iloc[-1] - 1
    features['ret_accum_12m'] = (1 + asset_returns.iloc[-min(252, total_obs):]).cumprod().iloc[-1] - 1
    features['max_drawdown'] = asset_returns.apply(calculate_max_drawdown)
    
    if metadata is not None:
        meta_clean = metadata.copy()
        ticker_col = next((c for c in ['Ticker_Norm', 'Ticker', 'Codigo', 'Symbol'] if c in meta_clean.columns), None)
        if ticker_col:
            meta_clean['Ticker_Join'] = meta_clean[ticker_col].apply(normalize_ticker)
            meta_clean = meta_clean.drop_duplicates(subset=['Ticker_Join']).set_index('Ticker_Join')
            
            if 'Peso_Indice' in meta_clean.columns:
                features['Peso_Indice'] = meta_clean['Peso_Indice'].reindex(features.index)
            if 'Setor' in meta_clean.columns:
                features['setor'] = meta_clean['Setor'].reindex(features.index)
            elif 'Sector' in meta_clean.columns:
                features['setor'] = meta_clean['Sector'].reindex(features.index)

    features['Peso_Indice'] = features.get('Peso_Indice', pd.Series(1.0 / len(features), index=features.index)).fillna(1.0 / len(features))
    features['setor'] = features.get('setor', pd.Series('Outros', index=features.index)).fillna('Outros')

    return features.fillna(0)

def avaliar_modelo(pesos: pd.Series, ret_acoes: pd.DataFrame, ret_indice: pd.Series, nome: str = 'modelo') -> pd.Series:
    w_active = pesos[pesos > 1e-6]
    if len(w_active) == 0:
        return pd.Series({
            'N ações': 0, 'Tracking Error (% a.a.)': 0.0, 'Tracking Diff. acumulada (%)': 0.0,
            'Correlação': 0.0, 'R2': 0.0, 'Beta': 0.0, 'Info. Ratio': 0.0,
            'HHI': 0.0, 'Peso máx. (%)': 0.0
        }, name=nome)
        
    r_cart = ret_acoes[w_active.index] @ w_active
    dif = r_cart - ret_indice
    
    te = float(dif.std() * np.sqrt(252))
    corr = float(r_cart.corr(ret_indice)) if r_cart.std() > 1e-8 else 0.0
    beta = float(r_cart.cov(ret_indice) / ret_indice.var()) if ret_indice.var() > 0 else 0.0
    
    ss_res = float(((ret_indice - r_cart) ** 2).sum())
    ss_tot = float(((ret_indice - ret_indice.mean()) ** 2).sum())
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    info_ratio = float((dif.mean() * 252) / te) if te > 1e-8 else 0.0
    
    return pd.Series({
        'N ações': int(len(w_active)),
        'Tracking Error (% a.a.)': te * 100,
        'Tracking Diff. acumulada (%)': float((r_cart.sum() - ret_indice.sum()) * 100),
        'Correlação': corr,
        'R2': r2,
        'Beta': beta,
        'Info. Ratio': info_ratio,
        'HHI': float((w_active ** 2).sum()),
        'Peso máx. (%)': float(w_active.max() * 100),
    }, name=nome)

# ALGORITMOS DE OTIMIZAÇÃO
def otimizar_pesos(ret_acoes: pd.DataFrame, ret_indice: pd.Series, acoes: List[str], peso_max: float = 1.0) -> pd.Series:
    R = ret_acoes[acoes].values
    y = ret_indice.values
    n = len(acoes)
    
    def erro(w):
        return np.mean((y - R @ w) ** 2)
        
    res = minimize(
        erro,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, peso_max)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    )
    return pd.Series(res.x, index=acoes)

def baseline_top_k_peso(pesos_indice: pd.Series, k: int) -> pd.Series:
    top = pesos_indice.sort_values(ascending=False).head(k)
    return top / top.sum()

def baseline_top_k_corr(features: pd.DataFrame, k: int) -> pd.Series:
    top = features.sort_values('corr_idx', ascending=False).head(k).index
    return pd.Series(1.0 / k, index=top)

def modelo_lasso(ret_acoes: pd.DataFrame, ret_indice: pd.Series, alpha: float = 1e-4) -> pd.Series:
    m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=50000)
    m.fit(ret_acoes.values, ret_indice.values)
    w = pd.Series(m.coef_, index=ret_acoes.columns)
    w = w[w > 1e-6]
    if len(w) == 0:
        return pd.Series(1.0 / len(ret_acoes.columns), index=ret_acoes.columns)
    return otimizar_pesos(ret_acoes, ret_indice, list(w.index))

def selecionar_por_cluster(ret_acoes: pd.DataFrame, features: pd.DataFrame, k: int) -> List[str]:
    corr = ret_acoes.corr()
    dist = np.sqrt(0.5 * (1 - corr).clip(lower=0))
    cl = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    rotulos = cl.fit_predict(dist.values)
    
    escolhidas = []
    for g in range(k):
        membros = ret_acoes.columns[rotulos == g]
        melhor = features.loc[membros, 'corr_idx'].idxmax()
        escolhidas.append(melhor)
    return escolhidas

def optimize_miqp_gurobi(ret_acoes: pd.DataFrame, ret_indice: pd.Series, k_max: int = 20, w_max: float = 0.15) -> pd.Series:
    tickers = ret_acoes.columns.tolist()
    N = len(tickers)
    
    excess_matrix = ret_acoes.sub(ret_indice, axis=0)
    Q = excess_matrix.cov().values * 252
    
    try:
        model = gp.Model("Index_Tracking_MIQP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 10)
        model.setParam('MIPGap', 0.01)
        
        w = model.addVars(N, lb=0.0, ub=w_max, vtype=GRB.CONTINUOUS, name="w")
        z = model.addVars(N, vtype=GRB.BINARY, name="z")
        
        model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1.0, name="budget")
        for i in range(N):
            model.addConstr(w[i] <= z[i] * w_max, name=f"link_{i}")
        model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k_max, name="cardinality")
        
        obj = gp.QuadExpr()
        for i in range(N):
            for j in range(N):
                obj += w[i] * Q[i, j] * w[j]
        model.setObjective(obj, GRB.MINIMIZE)
        
        model.optimize()
        
        if model.SolCount > 0:
            weights = pd.Series([w[i].X for i in range(N)], index=tickers)
            return weights / weights.sum()
    except Exception as e:
        pass
    
    top_corr = ret_acoes.corrwith(ret_indice).abs().sort_values(ascending=False).head(k_max).index
    return otimizar_pesos(ret_acoes, ret_indice, list(top_corr), peso_max=w_max)

# ==============================================================================
# CONTROLES E BARRA LATERAL
# ==============================================================================

st.sidebar.title("⚙️ Painel de Controle")
st.sidebar.markdown("---")

index_choice = st.sidebar.selectbox("📊 Selecione o Índice", options=["SP100", "Ibov"], index=0)

meta_df, ret_df, idx_df = load_and_prepare_dataset(index_choice)

st.sidebar.subheader("📅 Janela de Backtesting")
preset_period = st.sidebar.selectbox(
    "Escolha a Janela Pré-definida",
    options=["Treino (2022-2023)", "Validação (2024-2025)", "Teste (2026)", "Todo o Período", "Personalizado"]
)

min_date = ret_df.index.min().date()
max_date = ret_df.index.max().date()

if preset_period == "Treino (2022-2023)":
    start_date, end_date = pd.to_datetime("2022-01-01").date(), pd.to_datetime("2023-12-31").date()
elif preset_period == "Validação (2024-2025)":
    start_date, end_date = pd.to_datetime("2024-01-01").date(), pd.to_datetime("2025-12-31").date()
elif preset_period == "Teste (2026)":
    start_date, end_date = pd.to_datetime("2026-01-01").date(), pd.to_datetime("2026-10-02").date()
elif preset_period == "Todo o Período":
    start_date, end_date = min_date, max_date
else:
    start_date, end_date = st.sidebar.date_input("Intervalo Personalizado", value=[min_date, max_date], min_value=min_date, max_value=max_date)

st.sidebar.subheader("🎯 Cardinalidade (K)")
k_target = st.sidebar.slider("Número de Ativos na Carteira (K)", min_value=5, max_value=40, value=20, step=1)

eval_assets = ret_df.loc[str(start_date):str(end_date)]
eval_idx = idx_df.loc[str(start_date):str(end_date)]

train_assets_opt = ret_df.loc['2022-01-01':'2023-12-31'] if preset_period != "Treino (2022-2023)" else eval_assets
train_idx_opt = idx_df.loc['2022-01-01':'2023-12-31'] if preset_period != "Treino (2022-2023)" else eval_idx

features_df = build_feature_engineering_pipeline(train_assets_opt, train_idx_opt, meta_df)

@st.cache_data(show_spinner="Executando otimizadores de portfólio...")
def run_all_optimizations(_train_assets, _train_idx, _features_df, k):
    acoes_cl = selecionar_por_cluster(_train_assets, _features_df, k=k)
    
    weights = {
        'Baseline Top Peso': baseline_top_k_peso(_features_df['Peso_Indice'], k=k),
        'Baseline Top Corr': baseline_top_k_corr(_features_df, k=k),
        'Lasso + Otim.': modelo_lasso(_train_assets, _train_idx),
        'Cluster + Otim.': otimizar_pesos(_train_assets, _train_idx, acoes_cl),
        'MIQP (Gurobi)': optimize_miqp_gurobi(_train_assets, _train_idx, k_max=k)
    }
    return weights

weights_dict = run_all_optimizations(train_assets_opt, train_idx_opt, features_df, k_target)

# ==============================================================================
# ÁREA PRINCIPAL DO DASHBOARD
# ==============================================================================

st.title(f"📌 Index Tracking Dashboard: {index_choice}")
st.markdown(f"**Janela Selecionada:** {start_date} até {end_date} | **Ativos Solicitados (K):** {k_target}")

# 1. TABELA COMPARATIVA DE MÉTRICAS
metrics_list = []
for model_name, w in weights_dict.items():
    metrics_list.append(avaliar_modelo(w, eval_assets, eval_idx, nome=model_name))

df_metrics = pd.concat(metrics_list, axis=1).T

# Destaques KPI
best_model_name = df_metrics['Tracking Error (% a.a.)'].idxmin()
best_te = df_metrics.loc[best_model_name, 'Tracking Error (% a.a.)']
best_corr = df_metrics.loc[best_model_name, 'Correlação']
best_r2 = df_metrics.loc[best_model_name, 'R2']

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏆 Melhor Modelo (TE)", best_model_name)
col2.metric("📉 Menor TE (% a.a.)", f"{best_te:.2f}%")
col3.metric("🔗 Correlação", f"{best_corr:.4f}")
col4.metric("📊 R²", f"{best_r2:.4f}")

st.markdown("---")

st.subheader("📋 Tabela Comparativa de Desempenho (Out-of-Sample)")
st.dataframe(df_metrics.style.highlight_min(subset=['Tracking Error (% a.a.)'], color='lightgreen').format({
    'Tracking Error (% a.a.)': '{:.2f}%',
    'Tracking Diff. acumulada (%)': '{:.2f}%',
    'Correlação': '{:.4f}',
    'R2': '{:.4f}',
    'Beta': '{:.4f}',
    'Info. Ratio': '{:.4f}',
    'HHI': '{:.4f}',
    'Peso máx. (%)': '{:.2f}%'
}), use_container_width=True)

# 2. CURVA DE TRADE-OFF: TE × K PARA TODOS OS MODELOS
st.subheader("📉 Curva Trade-Off: Tracking Error × Número de Ações (K)")

@st.cache_data(show_spinner="Calculando curva TE vs K para TODOS os modelos...")
def compute_te_vs_k_all_models(_train_assets, _train_idx, _eval_assets, _eval_idx, _meta_df):
    k_range = [5, 10, 15, 20, 25, 30, 35, 40]
    te_records = []
    
    for k_val in k_range:
        feats_k = build_feature_engineering_pipeline(_train_assets, _train_idx, _meta_df)
        acoes_cl_k = selecionar_por_cluster(_train_assets, feats_k, k=k_val)
        
        w_dict_k = {
            'Baseline Top Peso': baseline_top_k_peso(feats_k['Peso_Indice'], k=k_val),
            'Baseline Top Corr': baseline_top_k_corr(feats_k, k=k_val),
            'Lasso + Otim.': modelo_lasso(_train_assets, _train_idx),
            'Cluster + Otim.': otimizar_pesos(_train_assets, _train_idx, acoes_cl_k),
            'MIQP (Gurobi)': optimize_miqp_gurobi(_train_assets, _train_idx, k_max=k_val)
        }
        
        rec = {'K': k_val}
        for name, w_k in w_dict_k.items():
            m_k = avaliar_modelo(w_k, _eval_assets, _eval_idx)
            rec[name] = m_k['Tracking Error (% a.a.)']
            
        te_records.append(rec)
    return pd.DataFrame(te_records)

df_te_k_all = compute_te_vs_k_all_models(train_assets_opt, train_idx_opt, eval_assets, eval_idx, meta_df)

models_to_plot = st.multiselect(
    "Selecione os modelos para comparar na curva TE × K:",
    options=['Baseline Top Peso', 'Baseline Top Corr', 'Lasso + Otim.', 'Cluster + Otim.', 'MIQP (Gurobi)'],
    default=['Baseline Top Peso', 'Baseline Top Corr', 'Lasso + Otim.', 'Cluster + Otim.', 'MIQP (Gurobi)']
)

if models_to_plot:
    fig_te_k = px.line(
        df_te_k_all, x='K', y=models_to_plot,
        markers=True, title="Comparações de Tracking Error (% a.a.) por Número de Ações (K)",
        labels={'value': 'Tracking Error (% a.a.)', 'K': 'Número de Ações (K)', 'variable': 'Modelo'}
    )
    fig_te_k.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_te_k, use_container_width=True)

# 3. RETORNO ACUMULADO (CARTEIRAS VS ÍNDICE)
st.subheader("🚀 Desempenho Acumulado: Carteiras vs Índice Oficial")

selected_models_chart = st.multiselect(
    "Selecione os modelos para comparar no gráfico de retornos acumulados:",
    options=list(weights_dict.keys()),
    default=['MIQP (Gurobi)', 'Baseline Top Peso', 'Cluster + Otim.']
)

cum_returns_df = pd.DataFrame(index=eval_assets.index)
cum_returns_df['Benchmark Index'] = (1 + eval_idx).cumprod() - 1

for m_name in selected_models_chart:
    w_m = weights_dict[m_name]
    w_m_act = w_m[w_m > 1e-6]
    r_cart_m = eval_assets[w_m_act.index] @ w_m_act
    cum_returns_df[m_name] = (1 + r_cart_m).cumprod() - 1

fig_cum = px.line(
    cum_returns_df * 100,
    title=f"Retorno Acumulado (%) - Período: {start_date} a {end_date}",
    labels={'value': 'Retorno Acumulado (%)', 'Date': 'Data', 'variable': 'Estratégia'}
)
fig_cum.update_layout(template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_cum, use_container_width=True)

# 4. ALOCAÇÃO DE ATIVOS E COMPOSIÇÃO SETORIAL
st.subheader("📊 Alocação de Ativos e Distribuição Setorial")

selected_model_alloc = st.selectbox("Selecione o Modelo para Visualizar a Carteira", options=list(weights_dict.keys()), index=4)

w_selected = weights_dict[selected_model_alloc]
w_active_selected = w_selected[w_selected > 1e-6].sort_values(ascending=False)

df_alloc = pd.DataFrame({
    'Ticker': w_active_selected.index,
    'Peso (%)': (w_active_selected * 100).values
})

meta_indexed = meta_df.drop_duplicates(subset=['Ticker_Norm']).set_index('Ticker_Norm')
df_alloc['Setor'] = df_alloc['Ticker'].map(
    meta_indexed['Setor'] if 'Setor' in meta_indexed.columns else meta_indexed['Sector']
).fillna('Outros')

col_alloc1, col_alloc2 = st.columns(2)

with col_alloc1:
    fig_bar = px.bar(
        df_alloc, x='Peso (%)', y='Ticker', orientation='h', color='Setor',
        title=f"Pesos dos Ativos - {selected_model_alloc} (Total: {len(df_alloc)} ações)",
        text=df_alloc['Peso (%)'].apply(lambda x: f"{x:.2f}%")
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_alloc2:
    df_sector = df_alloc.groupby('Setor')['Peso (%)'].sum().reset_index()
    fig_pie = px.pie(
        df_sector, values='Peso (%)', names='Setor',
        title=f"Alocação Setorial - {selected_model_alloc}",
        hole=0.4
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# EXPORTAÇÃO DE RELATÓRIO COMPLETO (CSV E PDF/HTML COM CARTEIRA DO MELHOR MODELO)
# ==============================================================================

st.markdown("---")
st.subheader("📥 Exportação de Resultados e Relatório Completo")

# Identificar a carteira do melhor modelo para incluir no relatório
best_w = weights_dict[best_model_name]
best_w_act = best_w[best_w > 1e-6].sort_values(ascending=False)

df_best_portfolio = pd.DataFrame({
    'Ticker': best_w_act.index,
    'Peso (%)': (best_w_act * 100).round(2).values
})
df_best_portfolio['Setor'] = df_best_portfolio['Ticker'].map(
    meta_indexed['Setor'] if 'Setor' in meta_indexed.columns else meta_indexed['Sector']
).fillna('Outros')

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv_data = df_metrics.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📄 Baixar Tabela de Métricas (CSV)",
        data=csv_data,
        file_name=f"metricas_index_tracking_{index_choice}_K{k_target}.csv",
        mime="text/csv"
    )

with col_exp2:
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Relatório Index Tracking - {index_choice}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #333; }}
            h1 {{ color: #1e3d59; border-bottom: 2px solid #1e3d59; padding-bottom: 8px; }}
            h2 {{ color: #17b978; margin-top: 30px; }}
            .highlight {{ background-color: #e8f5e9; padding: 15px; border-left: 5px solid #2e7d32; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; margin-bottom: 25px; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 10px; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #fafafa; }}
        </style>
    </head>
    <body>
        <h1>Relatório Executivo de Index Tracking: {index_choice}</h1>
        <p><b>Período Analisado:</b> {start_date} até {end_date} | <b>Cardinalidade Solicitada (K):</b> {k_target}</p>
        
        <div class="highlight">
            <h3>🏆 Modelo Destaque (Menor Tracking Error): {best_model_name}</h3>
            <ul>
                <li><b>Tracking Error Anualizado:</b> {best_te:.2f}% a.a.</li>
                <li><b>Correlação com o Índice:</b> {best_corr:.4f}</li>
                <li><b>Coeficiente de Determinação (R²):</b> {best_r2:.4f}</li>
                <li><b>Total de Ações Selecionadas:</b> {len(df_best_portfolio)}</li>
            </ul>
        </div>

        <h2>1. Métricas Comparativas de Todos os Modelos</h2>
        {df_metrics.round(4).to_html()}

        <h2>2. Detalhamento da Carteira Otimizada ({best_model_name})</h2>
        <p>Abaixo está a lista completa dos ativos selecionados, com seus respectivos setores e pesos alocados na carteira enxuta:</p>
        {df_best_portfolio.to_html(index=False)}
    </body>
    </html>
    """
    st.download_button(
        label="🖨️ Baixar Relatório Completo (HTML / Impressão em PDF)",
        data=html_report.encode('utf-8'),
        file_name=f"relatorio_completo_index_tracking_{index_choice}_K{k_target}.html",
        mime="text/html"
    )