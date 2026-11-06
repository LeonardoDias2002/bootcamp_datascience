import os
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, enet_path
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (NYT EDITORIAL + PALETA FINOR)
# ==============================================================================
st.set_page_config(
    page_title='Index Tracking: S&P 100 & IBOVESPA',
    layout='wide',
    initial_sidebar_state='expanded',
)

CORES: Dict[str, str] = {
    'sp100': '#0F4C81',  # Azul Marinho Principal (S&P 100)
    'sp100_sec': '#2B6999',  # Azul Secundário
    'ibov': '#1B4D3E',  # Verde Esmeralda Fechado (IBOVESPA)
    'ibov_sec': '#2E7D67',  # Verde Secundário
    'alerta': '#C53030',  # Vermelho Destaque / Alerta
    'laranja': '#D97706',  # Âmbar Sóbrio
    'roxo': '#4C51BF',  # Indigo Sóbrio
    'neutro': '#88837D',  # Cinza Histórico / Baseline
    'grafite': '#1A202C',  # Preto/Grafite (Benchmark)
}

DOMINIOS_SP100: Dict[str, str] = {
    'AAPL': 'apple.com',
    'MSFT': 'microsoft.com',
    'NVDA': 'nvidia.com',
    'AMZN': 'amazon.com',
    'GOOG': 'abc.xyz',
    'GOOGL': 'abc.xyz',
    'META': 'meta.com',
    'BRK-B': 'berkshirehathaway.com',
    'LLY': 'lilly.com',
    'AVGO': 'broadcom.com',
    'JPM': 'jpmorganchase.com',
    'TSLA': 'tesla.com',
    'WMT': 'walmart.com',
    'V': 'visa.com',
    'XOM': 'exxonmobil.com',
    'UNH': 'unitedhealthgroup.com',
    'MA': 'mastercard.com',
    'PG': 'pg.com',
    'COST': 'costco.com',
    'JNJ': 'jnj.com',
    'ORCL': 'oracle.com',
    'HD': 'homedepot.com',
    'BAC': 'bankofamerica.com',
    'ABBV': 'abbvie.com',
    'KO': 'coca-colacompany.com',
    'NFLX': 'netflix.com',
    'MRK': 'merck.com',
    'CVX': 'chevron.com',
    'CRM': 'salesforce.com',
    'AMD': 'amd.com',
    'PEP': 'pepsico.com',
    'TMO': 'thermofisher.com',
    'LIN': 'linde.com',
    'TMUS': 't-mobile.com',
    'WFC': 'wellsfargo.com',
    'ADBE': 'adobe.com',
    'CSCO': 'cisco.com',
    'MCD': 'mcdonalds.com',
    'ABT': 'abbott.com',
    'PM': 'pmi.com',
    'NOW': 'servicenow.com',
    'IBM': 'ibm.com',
    'AXP': 'americanexpress.com',
    'MS': 'morganstanley.com',
    'GE': 'ge.com',
    'CAT': 'caterpillar.com',
    'ISRG': 'intuitive.com',
    'TXN': 'ti.com',
    'GS': 'goldmansachs.com',
    'INTU': 'intuit.com',
    'VZ': 'verizon.com',
    'DIS': 'thewaltdisneycompany.com',
    'AMGN': 'amgen.com',
    'QCOM': 'qualcomm.com',
    'BKNG': 'bookingholdings.com',
    'PF': 'pfizer.com',
    'PFE': 'pfizer.com',
    'T': 'att.com',
    'BLK': 'blackrock.com',
    'RTX': 'rtx.com',
    'NEE': 'nexteraenergy.com',
    'CMCSA': 'corporate.comcast.com',
    'DHR': 'danaher.com',
    'UBER': 'uber.com',
    'SPGI': 'spglobal.com',
    'LOW': 'lowes.com',
    'UNP': 'up.com',
    'LMT': 'lockheedmartin.com',
    'HON': 'honeywell.com',
    'COP': 'conocophillips.com',
    'PLTR': 'palantir.com',
    'SYK': 'stryker.com',
    'SCHW': 'schwab.com',
    'AMAT': 'appliedmaterials.com',
    'LRCX': 'lamresearch.com',
    'MU': 'micron.com',
    'ANET': 'arista.com',
    'PANW': 'paloaltonetworks.com',
    'DELL': 'dell.com',
    'C': 'citigroup.com',
    'BA': 'boeing.com',
    'GILD': 'gilead.com',
    'MDT': 'medtronic.com',
    'DE': 'deere.com',
    'UPS': 'ups.com',
    'SBUX': 'starbucks.com',
    'MDLZ': 'mondelezinternational.com',
    'BMY': 'bms.com',
    'SO': 'southerncompany.com',
    'DUK': 'duke-energy.com',
    'MO': 'altria.com',
    'CVS': 'cvshealth.com',
    'GD': 'gd.com',
    'FDX': 'fedex.com',
    'EMR': 'emerson.com',
    'MMM': '3m.com',
    'TGT': 'target.com',
    'USB': 'usbank.com',
    'BNY': 'bny.com',
    'COF': 'capitalone.com',
    'GM': 'gm.com',
    'INTC': 'intel.com',
    'ACN': 'accenture.com',
    'AMT': 'americantower.com',
}

DOMINIOS_IBOV: Dict[str, str] = {
    'VALE3': 'vale.com',
    'PETR4': 'petrobras.com.br',
    'PETR3': 'petrobras.com.br',
    'ITUB4': 'itau.com.br',
    'ITSA4': 'itausa.com.br',
    'BBDC4': 'bradesco.com.br',
    'BBDC3': 'bradesco.com.br',
    'BBAS3': 'bb.com.br',
    'WEGE3': 'weg.net',
    'ABEV3': 'ambev.com.br',
    'B3SA3': 'b3.com.br',
    'BPAC11': 'btgpactual.com',
    'RENT3': 'localiza.com',
    'SUZB3': 'suzano.com.br',
    'RDOR3': 'rededorsaoluiz.com.br',
    'EQTL3': 'equatorialenergia.com.br',
    'PRIO3': 'prio3.com.br',
    'RADL3': 'rdsaude.com.br',
    'RAIL3': 'rumolog.com',
    'SBSP3': 'sabesp.com.br',
    'VIVT3': 'telefonica.com.br',
    'ENEV3': 'eneva.com.br',
    'BBSE3': 'bbseguros.com.br',
    'GGBR4': 'gerdau.com',
    'GOAU4': 'gerdau.com',
    'CMIG4': 'cemig.com.br',
    'CPLE3': 'copel.com',
    'UGPA3': 'ultra.com.br',
    'TIMS3': 'tim.com.br',
    'LREN3': 'lojasrenner.com.br',
    'TOTS3': 'totvs.com',
    'KLBN11': 'klabin.com.br',
    'ENGI11': 'energisa.com.br',
    'EGIE3': 'engie.com.br',
    'CSAN3': 'cosan.com.br',
    'VBBR3': 'vibraenergia.com.br',
    'ASAI3': 'assai.com.br',
    'HYPE3': 'hypera.com.br',
    'CPFE3': 'cpfl.com.br',
    'SANB11': 'santander.com.br',
    'MULT3': 'multiplan.com.br',
    'TAEE11': 'taesa.com.br',
    'CMIN3': 'csnmineracao.com.br',
    'CSNA3': 'csn.com.br',
    'CYRE3': 'cyrela.com.br',
    'MGLU3': 'magazineluiza.com.br',
    'HAPV3': 'hapvida.com.br',
    'FLRY3': 'fleury.com.br',
    'BRAP4': 'bradespar.com.br',
    'ALOS3': 'allos.co',
    'IGTI11': 'iguatemi.com.br',
    'AURE3': 'aurenenergia.com.br',
    'SMFT3': 'smartfit.com.br',
    'YDUQ3': 'yduqs.com.br',
    'COGN3': 'cogna.com.br',
    'USIM5': 'usiminas.com',
    'BEEF3': 'minervafoods.com',
    'MRVE3': 'mrv.com.br',
    'DIRR3': 'direcional.com.br',
    'CURY3': 'cury.net',
    'POMO4': 'marcopolo.com.br',
    'VAMO3': 'vamos.com.br',
    'CEAB3': 'cea.com.br',
    'CXSE3': 'caixaseguridade.com.br',
    'PSSA3': 'portoseguro.com.br',
    'CSMG3': 'copasa.com.br',
}

TRADUCAO_SP100: Dict[str, str] = {
    'Information Technology': 'Tecnologia da Informação',
    'Health Care': 'Saúde',
    'Financials': 'Financeiro',
    'Industrials': 'Industrial',
    'Communication Services': 'Serviços de Comunicação',
    'Consumer Discretionary': 'Consumo Discricionário',
    'Consumer Staples': 'Consumo Básico',
    'Energy': 'Energia',
    'Utilities': 'Utilidades Públicas',
    'Real Estate': 'Imobiliário',
    'Materials': 'Materiais',
}

# CSS sem espaços de recuo para evitar qualquer falha de parsing no navegador
CSS_GLOBAL = """<style>
:root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
color-scheme: light !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
background-color: #FFFFFF !important;
color: #1A202C !important;
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
[data-testid="stSidebar"] {
background-color: #F8FAFC !important;
border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * {
color: #1E293B !important;
}
label, [data-testid="stWidgetLabel"] *, .stMarkdown p {
color: #1E293B !important;
-webkit-text-fill-color: #1E293B !important;
}
[role="tablist"] {
gap: 6px !important;
border-bottom: 2px solid #CBD5E1 !important;
}
[role="tab"] {
background-color: #E2E8F0 !important;
border: 1px solid #CBD5E1 !important;
border-bottom: none !important;
border-radius: 6px 6px 0 0 !important;
padding: 10px 18px !important;
opacity: 1 !important;
}
[role="tab"]:hover {
background-color: #CBD5E1 !important;
}
[role="tab"], [role="tab"] *, [role="tab"] p {
color: #0F172A !important;
-webkit-text-fill-color: #0F172A !important;
font-weight: 700 !important;
font-size: 0.94rem !important;
opacity: 1 !important;
}
[role="tab"][aria-selected="true"] {
background-color: #0F4C81 !important;
border-color: #0F4C81 !important;
}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *, [role="tab"][aria-selected="true"] p {
color: #FFFFFF !important;
-webkit-text-fill-color: #FFFFFF !important;
font-weight: 800 !important;
}
[role="tab"] .react-aria-SelectionIndicator {
background-color: #0F4C81 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
input[type="number"] {
background-color: #F8FAFC !important;
color: #0F172A !important;
-webkit-text-fill-color: #0F172A !important;
border: 1px solid #CBD5E1 !important;
font-weight: 600 !important;
}
div[data-baseweb="select"] * {
color: #0F172A !important;
-webkit-text-fill-color: #0F172A !important;
}
span[data-baseweb="tag"] {
background-color: #0F4C81 !important;
color: #FFFFFF !important;
}
span[data-baseweb="tag"] * {
color: #FFFFFF !important;
-webkit-text-fill-color: #FFFFFF !important;
}
.header-box {
display: flex;
align-items: center;
gap: 18px;
padding-bottom: 0.85rem;
margin-bottom: 1.1rem;
border-bottom: 2px solid #EDF2F7;
}
.idx-emblem {
width: 64px;
height: 64px;
border-radius: 12px;
display: flex;
flex-direction: column;
align-items: center;
justify-content: center;
color: #FFFFFF !important;
font-weight: 900;
flex-shrink: 0;
box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}
.finor-tag {
font-size: 0.80rem;
font-weight: 700;
color: #2B6999;
letter-spacing: 0.06em;
text-transform: uppercase;
margin-bottom: 0.1rem;
}
.finor-title {
font-size: 1.80rem;
font-weight: 800;
color: #1A202C;
line-height: 1.18;
margin-bottom: 0.2rem;
}
.finor-subtitle {
font-size: 0.95rem;
color: #555555;
}
.sec-title {
font-size: 1.20rem;
font-weight: 700;
color: #1A202C;
margin-top: 0.5rem;
margin-bottom: 0.1rem;
}
.sec-sub {
font-size: 0.88rem;
color: #666666;
margin-bottom: 0.75rem;
}
.kpi-grid {
display: grid;
grid-template-columns: repeat(4, 1fr);
gap: 14px;
margin-bottom: 1.2rem;
}
.kpi-card {
background-color: #F8FAFC;
border: 1px solid #E2E8F0;
border-left: 4px solid #0F4C81;
border-radius: 6px;
padding: 12px 16px;
}
.kpi-label {
font-size: 0.74rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.04em;
color: #475569 !important;
margin-bottom: 4px;
}
.kpi-value {
font-size: 1.48rem;
font-weight: 800;
color: #0F172A !important;
margin-bottom: 5px;
}
.kpi-delta {
display: inline-block;
font-size: 0.76rem;
font-weight: 600;
padding: 2px 8px;
border-radius: 999px;
}
.kpi-delta.good { background-color: #DCFCE7; color: #166534 !important; }
.kpi-delta.neutral { background-color: #E2E8F0; color: #1E293B !important; }
.stocks-grid {
display: grid;
grid-template-columns: repeat(auto-fill, minmax(235px, 1fr));
gap: 12px;
margin-top: 0.6rem;
margin-bottom: 1.2rem;
}
.stock-card {
background-color: #FFFFFF;
border: 1px solid #CBD5E1;
border-radius: 8px;
padding: 12px 14px;
display: flex;
align-items: center;
gap: 12px;
transition: box-shadow 0.15s ease;
}
.stock-card:hover {
box-shadow: 0 4px 12px rgba(0,0,0,0.08);
border-color: #94A3B8;
}
.logo-wrap {
position: relative;
width: 44px;
height: 44px;
border-radius: 8px;
background-color: #E2E8F0;
border: 1px solid #CBD5E1;
display: flex;
align-items: center;
justify-content: center;
overflow: hidden;
flex-shrink: 0;
}
.logo-fallback {
font-size: 0.76rem;
font-weight: 800;
color: #1E293B !important;
letter-spacing: -0.02em;
z-index: 1;
user-select: none;
}
.logo-img {
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
object-fit: contain;
background-color: #FFFFFF;
padding: 4px;
z-index: 2;
}
.stock-info {
flex: 1;
min-width: 0;
}
.stock-top {
display: flex;
justify-content: space-between;
align-items: baseline;
}
.stock-ticker {
font-size: 0.95rem;
font-weight: 800;
color: #0F172A !important;
}
.stock-weight {
font-size: 0.92rem;
font-weight: 800;
color: #0F4C81 !important;
}
.stock-name {
font-size: 0.78rem;
color: #334155 !important;
white-space: nowrap;
overflow: hidden;
text-overflow: ellipsis;
margin-top: 1px;
}
.stock-sector {
font-size: 0.72rem;
color: #64748B !important;
white-space: nowrap;
overflow: hidden;
text-overflow: ellipsis;
}
.stock-price {
font-size: 0.78rem;
font-weight: 700;
color: #334155 !important;
margin-top: 1px;
}
.weight-bar-bg {
width: 100%;
height: 4px;
background-color: #E2E8F0;
border-radius: 2px;
margin-top: 6px;
overflow: hidden;
}
.finor-table-container {
overflow-x: auto;
margin-top: 0.5rem;
margin-bottom: 1.2rem;
border: 1px solid #CBD5E1;
border-radius: 6px;
}
table.finor-table {
width: 100%;
border-collapse: collapse;
background-color: #FFFFFF !important;
color: #1E293B !important;
font-size: 0.87rem;
}
table.finor-table thead tr {
background-color: #0F4C81 !important;
color: #FFFFFF !important;
text-align: right;
}
table.finor-table thead th:first-child { text-align: left; }
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
table.finor-table tbody tr:nth-child(even) { background-color: #F8FAFC !important; }
table.finor-table tbody tr.best-row {
background-color: #ECFDF5 !important;
font-weight: 700 !important;
}
.badge-best {
background-color: #1B4D3E;
color: #FFFFFF !important;
padding: 2px 7px;
border-radius: 4px;
font-size: 0.73rem;
font-weight: 700;
}
</style>"""

st.markdown(CSS_GLOBAL, unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS, METADADOS E COTAÇÕES REAIS
# ==============================================================================
DATA_CORTE_TREINO = '2024-12-31'


@st.cache_data
def carregar_mercado_completo(
    mercado: str,
) -> Tuple[
    pd.DataFrame, pd.Series, Dict[str, str], Dict[str, str], pd.Series, pd.Timestamp
]:
    if mercado == 'S&P 100 (EUA)':
        precos: pd.DataFrame = pd.read_csv(
            'sp100_precos.csv', index_col='Date', parse_dates=True
        )
        indice_df: pd.DataFrame = pd.read_csv(
            'sp100_indice.csv', index_col='Date', parse_dates=True
        )
        mapa_set: Dict[str, str] = {}
        mapa_nomes: Dict[str, str] = {}
        if os.path.exists('SP100_wikipedia_21set2026.csv'):
            df_s = pd.read_csv('SP100_wikipedia_21set2026.csv')
            df_s['Sector'] = (
                df_s['Sector'].astype(str).map(lambda x: TRADUCAO_SP100.get(x, x))
            )
            mapa_set = dict(
                zip(df_s['Symbol'].astype(str).str.strip(), df_s['Sector'].astype(str))
            )
            col_nome = (
                'Name'
                if 'Name' in df_s.columns
                else ('Security' if 'Security' in df_s.columns else None)
            )
            if col_nome:
                mapa_nomes = dict(
                    zip(
                        df_s['Symbol'].astype(str).str.strip(),
                        df_s[col_nome].astype(str),
                    )
                )
    else:
        precos = pd.read_csv('ibov_precos.csv', index_col='Date', parse_dates=True)
        indice_df = pd.read_csv('ibov_indice.csv', index_col='Date', parse_dates=True)
        mapa_set = {}
        mapa_nomes = {}
        if os.path.exists('IBOVESPA_B3_24set2026.csv'):
            df_s = pd.read_csv('IBOVESPA_B3_24set2026.csv')
            codigos_sa = df_s['Codigo'].astype(str).str.strip() + '.SA'
            mapa_set = dict(zip(codigos_sa, df_s['Setor'].astype(str)))
            if 'Acao' in df_s.columns:
                mapa_nomes = dict(zip(codigos_sa, df_s['Acao'].astype(str)))

    indice: pd.Series = indice_df.iloc[:, 0]

    pct_nulos = precos.loc[:DATA_CORTE_TREINO].isna().mean() * 100.0
    validas: List[str] = [str(c) for c in pct_nulos[pct_nulos <= 5.0].index.tolist()]

    precos_clean: pd.DataFrame = precos[validas].ffill().dropna(how='any')
    indice_clean: pd.Series = indice.ffill().dropna()

    ret_acoes: pd.DataFrame = np.log(precos_clean / precos_clean.shift(1)).dropna(
        how='any'
    )
    ret_idx: pd.Series = np.log(indice_clean / indice_clean.shift(1)).dropna()

    datas_comuns = ret_acoes.index.intersection(ret_idx.index)
    ret_acoes = ret_acoes.loc[datas_comuns]
    ret_idx = ret_idx.loc[datas_comuns]
    precos_clean = precos_clean.loc[datas_comuns]

    setores_map: Dict[str, str] = {
        str(c): mapa_set.get(str(c), 'Outros') for c in ret_acoes.columns
    }
    nomes_map: Dict[str, str] = {
        str(c): mapa_nomes.get(str(c), str(c).replace('.SA', ''))
        for c in ret_acoes.columns
    }
    ultimas_cotacoes: pd.Series = precos_clean.iloc[-1]
    data_ultima_cot: pd.Timestamp = pd.Timestamp(precos_clean.index[-1])

    return ret_acoes, ret_idx, setores_map, nomes_map, ultimas_cotacoes, data_ultima_cot


# ==============================================================================
# 3. MOTOR DE MODELAGEM CALIBRADO E VALIDAÇÃO WALK-FORWARD
# ==============================================================================
MODELO_SLSQP: str = 'SLSQP + Ajuste de Volatilidade'
MODELO_ENET: str = 'Elastic Net Positivo'
MODELOS: List[str] = [MODELO_SLSQP, MODELO_ENET]
ESCOLHA_AUTO: str = 'Campeão (menor TE)'
K_PADRAO: int = 15
PESO_MAX_PADRAO: float = 0.30
LAMBDA_VOL_PADRAO: float = 0.25
L1_RATIO_PADRAO: float = 0.5
KS_GRID: List[int] = [5, 8, 10, 12, 15, 20, 25, 30, 35]


def construir_features_treino(
    treino_X: pd.DataFrame, treino_y: pd.Series, setores_map: Dict[str, str]
) -> pd.DataFrame:
    var_idx = float(treino_y.var())
    feats = pd.DataFrame(index=treino_X.columns)
    feats['corr_idx'] = treino_X.corrwith(treino_y)
    feats['beta'] = treino_X.apply(lambda s: float(s.cov(treino_y)) / var_idx)
    feats['vol_anual'] = treino_X.std() * np.sqrt(252.0)
    feats['te_ind'] = treino_X.sub(treino_y, axis=0).std() * np.sqrt(252.0)
    feats['setor'] = [setores_map.get(str(idx), 'Outros') for idx in feats.index]
    return feats.sort_values('corr_idx', ascending=False)


def filtrar_colinearidade(
    treino_X: pd.DataFrame,
    feats: pd.DataFrame,
    max_cand: int = 100,
    lim_corr: float = 0.90,
) -> List[str]:
    cands: List[str] = [
        str(c)
        for c in feats.sort_values('corr_idx', ascending=False)
        .head(max_cand)
        .index.tolist()
    ]
    corr_m = treino_X[cands].corr()
    sel: List[str] = []
    for a in cands:
        if all(abs(float(corr_m.loc[a, s])) < lim_corr for s in sel):
            sel.append(a)
    return sel


def resolver_slsqp(
    treino_X: pd.DataFrame,
    treino_y: pd.Series,
    acoes: List[str],
    peso_max: float = 0.30,
    lambda_vol: float = 0.0,
    w_ant: Optional[pd.Series] = None,
    lambda_turn: float = 0.0,
) -> pd.Series:
    R = treino_X[acoes].to_numpy(dtype=float) * 100.0
    y = treino_y.to_numpy(dtype=float) * 100.0
    n = len(acoes)
    cov_anual = np.cov(R, rowvar=False) * 252.0
    vol_idx = float(np.std(y, ddof=1) * np.sqrt(252.0))

    w_prev_vec: Optional[np.ndarray] = None
    if w_ant is not None and not w_ant.empty and lambda_turn > 0:
        w_prev_vec = w_ant.reindex(acoes, fill_value=0.0).to_numpy(dtype=float)

    def func_obj(w: np.ndarray) -> float:
        mse = float(np.mean((y - R @ w) ** 2))
        obj = mse
        if lambda_vol > 0:
            v_c = float(np.sqrt(max(float(w.T @ cov_anual @ w), 1e-12)))
            obj += lambda_vol * (((v_c - vol_idx) ** 2) / 252.0)
        if w_prev_vec is not None and lambda_turn > 0:
            obj += (lambda_turn * 10000.0) * float(np.sum((w - w_prev_vec) ** 2))
        return obj

    w0 = (
        w_prev_vec
        if (w_prev_vec is not None and float(w_prev_vec.sum()) > 0.5)
        else np.ones(n) / n
    )
    w0 = w0 / float(w0.sum())
    ub = min(1.0, max(peso_max, 1.0 / n + 1e-3))
    res = minimize(
        func_obj,
        x0=w0,
        method='SLSQP',
        bounds=[(0.0, ub)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: float(np.sum(w)) - 1.0}],
        options={'ftol': 1e-11, 'maxiter': 750},
    )
    w_s = pd.Series(np.maximum(res.x, 0.0), index=acoes)
    w_s = w_s[w_s > 1e-5]
    return w_s / float(w_s.sum())


def ranquear_candidatas(
    treino_X: pd.DataFrame,
    treino_y: pd.Series,
    setores_map: Dict[str, str],
    peso_max: float,
    lambda_vol: float,
) -> List[str]:
    """Filtra redundância (r >= 0,90) e ordena as ações pelo peso da otimização relaxada.
    Não depende de K, então pode ser calculado uma vez e reutilizado para vários K."""
    feats = construir_features_treino(treino_X, treino_y, setores_map)
    pool_sem_red = filtrar_colinearidade(
        treino_X, feats, max_cand=len(treino_X.columns), lim_corr=0.90
    )
    w_rel = resolver_slsqp(
        treino_X, treino_y, pool_sem_red, peso_max=peso_max, lambda_vol=lambda_vol
    )
    return [str(c) for c in w_rel.sort_values(ascending=False).index.tolist()]


def treinar_modelo_campeao(
    treino_X: pd.DataFrame,
    treino_y: pd.Series,
    setores_map: Dict[str, str],
    k: int = 15,
    peso_max: float = 0.30,
    lambda_vol: float = 0.25,
    w_ant: Optional[pd.Series] = None,
    lambda_turn: float = 0.0,
    ranking: Optional[List[str]] = None,
) -> pd.Series:
    if ranking is None:
        ranking = ranquear_candidatas(
            treino_X, treino_y, setores_map, peso_max, lambda_vol
        )
    return resolver_slsqp(
        treino_X,
        treino_y,
        ranking[:k],
        peso_max=peso_max,
        lambda_vol=lambda_vol,
        w_ant=w_ant,
        lambda_turn=lambda_turn,
    )


def _aplicar_teto_pesos(w: np.ndarray, teto: float) -> np.ndarray:
    """Normaliza os pesos para somar 1 e impõe o teto por ação, redistribuindo
    o excesso proporcionalmente entre as posições que ainda estão abaixo do teto."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    w = w / float(w.sum())
    for _ in range(100):
        excesso = float(np.clip(w - teto, 0.0, None).sum())
        if excesso < 1e-12:
            break
        w = np.minimum(w, teto)
        livres = w < teto - 1e-12
        if not livres.any():
            break
        soma_livres = float(w[livres].sum())
        if soma_livres <= 1e-12:
            w[livres] += excesso / int(livres.sum())
        else:
            w[livres] += excesso * w[livres] / soma_livres
    return w


def calcular_caminho_enet(
    treino_X: pd.DataFrame,
    treino_y: pd.Series,
    l1_ratio: float = L1_RATIO_PADRAO,
    n_alphas: int = 100,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Caminho de regularização do Elastic Net com coeficientes não negativos e o
    alpha escolhido por validação cruzada temporal (minimiza o erro de rastreamento
    fora da amostra). Não depende de K: pode ser calculado uma vez e reutilizado.
    Retorna (alphas em ordem decrescente, coeficientes [n_ativos x n_alphas], alpha_cv)."""
    X = treino_X.to_numpy(dtype=float) * 100.0
    y = treino_y.to_numpy(dtype=float) * 100.0
    n_obs = X.shape[0]
    # Acima de alpha_max todos os coeficientes (>= 0) são nulos.
    alpha_max = float(np.max(X.T @ y)) / (n_obs * l1_ratio)
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        return np.array([1.0]), np.zeros((X.shape[1], 1)), 1.0
    alphas = alpha_max * np.logspace(0.0, -4.0, n_alphas)
    cv = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        positive=True,
        fit_intercept=False,
        cv=TimeSeriesSplit(n_splits=n_splits),
        max_iter=5000,
        tol=1e-6,
    ).fit(X, y)
    alphas_out, coefs, _ = enet_path(
        X,
        y,
        l1_ratio=l1_ratio,
        alphas=alphas,
        positive=True,
        max_iter=5000,
        tol=1e-6,
    )
    return np.asarray(alphas_out), np.asarray(coefs), float(cv.alpha_)


def resolver_enet_positivo(
    treino_X: pd.DataFrame,
    treino_y: pd.Series,
    k: int = 15,
    peso_max: float = 0.30,
    l1_ratio: float = L1_RATIO_PADRAO,
    caminho: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
) -> pd.Series:
    """Elastic Net com restrição de sinal (w >= 0) para replicar o índice.

    1) Ranking: ajusta o Elastic Net positivo em todo o universo com o alpha
       escolhido por validação cruzada temporal (se esse alpha deixar menos de K
       ações com peso > 0, usa o primeiro alpha menor que deixe) e fica com as K
       de maior coeficiente. A parte L2 divide o peso entre ações correlacionadas,
       então não é necessário o filtro de colinearidade usado no SLSQP.
    2) Reajuste: reestima o Elastic Net positivo só com as K ações escolhidas.
    3) Orçamento: normaliza para soma 1 e impõe o teto de peso por ação."""
    acoes = [str(c) for c in treino_X.columns]
    k = int(max(1, min(k, len(acoes))))
    if caminho is None:
        caminho = calcular_caminho_enet(treino_X, treino_y, l1_ratio)
    alphas, coefs, alpha_cv = caminho

    n_ativos = (coefs > 1e-10).sum(axis=0)
    elegiveis = np.where((alphas <= alpha_cv * (1.0 + 1e-9)) & (n_ativos >= k))[0]
    pos = int(elegiveis[0]) if len(elegiveis) > 0 else len(alphas) - 1
    alpha_sel = float(alphas[pos])
    coef_rank = coefs[:, pos]

    ordem = np.argsort(coef_rank)[::-1][:k]
    sel = ordem[coef_rank[ordem] > 1e-10]
    if len(sel) == 0:
        # Caso degenerado (nenhuma ação entrou no caminho): top-K por correlação.
        sel = np.argsort(treino_X.corrwith(treino_y).to_numpy(dtype=float))[::-1][:k]
        coef_sel = np.ones(len(sel))
    else:
        modelo = ElasticNet(
            alpha=alpha_sel,
            l1_ratio=l1_ratio,
            positive=True,
            fit_intercept=False,
            max_iter=10000,
            tol=1e-6,
        )
        modelo.fit(
            treino_X.to_numpy(dtype=float)[:, sel] * 100.0,
            treino_y.to_numpy(dtype=float) * 100.0,
        )
        coef_sel = np.asarray(modelo.coef_, dtype=float)
        if float(coef_sel.sum()) <= 1e-12:
            coef_sel = coef_rank[sel]

    n = len(sel)
    ub = min(1.0, max(peso_max, 1.0 / n + 1e-3))
    w = _aplicar_teto_pesos(coef_sel, ub)
    w_s = pd.Series(w, index=[acoes[i] for i in sel])
    w_s = w_s[w_s > 1e-5]
    return w_s / float(w_s.sum())


@st.cache_data(show_spinner=False)
def rodar_walk_forward_completo(
    ret_acoes: pd.DataFrame,
    ret_idx: pd.Series,
    setores_map: Dict[str, str],
    k: int = 15,
    peso_max: float = 0.30,
    lambda_vol: float = 0.25,
    lambda_turn: float = 1e-4,
    custo_bps: int = 10,
    modelo: str = MODELO_SLSQP,
    l1_ratio: float = L1_RATIO_PADRAO,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    inicio = pd.to_datetime('2025-01-01')
    pontos = pd.date_range(inicio, ret_acoes.index[-1], freq='3MS')
    if pontos[-1] < ret_acoes.index[-1]:
        pontos = pontos.union([ret_acoes.index[-1] + pd.Timedelta(days=1)])

    ret_oos_bruto: List[pd.Series] = []
    ret_oos_liq: List[pd.Series] = []
    turnovers: List[float] = []
    janelas: Dict[str, Dict[str, object]] = {}
    w_ant = pd.Series(dtype=float)

    for t0, t1 in zip(pontos[:-1], pontos[1:]):
        tr_x = ret_acoes.loc[t0 - pd.DateOffset(months=24) : t0 - pd.Timedelta(days=1)]
        tr_y = ret_idx.loc[tr_x.index]

        if modelo == MODELO_ENET:
            w_novo = resolver_enet_positivo(
                tr_x, tr_y, k=k, peso_max=peso_max, l1_ratio=l1_ratio
            )
        else:
            w_novo = treinar_modelo_campeao(
                tr_x,
                tr_y,
                setores_map,
                k=k,
                peso_max=peso_max,
                lambda_vol=lambda_vol,
                w_ant=w_ant,
                lambda_turn=lambda_turn,
            )
        if not w_ant.empty:
            uniao = w_novo.index.union(w_ant.index)
            diff_w = w_novo.reindex(uniao, fill_value=0.0) - w_ant.reindex(
                uniao, fill_value=0.0
            )
            turn = 0.5 * float(diff_w.abs().sum())
            turnovers.append(turn)
        else:
            turn = 1.0
        w_ant = w_novo

        cols_w = [str(c) for c in w_novo.index.tolist()]
        bloco_x = ret_acoes.loc[t0 : t1 - pd.Timedelta(days=1), cols_w]
        if len(bloco_x) > 0:
            rb: pd.Series = bloco_x @ w_novo
            rl: pd.Series = rb.copy()
            rl.iloc[0] = float(rl.iloc[0]) - turn * (custo_bps / 10000.0)
            ret_oos_bruto.append(rb)
            ret_oos_liq.append(rl)

            yb = ret_idx.loc[bloco_x.index]
            r_cart = (float(np.exp(rl.sum())) - 1.0) * 100.0
            r_idx = (float(np.exp(yb.sum())) - 1.0) * 100.0
            te_j = (
                float((rb - yb).std() * np.sqrt(252.0) * 100.0)
                if len(bloco_x) > 1
                else np.nan
            )
            janelas[f'Carteira {len(janelas) + 1}'] = {
                'Início': pd.Timestamp(bloco_x.index[0]).strftime('%d/%m/%Y'),
                'Fim': pd.Timestamp(bloco_x.index[-1]).strftime('%d/%m/%Y'),
                'Pregões': int(len(bloco_x)),
                'Ações (K)': int(len(w_novo)),
                'Retorno Carteira (%)': round(r_cart, 2),
                'Retorno Índice (%)': round(r_idx, 2),
                'Diferença (p.p.)': round(r_cart - r_idx, 2),
                'TE (% a.a.)': round(te_j, 2) if pd.notna(te_j) else np.nan,
                'Giro (%)': round(turn * 100.0, 1),
            }

    sb = pd.concat(ret_oos_bruto)
    sl = pd.concat(ret_oos_liq)
    si = ret_idx.loc[sb.index]

    te = float((sb - si).std() * np.sqrt(252.0) * 100.0)
    vol_c = float(sb.std() * np.sqrt(252.0) * 100.0)
    vol_i = float(si.std() * np.sqrt(252.0) * 100.0)
    ss_res = float(((si - sb) ** 2).sum())
    ss_tot = float(((si - float(si.mean())) ** 2).sum())
    td_b = float(((np.exp(sb.sum()) - 1.0) - (np.exp(si.sum()) - 1.0)) * 100.0)
    td_l = float(((np.exp(sl.sum()) - 1.0) - (np.exp(si.sum()) - 1.0)) * 100.0)

    df_wf = pd.DataFrame(
        [
            {
                'K Efetivo': int((w_ant > 1e-5).sum()),
                'TE Walk-Forward (% a.a.)': round(te, 2),
                'Vol. Carteira (%)': round(vol_c, 2),
                'Gap Vol (p.p.)': round(abs(vol_c - vol_i), 2),
                'Correlação (r)': round(float(sb.corr(si)), 4),
                'R²': round(1.0 - ss_res / ss_tot, 4),
                'Turnover Médio Trim. (%)': round(float(np.mean(turnovers)) * 100.0, 1),
                'TD Bruta (%)': round(td_b, 2),
                'TD Líquida (c/ custos %)': round(td_l, 2),
                'Carteiras OOS': len(turnovers) + 1,
            }
        ],
        index=pd.Index([modelo], name='Modelo'),
    )
    df_janelas = pd.DataFrame.from_dict(janelas, orient='index')
    return df_wf, sl, w_ant, si, df_janelas


@st.cache_data(show_spinner=False)
def calcular_curva_k(
    ret_acoes: pd.DataFrame,
    ret_idx: pd.Series,
    setores_map: Dict[str, str],
    peso_max: float = 0.30,
    lambda_vol: float = 0.25,
    modelo: str = MODELO_SLSQP,
    l1_ratio: float = L1_RATIO_PADRAO,
) -> pd.DataFrame:
    tr_x = ret_acoes.loc[:DATA_CORTE_TREINO]
    tr_y = ret_idx.loc[:DATA_CORTE_TREINO]
    oos_x = ret_acoes.loc[DATA_CORTE_TREINO:].iloc[1:]
    oos_y = ret_idx.loc[oos_x.index]

    # A seleção (ranking do SLSQP ou caminho do Elastic Net) não depende de K:
    # calcula uma vez e reaproveita no grid.
    ranking: Optional[List[str]] = None
    caminho: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    if modelo == MODELO_ENET:
        caminho = calcular_caminho_enet(tr_x, tr_y, l1_ratio)
    else:
        ranking = ranquear_candidatas(tr_x, tr_y, setores_map, peso_max, lambda_vol)
    linhas: List[Dict[str, float]] = []
    for k_i in KS_GRID:
        if modelo == MODELO_ENET:
            w_i = resolver_enet_positivo(
                tr_x,
                tr_y,
                k=k_i,
                peso_max=peso_max,
                l1_ratio=l1_ratio,
                caminho=caminho,
            )
        else:
            w_i = treinar_modelo_campeao(
                tr_x,
                tr_y,
                setores_map,
                k=k_i,
                peso_max=peso_max,
                lambda_vol=lambda_vol,
                ranking=ranking,
            )
        cols_i = [str(c) for c in w_i.index.tolist()]
        te_i = float((oos_x[cols_i] @ w_i - oos_y).std() * np.sqrt(252.0) * 100.0)
        linhas.append({'K': k_i, 'TE': round(te_i, 2)})
    return pd.DataFrame(linhas)


def rodar_todos_modelos(
    ret_acoes: pd.DataFrame,
    ret_idx: pd.Series,
    setores_map: Dict[str, str],
    k: int,
    peso_max: float,
    lambda_vol: float,
    l1_ratio: float,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]]:
    """Walk-forward dos dois algoritmos. Cada um recebe só os parâmetros que o
    afetam: mexer em λ_vol não reprocessa o Elastic Net, e mexer em ρ (mix L1/L2)
    não reprocessa o SLSQP, porque o cache do Streamlit usa os argumentos como chave."""
    return {
        MODELO_SLSQP: rodar_walk_forward_completo(
            ret_acoes,
            ret_idx,
            setores_map,
            k=k,
            peso_max=peso_max,
            lambda_vol=lambda_vol,
            modelo=MODELO_SLSQP,
        ),
        MODELO_ENET: rodar_walk_forward_completo(
            ret_acoes,
            ret_idx,
            setores_map,
            k=k,
            peso_max=peso_max,
            modelo=MODELO_ENET,
            l1_ratio=l1_ratio,
        ),
    }


def calcular_curvas_k_todos(
    ret_acoes: pd.DataFrame,
    ret_idx: pd.Series,
    setores_map: Dict[str, str],
    peso_max: float,
    lambda_vol: float,
    l1_ratio: float,
) -> Dict[str, pd.DataFrame]:
    """Curva TE × K dos dois algoritmos (mesma lógica de cache de rodar_todos_modelos)."""
    return {
        MODELO_SLSQP: calcular_curva_k(
            ret_acoes,
            ret_idx,
            setores_map,
            peso_max=peso_max,
            lambda_vol=lambda_vol,
            modelo=MODELO_SLSQP,
        ),
        MODELO_ENET: calcular_curva_k(
            ret_acoes,
            ret_idx,
            setores_map,
            peso_max=peso_max,
            modelo=MODELO_ENET,
            l1_ratio=l1_ratio,
        ),
    }


# ==============================================================================
# 4. AUXILIARES VISUAIS (SEM INDENTAÇÃO HTML PARA EVITAR BLOCO DE CÓDIGO)
# ==============================================================================
def obter_url_logo(ticker: str, is_ibov: bool) -> str:
    t_limpo = ticker.replace('.SA', '').strip()
    dom = DOMINIOS_IBOV.get(t_limpo) if is_ibov else DOMINIOS_SP100.get(t_limpo)
    if dom:
        return f'https://www.google.com/s2/favicons?domain={dom}&sz=128'
    return ''


def renderizar_cards_carteira(
    pesos: pd.Series,
    nomes_map: Dict[str, str],
    setores_map: Dict[str, str],
    cotacoes: pd.Series,
    is_ibov: bool,
    cor_barra: str,
) -> str:
    moeda = 'R$' if is_ibov else 'US$'
    w_ord = pesos.sort_values(ascending=False)
    max_w = float(w_ord.max()) if len(w_ord) > 0 else 1.0

    cards_html: List[str] = ['<div class="stocks-grid">']
    for t_raw, w_val in w_ord.items():
        t_str = str(t_raw)
        w = float(w_val)
        t_clean = t_str.replace('.SA', '')
        sigla_curta = t_clean[:5]
        nome_emp = nomes_map.get(t_str, t_clean)
        setor_emp = setores_map.get(t_str, 'Outros')
        preco_atual = float(cotacoes.get(t_str, np.nan))
        preco_txt = f'{moeda} {preco_atual:,.2f}' if pd.notna(preco_atual) else ''
        url_logo = obter_url_logo(t_str, is_ibov)
        larg_pct = min(100.0, (w / max_w) * 100.0)

        img_tag = (
            f"<img src='{url_logo}' class='logo-img' onerror=\"this.style.display='none';\" />"
            if url_logo
            else ''
        )

        # Sem indentação de 4 espaços para o Markdown do Streamlit não transformar em bloco <pre><code>
        card_item = (
            f'<div class="stock-card">'
            f'<div class="logo-wrap"><span class="logo-fallback">{sigla_curta}</span>{img_tag}</div>'
            f'<div class="stock-info">'
            f'<div class="stock-top">'
            f'<span class="stock-ticker">{t_clean}</span>'
            f'<span class="stock-weight" style="color: {cor_barra} !important;">{w * 100:.2f}%</span>'
            f'</div>'
            f'<div class="stock-name" title="{nome_emp}">{nome_emp}</div>'
            f'<div class="stock-sector" title="{setor_emp}">{setor_emp}</div>'
            f'<div class="stock-price">{preco_txt}</div>'
            f'<div class="weight-bar-bg">'
            f'<div style="width: {larg_pct:.1f}%; height: 100%; background-color: {cor_barra};"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        cards_html.append(card_item)
    cards_html.append('</div>')
    return ''.join(cards_html)


def renderizar_tabela_html(
    df: pd.DataFrame,
    col_destaque: str = 'TE Walk-Forward (% a.a.)',
    cor_header: str = '#0F4C81',
    nome_indice_col: str = 'Família de Modelo',
) -> str:
    melhor_idx = str(df[col_destaque].idxmin()) if col_destaque in df.columns else ''
    html = ['<div class="finor-table-container"><table class="finor-table">']
    html.append(
        f'<thead style="background-color: {cor_header} !important;"><tr><th>{nome_indice_col}</th>'
    )
    for c in df.columns:
        html.append(
            f'<th style="background-color: {cor_header} !important; color: #FFFFFF !important;">{c}</th>'
        )
    html.append('</tr></thead><tbody>')

    for idx_raw, row in df.iterrows():
        idx_str = str(idx_raw)
        is_best = idx_str == melhor_idx
        tr_class = ' class="best-row"' if is_best else ''
        nome_exib = (
            f'{idx_str} <span class="badge-best">Campeão</span>' if is_best else idx_str
        )
        html.append(f'<tr{tr_class}><td>{nome_exib}</td>')
        for c in df.columns:
            val = row[c]
            if isinstance(val, (int, np.integer)) or c in [
                'K Efetivo',
                'Carteiras OOS',
                'Qtd. Ações (Inteiras)',
            ]:
                txt = f'{int(val):,}'.replace(',', '.')
            elif c in ['Correlação (r)', 'R²']:
                txt = f'{float(val):.4f}'
            elif isinstance(val, (float, np.floating)):
                txt = f'{float(val):,.2f}'
            else:
                txt = str(val)
            html.append(f'<td>{txt}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return ''.join(html)


def configurar_grafico_padrao(
    fig: go.Figure, altura: int = 460, y_title: str = '', x_title: str = 'Data'
) -> go.Figure:
    fig.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        height=altura,
        font=dict(family='sans-serif', size=12, color='#222222'),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#FFFFFF', font_size=12, font_color='#1A202C', bordercolor='#CBD5E1'
        ),
        margin=dict(l=65, r=25, t=55, b=55),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#E2E8F0',
            borderwidth=1,
            font=dict(size=11.5, color='#1A202C'),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12.5, color='#1A202C')),
            tickfont=dict(size=11, color='#333333'),
            showgrid=False,
            showline=True,
            linecolor='#CCCCCC',
            linewidth=1,
            ticks='outside',
            tickcolor='#CCCCCC',
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12.5, color='#1A202C')),
            tickfont=dict(size=11, color='#333333'),
            showgrid=True,
            gridcolor='#E5E5E5',
            gridwidth=1,
            griddash='dash',
            showline=True,
            linecolor='#CCCCCC',
            linewidth=1,
            zeroline=False,
        ),
    )
    return fig


# ==============================================================================
# 5. SIDEBAR: SELETOR DE ÍNDICE E PARÂMETROS DA CARTEIRA
# ==============================================================================
@st.cache_resource(
    show_spinner='Pré-processando S&P 100 e IBOVESPA (SLSQP + Elastic Net, K = 15)...'
)
def preaquecer_cache() -> bool:
    """Roda uma única vez por servidor: deixa walk-forward e curva de K com os
    parâmetros padrão já calculados para os dois mercados. Só reprocessa quando
    o usuário muda algum parâmetro na barra lateral."""
    for mercado in ('S&P 100 (EUA)', 'IBOVESPA (Brasil)'):
        ret_a, ret_i, setores, _, _, _ = carregar_mercado_completo(mercado)
        rodar_todos_modelos(
            ret_a,
            ret_i,
            setores,
            K_PADRAO,
            PESO_MAX_PADRAO,
            LAMBDA_VOL_PADRAO,
            L1_RATIO_PADRAO,
        )
        calcular_curvas_k_todos(
            ret_a,
            ret_i,
            setores,
            PESO_MAX_PADRAO,
            LAMBDA_VOL_PADRAO,
            L1_RATIO_PADRAO,
        )
    return True


preaquecer_cache()

st.sidebar.markdown('### >>>>> ÍNDICE & PARÂMETROS')
mercado_sel = st.sidebar.radio(
    'Selecione o Índice (Benchmark)',
    ['S&P 100 (EUA)', 'IBOVESPA (Brasil)'],
    index=0,
)

escolha_modelo = st.sidebar.radio(
    'Carteira exibida nas abas 1 e 4',
    [ESCOLHA_AUTO, MODELO_SLSQP, MODELO_ENET],
    index=0,
    help='As abas 2 e 3 sempre comparam os dois algoritmos. "Campeão" exibe '
    'o de menor tracking error walk-forward.',
)

is_ibov = 'IBOV' in mercado_sel
cor_primaria = CORES['ibov'] if is_ibov else CORES['sp100']
cor_secundaria = CORES['ibov_sec'] if is_ibov else CORES['sp100_sec']
moeda_simb = 'R$' if is_ibov else 'US$'

k_val = st.sidebar.slider(
    'Cardinalidade da Carteira (K ações)',
    min_value=5,
    max_value=35,
    value=K_PADRAO,
    step=1,
)
peso_max_val = (
    st.sidebar.slider(
        'Peso Máximo por Ação (%)',
        min_value=7,
        max_value=30,
        value=int(PESO_MAX_PADRAO * 100),
        step=1,
    )
    / 100.0
)
lambda_vol_val = st.sidebar.slider(
    'Ajuste de Volatilidade (λ_vol)',
    min_value=0.0,
    max_value=1.0,
    value=LAMBDA_VOL_PADRAO,
    step=0.05,
    help='Parâmetro do SLSQP (não afeta o Elastic Net).',
)
l1_ratio_val = float(
    st.sidebar.slider(
        'Mix L1/L2 do Elastic Net (ρ)',
        min_value=0.1,
        max_value=1.0,
        value=L1_RATIO_PADRAO,
        step=0.1,
        help='Parâmetro do Elastic Net positivo (não afeta o SLSQP). '
        'ρ = 1 equivale ao Lasso positivo (mais esparso); ρ menor aumenta a '
        'parte Ridge, que divide o peso entre ações correlacionadas.',
    )
)

st.sidebar.markdown('---')
st.sidebar.markdown('### >>>>> PROTOCOLO DE VALIDAÇÃO')
st.sidebar.info(
    '**Validação Walk-Forward Estrita**\n\n'
    '• **Janela Móvel de Treino:** 24 meses\n'
    '• **Rebalanceamento:** Trimestral (7 carteiras consecutivas fora da amostra)\n'
    '• **Custos de Transação:** 0,10% (10 bps) descontados sobre o giro (*turnover*)'
)

(
    ret_acoes_full,
    ret_idx_full,
    setores_map,
    nomes_map,
    ultimas_cotacoes,
    data_ultima_cot,
) = carregar_mercado_completo(mercado_sel)

with st.spinner('Reprocessando as carteiras com os novos parâmetros...'):
    resultados = rodar_todos_modelos(
        ret_acoes_full,
        ret_idx_full,
        setores_map,
        k_val,
        peso_max_val,
        lambda_vol_val,
        l1_ratio_val,
    )
    curvas_k = calcular_curvas_k_todos(
        ret_acoes_full,
        ret_idx_full,
        setores_map,
        peso_max_val,
        lambda_vol_val,
        l1_ratio_val,
    )

df_wf_comp = pd.concat([resultados[m][0] for m in MODELOS])
if escolha_modelo == ESCOLHA_AUTO:
    modelo_sel = str(df_wf_comp['TE Walk-Forward (% a.a.)'].idxmin())
else:
    modelo_sel = escolha_modelo

# Carteira exibida nas abas 1 e 4 (as abas 2 e 3 comparam os dois algoritmos)
_, _, pesos_atuais, idx_oos, df_janelas = resultados[modelo_sel]
CORES_MODELOS: Dict[str, str] = {
    MODELO_SLSQP: cor_primaria,
    MODELO_ENET: CORES['laranja'],
}

metricas = df_wf_comp.loc[modelo_sel]
te_melhor = float(metricas['TE Walk-Forward (% a.a.)'])
td_liquida = float(metricas['TD Líquida (c/ custos %)'])
vol_cart_oos = float(metricas['Vol. Carteira (%)'])
vol_idx_oos = float(idx_oos.std() * np.sqrt(252.0) * 100.0)

# ==============================================================================
# 6. CABEÇALHO COM EMBLEMA DO ÍNDICE E KPIS EXECUTIVOS
# ==============================================================================
emblema_html = (
    '<div class="idx-emblem" style="background: linear-gradient(135deg, #1B4D3E 0%, #2E7D67 100%);">'
    '<span style="font-size: 1.15rem; line-height: 1;">[B]³</span>'
    '<span style="font-size: 0.65rem; font-weight: 700; margin-top: 3px; letter-spacing: 0.05em;">IBOV</span></div>'
    if is_ibov
    else '<div class="idx-emblem" style="background: linear-gradient(135deg, #0F4C81 0%, #2B6999 100%);">'
    '<span style="font-size: 1.05rem; line-height: 1;">S&P</span>'
    '<span style="font-size: 0.85rem; font-weight: 800; margin-top: 2px;">100</span></div>'
)

dt_ini_str = pd.Timestamp(idx_oos.index[0]).strftime('%d/%m/%Y')
dt_fim_str = pd.Timestamp(idx_oos.index[-1]).strftime('%d/%m/%Y')

header_html = (
    f'<div class="header-box">{emblema_html}<div>'
    f'<div class="finor-tag">>>>>> FINOR × BOOTCAMP DATA SCIENCE 2026 • APRESENTAÇÃO FINAL</div>'
    f'<div class="finor-title">Index Tracking Executivo & Simulador de Carteira: {mercado_sel}</div>'
    f'<div class="finor-subtitle">Replicação enxuta com <b>K = {k_val} ações</b> (de {ret_acoes_full.shape[1]} elegíveis) '
    f'validada em <b>7 carteiras trimestrais Walk-Forward fora da amostra</b> ({dt_ini_str} a {dt_fim_str}).</div>'
    f'</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

kpi_html = (
    f'<div class="kpi-grid">'
    f'<div class="kpi-card" style="border-left-color: {cor_primaria};">'
    f'<div class="kpi-label">DESVIO ANUALIZADO (TRACKING ERROR)</div>'
    f'<div class="kpi-value">{te_melhor:.2f}% a.a.</div>'
    f'<span class="kpi-delta good">Diferença acumulada líquida: {td_liquida:+.2f} p.p.</span>'
    f'</div>'
    f'<div class="kpi-card" style="border-left-color: {cor_primaria};">'
    f'<div class="kpi-label">MODELO</div>'
    f'<div class="kpi-value">{modelo_sel}</div>'
    f'<span class="kpi-delta good">R² = {float(metricas["R²"]):.4f} | r = {float(metricas["Correlação (r)"]):.4f}</span>'
    f'</div>'
    f'<div class="kpi-card" style="border-left-color: {cor_primaria};">'
    f'<div class="kpi-label">ADERÊNCIA DE VOLATILIDADE</div>'
    f'<div class="kpi-value">{vol_cart_oos:.2f}% a.a.</div>'
    f'<span class="kpi-delta neutral">Índice: {vol_idx_oos:.2f}% (Gap: {float(metricas["Gap Vol (p.p.)"]):.2f} p.p.)</span>'
    f'</div>'
    f'<div class="kpi-card" style="border-left-color: {cor_primaria};">'
    f'<div class="kpi-label">GIRO TRIMESTRAL (TURNOVER)</div>'
    f'<div class="kpi-value">{float(metricas["Turnover Médio Trim. (%)"]):.1f}%</div>'
    f'<span class="kpi-delta neutral">7 Rebalanceamentos OOS (c/ custo 0,10%)</span>'
    f'</div>'
    f'</div>'
)
st.markdown(kpi_html, unsafe_allow_html=True)

# ==============================================================================
# 7. ABAS DO APLICATIVO FINAL
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        '1. Carteira Campeã & Simulador de Investimento',
        '2. Desempenho Walk-Forward (7 Carteiras OOS)',
        '3. Curva Desvio × Nº de Ações (Escolha do K)',
        '4. Equilíbrio Setorial & Risco',
    ]
)

# ------------------------------------------------------------------------------
# ABA 1: VITRINE DA CARTEIRA E SIMULADOR DE INVESTIMENTO
# ------------------------------------------------------------------------------
with tab1:
    col_aporte, _col_vazia = st.columns([1.0, 1.3])
    with col_aporte:
        valor_padrao = 100000.0 if is_ibov else 50000.0
        passo_val = 5000.0 if is_ibov else 2500.0
        valor_investir = float(
            st.number_input(
                f'Valor do aporte para simulação de compra ({moeda_simb}):',
                min_value=1000.0,
                max_value=100000000.0,
                value=valor_padrao,
                step=passo_val,
                format='%.2f',
            )
        )

    w_atual = pesos_atuais.sort_values(ascending=False)

    st.markdown(
        f'<div class="sec-title">>>>>> Composição da Carteira Recomendada: {modelo_sel} ({len(w_atual)} Ações)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sec-sub">Pesos alocados no último rebalanceamento e cotação de fechamento em <b>{data_ultima_cot.strftime("%d/%m/%Y")}</b>. A barra de cada card mostra o peso da ação em relação à maior posição da carteira.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        renderizar_cards_carteira(
            w_atual, nomes_map, setores_map, ultimas_cotacoes, is_ibov, cor_primaria
        ),
        unsafe_allow_html=True,
    )

    cotacoes_cart = ultimas_cotacoes.loc[w_atual.index]
    orcamento_alvo = w_atual * valor_investir
    qtd_inteira = np.floor(orcamento_alvo / cotacoes_cart).astype(int)
    valor_efetivo = qtd_inteira * cotacoes_cart
    peso_efetivo = (valor_efetivo / valor_investir) * 100.0
    caixa_residual = valor_investir - float(valor_efetivo.sum())

    st.markdown(
        f'<div class="sec-title">>>>>> Calculadora de Execução: Quantidade de Ações para {moeda_simb} {valor_investir:,.2f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sec-sub">Lotes inteiros calculados pela última cotação — '
        f'<b>Total Alocado: {moeda_simb} {float(valor_efetivo.sum()):,.2f} ({float(peso_efetivo.sum()):.2f}%)</b> | '
        f'<b>Saldo Residual em Caixa: {moeda_simb} {caixa_residual:,.2f}</b></div>',
        unsafe_allow_html=True,
    )

    df_boleta = pd.DataFrame(
        {
            'Empresa': [
                nomes_map.get(str(t), str(t).replace('.SA', '')) for t in w_atual.index
            ],
            'Setor': [setores_map.get(str(t), 'Outros') for t in w_atual.index],
            'Peso Modelo (%)': (w_atual.to_numpy(dtype=float) * 100.0).round(2),
            f'Cotação ({moeda_simb})': cotacoes_cart.to_numpy(dtype=float).round(2),
            f'Orçamento Alvo ({moeda_simb})': orcamento_alvo.to_numpy(
                dtype=float
            ).round(2),
            'Qtd. Ações (Inteiras)': qtd_inteira.to_numpy(dtype=int),
            f'Valor Comprado ({moeda_simb})': valor_efetivo.to_numpy(dtype=float).round(
                2
            ),
            'Peso Realizado (%)': peso_efetivo.to_numpy(dtype=float).round(2),
        },
        index=[str(t).replace('.SA', '') for t in w_atual.index],
    )

    st.markdown(
        renderizar_tabela_html(
            df_boleta,
            col_destaque='',
            cor_header=cor_primaria,
            nome_indice_col='Ticker',
        ),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# ABA 2: DESEMPENHO WALK-FORWARD
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(
        '<div class="sec-title">>>>>> Validação Walk-Forward no Futuro (7 Carteiras Trimestrais Consecutivas)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Trajetória acumulada fora da amostra (Base 100 em Jan/2025) já descontando custos de transação (0,10%) a cada rebalanceamento trimestral</div>',
        unsafe_allow_html=True,
    )

    idx_acum = np.exp(idx_oos.cumsum()) * 100.0
    fig_wf = go.Figure()
    fig_wf.add_hline(y=100, line_dash='dot', line_color='#A0AEC0', line_width=1.2)
    fig_wf.add_trace(
        go.Scatter(
            x=idx_acum.index,
            y=idx_acum,
            mode='lines',
            name=f'Benchmark ({mercado_sel})',
            line=dict(color=CORES['grafite'], width=3.3),
        )
    )
    for m_nome in MODELOS:
        cart_acum_m = np.exp(resultados[m_nome][1].cumsum()) * 100.0
        te_m = float(df_wf_comp.loc[m_nome, 'TE Walk-Forward (% a.a.)'])
        fig_wf.add_trace(
            go.Scatter(
                x=cart_acum_m.index,
                y=cart_acum_m,
                mode='lines',
                name=f'{m_nome} (TE: {te_m:.2f}%)',
                line=dict(color=CORES_MODELOS[m_nome], width=3.0),
            )
        )

    for dt_reb in pd.date_range('2025-04-01', idx_oos.index[-1], freq='3MS'):
        fig_wf.add_vline(x=dt_reb, line_dash='dot', line_color='#E2E8F0', line_width=1)

    fig_wf = configurar_grafico_padrao(
        fig_wf,
        altura=470,
        y_title=f'Evolução do Patrimônio (Base 100 = {moeda_simb} 100)',
        x_title='Data (Linhas pontilhadas verticais indicam rebalanceamentos trimestrais)',
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown(
        '<div class="sec-title">>>>>> Resumo Consolidado (Validação Walk-Forward)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Métricas do período fora da amostra completo para os dois algoritmos. '
        'O <b>Elastic Net positivo</b> exige pesos ≥ 0 e escolhe o alpha por validação cruzada temporal; '
        'ranqueia as ações pelos coeficientes, reajusta as K escolhidas e normaliza para soma 100% com o mesmo '
        'teto por ação. Diferente do SLSQP, não tem penalização de giro nem ajuste de volatilidade.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        renderizar_tabela_html(
            df_wf_comp,
            col_destaque='TE Walk-Forward (% a.a.)',
            cor_header=cor_primaria,
            nome_indice_col='Modelo',
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sec-title">>>>>> Detalhe por Carteira Trimestral</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sec-sub">Retorno líquido de custos de cada janela de rebalanceamento contra o índice — carteira <b>{modelo_sel}</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        renderizar_tabela_html(
            df_janelas,
            col_destaque='',
            cor_header=cor_primaria,
            nome_indice_col='Carteira',
        ),
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# ABA 3: CURVA DE TRACKING ERROR X CARDINALIDADE (K)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(
        '<div class="sec-title">>>>>> A partir de K = 15 ações, adicionar mais ativos traz retorno decrescente</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">Curva de desvio anualizado fora da amostra variando o tamanho da carteira de K = 5 a K = 35 ações</div>',
        unsafe_allow_html=True,
    )

    fig_k = go.Figure()
    for m_nome in MODELOS:
        fig_k.add_trace(
            go.Scatter(
                x=curvas_k[m_nome]['K'],
                y=curvas_k[m_nome]['TE'],
                mode='lines+markers',
                name=m_nome,
                line=dict(color=CORES_MODELOS[m_nome], width=3.0),
                marker=dict(size=8),
            )
        )
    fig_k.add_vline(
        x=k_val, line_dash='dash', line_color=CORES['alerta'], line_width=1.5
    )
    fig_k.add_annotation(
        x=k_val,
        y=max(float(c['TE'].max()) for c in curvas_k.values()) * 0.94,
        text=f'K selecionado = {k_val} ações',
        showarrow=False,
        xanchor='left',
        xshift=6,
        font=dict(color=CORES['alerta'], size=11.5),
    )
    fig_k = configurar_grafico_padrao(
        fig_k,
        altura=450,
        x_title='Número de Ações na Carteira (K)',
        y_title='Tracking Error Fora da Amostra (% a.a.)',
    )
    st.plotly_chart(fig_k, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 4: EQUILÍBRIO SETORIAL E CONTROLE DE VOLATILIDADE
# ------------------------------------------------------------------------------
with tab4:
    col_p1, col_p2 = st.columns([1.05, 1.0])
    with col_p1:
        st.markdown(
            '<div class="sec-title">>>>>> Aderência de Volatilidade: Carteira vs. Índice</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sec-sub">Volatilidade anualizada (% a.a.) no período fora da amostra — gap de <b>{float(metricas["Gap Vol (p.p.)"]):.2f} p.p.</b></div>',
            unsafe_allow_html=True,
        )

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(
                y=['Índice (benchmark)', 'Carteira'],
                x=[vol_idx_oos, vol_cart_oos],
                orientation='h',
                marker_color=[CORES['grafite'], cor_primaria],
                text=[f'{vol_idx_oos:.2f}%', f'{vol_cart_oos:.2f}%'],
                textposition='outside',
                textfont=dict(color='#1A202C', size=11),
            )
        )
        fig_vol = configurar_grafico_padrao(
            fig_vol,
            altura=420,
            x_title='Volatilidade Anualizada (% a.a.)',
            y_title='',
        )
        fig_vol.update_xaxes(range=[0, max(vol_idx_oos, vol_cart_oos) * 1.28])
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_p2:
        st.markdown(
            '<div class="sec-title">>>>>> Equilíbrio Setorial: Carteira vs. Índice</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sec-sub">Distribuição setorial da carteira comparada ao universo completo do {mercado_sel}</div>',
            unsafe_allow_html=True,
        )

        s_set_cart = (
            pd.Series(
                pesos_atuais.to_numpy(dtype=float) * 100.0,
                index=[setores_map.get(str(t), 'Outros') for t in pesos_atuais.index],
            )
            .groupby(level=0)
            .sum()
        )
        s_set_univ = (
            pd.Series(
                1.0,
                index=[
                    setores_map.get(str(t), 'Outros') for t in ret_acoes_full.columns
                ],
            )
            .groupby(level=0)
            .sum()
        )
        s_set_univ = (s_set_univ / float(s_set_univ.sum())) * 100.0

        df_set_comp = (
            pd.DataFrame(
                {
                    'Carteira Campeã (%)': s_set_cart,
                    'Universo do Índice (%)': s_set_univ,
                }
            )
            .fillna(0.0)
            .sort_values('Carteira Campeã (%)', ascending=True)
        )

        fig_set = go.Figure()
        fig_set.add_trace(
            go.Bar(
                y=df_set_comp.index,
                x=df_set_comp['Universo do Índice (%)'],
                name='Proporção no Índice (%)',
                orientation='h',
                marker_color=CORES['neutro'],
            )
        )
        fig_set.add_trace(
            go.Bar(
                y=df_set_comp.index,
                x=df_set_comp['Carteira Campeã (%)'],
                name='Peso na Carteira (%)',
                orientation='h',
                marker_color=cor_primaria,
            )
        )
        fig_set.update_layout(barmode='group')
        fig_set = configurar_grafico_padrao(
            fig_set,
            altura=420,
            x_title='Participação Setorial (%)',
            y_title='Setor Econômico',
        )
        st.plotly_chart(fig_set, use_container_width=True)
