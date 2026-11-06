import os
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, enet_path
from sklearn.model_selection import TimeSeriesSplit
import gurobipy as gp
from gurobipy import GRB

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (NYT EDITORIAL + PALETA FINOR)
# ==============================================================================
st.set_page_config(
    page_title='Index Tracking: S&P 100 & IBOVESPA',
    layout='wide',
    initial_sidebar_state='expanded',
)

CORES: Dict[str, str] = {
    'sp100': '#0F4C81',    # Azul Marinho Principal (S&P 100)
    'sp100_sec': '#2B6999',  # Azul Secundário
    'ibov': '#1B4D3E',     # Verde Esmeralda Fechado (IBOVESPA)
    'ibov_sec': '#2E7D67',   # Verde Secundário
    'alerta': '#C53030',   # Vermelho Destaque / Alerta
    'laranja': '#D97706',  # Âmbar Sóbrio
    'roxo': '#4C51BF',     # Indigo Sóbrio
    'neutro': '#88837D',   # Cinza Histórico / Baseline
    'grafite': '#1A202C',  # Preto/Grafite (Benchmark)
}

DOMINIOS_SP100: Dict[str, str] = {
    'AAPL': 'apple.com', 'MSFT': 'microsoft.com', 'NVDA': 'nvidia.com', 'AMZN': 'amazon.com',
    'GOOG': 'abc.xyz', 'GOOGL': 'abc.xyz', 'META': 'meta.com', 'BRK-B': 'berkshirehathaway.com',
    'LLY': 'lilly.com', 'AVGO': 'broadcom.com', 'JPM': 'jpmorganchase.com', 'TSLA': 'tesla.com',
    'WMT': 'walmart.com', 'V': 'visa.com', 'XOM': 'exxonmobil.com', 'UNH': 'unitedhealthgroup.com',
    'MA': 'mastercard.com', 'PG': 'pg.com', 'COST': 'costco.com', 'JNJ': 'jnj.com',
    'ORCL': 'oracle.com', 'HD': 'homedepot.com', 'BAC': 'bankofamerica.com', 'ABBV': 'abbvie.com',
    'KO': 'coca-colacompany.com', 'NFLX': 'netflix.com', 'MRK': 'merck.com', 'CVX': 'chevron.com',
    'CRM': 'salesforce.com', 'AMD': 'amd.com', 'PEP': 'pepsico.com', 'TMO': 'thermofisher.com',
    'LIN': 'linde.com', 'TMUS': 't-mobile.com', 'WFC': 'wellsfargo.com', 'ADBE': 'adobe.com',
    'CSCO': 'cisco.com', 'MCD': 'mcdonalds.com', 'ABT': 'abbott.com', 'PM': 'pmi.com',
    'NOW': 'servicenow.com', 'IBM': 'ibm.com', 'AXP': 'americanexpress.com', 'MS': 'morganstanley.com',
    'GE': 'ge.com', 'CAT': 'caterpillar.com', 'ISRG': 'intuitive.com', 'TXN': 'ti.com',
    'GS': 'goldmansachs.com', 'INTU': 'intuit.com', 'VZ': 'verizon.com', 'DIS': 'thewaltdisneycompany.com',
    'AMGN': 'amgen.com', 'QCOM': 'qualcomm.com', 'BKNG': 'bookingholdings.com', 'PFE': 'pfizer.com',
    'T': 'att.com', 'BLK': 'blackrock.com', 'RTX': 'rtx.com', 'NEE': 'nexteraenergy.com',
    'CMCSA': 'corporate.comcast.com', 'DHR': 'danaher.com', 'UBER': 'uber.com', 'SPGI': 'spglobal.com',
    'LOW': 'lowes.com', 'UNP': 'up.com', 'LMT': 'lockheedmartin.com', 'HON': 'honeywell.com',
    'COP': 'conocophillips.com', 'PLTR': 'palantir.com', 'SYK': 'stryker.com', 'SCHW': 'schwab.com',
    'AMAT': 'appliedmaterials.com', 'LRCX': 'lamresearch.com', 'MU': 'micron.com', 'ANET': 'arista.com',
    'PANW': 'paloaltonetworks.com', 'DELL': 'dell.com', 'C': 'citigroup.com', 'BA': 'boeing.com',
    'GILD': 'gilead.com', 'MDT': 'medtronic.com', 'DE': 'deere.com', 'UPS': 'ups.com',
    'SBUX': 'starbucks.com', 'MDLZ': 'mondelezinternational.com', 'BMY': 'bms.com', 'SO': 'southerncompany.com',
    'DUK': 'duke-energy.com', 'MO': 'altria.com', 'CVS': 'cvshealth.com', 'GD': 'gd.com',
    'FDX': 'fedex.com', 'EMR': 'emerson.com', 'MMM': '3m.com', 'TGT': 'target.com',
    'USB': 'usbank.com', 'BNY': 'bny.com', 'COF': 'capitalone.com', 'GM': 'gm.com',
    'INTC': 'intel.com', 'ACN': 'accenture.com', 'AMT': 'americantower.com',
}

DOMINIOS_IBOV: Dict[str, str] = {
    'VALE3': 'vale.com', 'PETR4': 'petrobras.com.br', 'PETR3': 'petrobras.com.br', 'ITUB4': 'itau.com.br',
    'ITSA4': 'itausa.com.br', 'BBDC4': 'bradesco.com.br', 'BBDC3': 'bradesco.com.br', 'BBAS3': 'bb.com.br',
    'WEGE3': 'weg.net', 'ABEV3': 'ambev.com.br', 'B3SA3': 'b3.com.br', 'BPAC11': 'btgpactual.com',
    'RENT3': 'localiza.com', 'SUZB3': 'suzano.com.br', 'RDOR3': 'rededorsaoluiz.com.br', 'EQTL3': 'equatorialenergia.com.br',
    'PRIO3': 'prio3.com.br', 'RADL3': 'rdsaude.com.br', 'RAIL3': 'rumolog.com', 'SBSP3': 'sabesp.com.br',
    'VIVT3': 'telefonica.com.br', 'ENEV3': 'eneva.com.br', 'BBSE3': 'bbseguros.com.br', 'GGBR4': 'gerdau.com',
    'GOAU4': 'gerdau.com', 'CMIG4': 'cemig.com.br', 'CPLE3': 'copel.com', 'UGPA3': 'ultra.com.br',
    'TIMS3': 'tim.com.br', 'LREN3': 'lojasrenner.com.br', 'TOTS3': 'totvs.com', 'KLBN11': 'klabin.com.br',
    'ENGI11': 'energisa.com.br', 'EGIE3': 'engie.com.br', 'CSAN3': 'cosan.com.br', 'VBBR3': 'vibraenergia.com.br',
    'ASAI3': 'assai.com.br', 'HYPE3': 'hypera.com.br', 'CPFE3': 'cpfl.com.br', 'SANB11': 'santander.com.br',
    'MULT3': 'multiplan.com.br', 'TAEE11': 'taesa.com.br', 'CMIN3': 'csnmineracao.com.br', 'CSNA3': 'csn.com.br',
    'CYRE3': 'cyrela.com.br', 'MGLU3': 'magazineluiza.com.br', 'HAPV3': 'hapvida.com.br', 'FLRY3': 'fleury.com.br',
    'BRAP4': 'bradespar.com.br', 'ALOS3': 'allos.co', 'IGTI11': 'iguatemi.com.br', 'AURE3': 'aurenenergia.com.br',
    'SMFT3': 'smartfit.com.br', 'YDUQ3': 'yduqs.com.br', 'COGN3': 'cogna.com.br', 'USIM5': 'usiminas.com',
    'BEEF3': 'minervafoods.com', 'MRVE3': 'mrv.com.br', 'DIRR3': 'direcional.com.br', 'CURY3': 'cury.net',
    'POMO4': 'marcopolo.com.br', 'VAMO3': 'vamos.com.br', 'CEAB3': 'cea.com.br', 'CXSE3': 'caixaseguridade.com.br',
    'PSSA3': 'portoseguro.com.br', 'CSMG3': 'copasa.com.br',
}

TRADUCAO_SP100: Dict[str, str] = {
    'Information Technology': 'Tecnologia da Informação', 'Health Care': 'Saúde',
    'Financials': 'Financeiro', 'Industrials': 'Industrial',
    'Communication Services': 'Serviços de Comunicação', 'Consumer Discretionary': 'Consumo Discricionário',
    'Consumer Staples': 'Consumo Básico', 'Energy': 'Energia',
    'Utilities': 'Utilidades Públicas', 'Real Estate': 'Imobiliário', 'Materials': 'Materiais',
}

CSS_GLOBAL = """<style>
:root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { color-scheme: light !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #FFFFFF !important; color: #1A202C !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
[data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0 !important; }
[data-testid="stSidebar"] * { color: #1E293B !important; }
label, [data-testid="stWidgetLabel"] *, .stMarkdown p { color: #1E293B !important; -webkit-text-fill-color: #1E293B !important; }
[role="tablist"] { gap: 6px !important; border-bottom: 2px solid #CBD5E1 !important; }
[role="tab"] { background-color: #E2E8F0 !important; border: 1px solid #CBD5E1 !important; border-bottom: none !important; border-radius: 6px 6px 0 0 !important; padding: 10px 18px !important; opacity: 1 !important; }
[role="tab"]:hover { background-color: #CBD5E1 !important; }
[role="tab"], [role="tab"] *, [role="tab"] p { color: #0F172A !important; -webkit-text-fill-color: #0F172A !important; font-weight: 700 !important; font-size: 0.94rem !important; opacity: 1 !important; }
[role="tab"][aria-selected="true"] { background-color: #0F4C81 !important; border-color: #0F4C81 !important; }
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *, [role="tab"][aria-selected="true"] p { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 800 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"], input[type="number"] { background-color: #F8FAFC !important; color: #0F172A !important; -webkit-text-fill-color: #0F172A !important; border: 1px solid #CBD5E1 !important; font-weight: 600 !important; }
span[data-baseweb="tag"] { background-color: #0F4C81 !important; color: #FFFFFF !important; }
.header-box { display: flex; align-items: center; gap: 18px; padding-bottom: 0.85rem; margin-bottom: 1.1rem; border-bottom: 2px solid #EDF2F7; }
.idx-emblem { width: 64px; height: 64px; border-radius: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #FFFFFF !important; font-weight: 900; flex-shrink: 0; box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
.finor-tag { font-size: 0.80rem; font-weight: 700; color: #2B6999; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.1rem; }
.finor-title { font-size: 1.80rem; font-weight: 800; color: #1A202C; line-height: 1.18; margin-bottom: 0.2rem; }
.finor-subtitle { font-size: 0.95rem; color: #555555; }
.sec-title { font-size: 1.20rem; font-weight: 700; color: #1A202C; margin-top: 0.5rem; margin-bottom: 0.1rem; }
.sec-sub { font-size: 0.88rem; color: #666666; margin-bottom: 0.75rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 1.2rem; }
.kpi-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #0F4C81; border-radius: 6px; padding: 12px 16px; }
.kpi-label { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #475569 !important; margin-bottom: 4px; }
.kpi-value { font-size: 1.48rem; font-weight: 800; color: #0F172A !important; margin-bottom: 5px; }
.kpi-delta { display: inline-block; font-size: 0.76rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; }
.kpi-delta.good { background-color: #DCFCE7; color: #166534 !important; }
.kpi-delta.neutral { background-color: #E2E8F0; color: #1E293B !important; }
.stocks-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(235px, 1fr)); gap: 12px; margin-top: 0.6rem; margin-bottom: 1.2rem; }
.stock-card { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 14px; display: flex; align-items: center; gap: 12px; transition: box-shadow 0.15s ease; }
.stock-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-color: #94A3B8; }
.logo-wrap { position: relative; width: 44px; height: 44px; border-radius: 8px; background-color: #E2E8F0; border: 1px solid #CBD5E1; display: flex; align-items: center; justify-content: center; overflow: hidden; flex-shrink: 0; }
.logo-fallback { font-size: 0.76rem; font-weight: 800; color: #1E293B !important; letter-spacing: -0.02em; z-index: 1; user-select: none; }
.logo-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; background-color: #FFFFFF; padding: 4px; z-index: 2; }
.stock-info { flex: 1; min-width: 0; }
.stock-top { display: flex; justify-content: space-between; align-items: baseline; }
.stock-ticker { font-size: 0.95rem; font-weight: 800; color: #0F172A !important; }
.stock-weight { font-size: 0.92rem; font-weight: 800; color: #0F4C81 !important; }
.stock-name { font-size: 0.78rem; color: #334155 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }
.stock-sector { font-size: 0.72rem; color: #64748B !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stock-price { font-size: 0.78rem; font-weight: 700; color: #334155 !important; margin-top: 1px; }
.weight-bar-bg { width: 100%; height: 4px; background-color: #E2E8F0; border-radius: 2px; margin-top: 6px; overflow: hidden; }
.finor-table-container { overflow-x: auto; margin-top: 0.5rem; margin-bottom: 1.2rem; border: 1px solid #CBD5E1; border-radius: 6px; }
table.finor-table { width: 100%; border-collapse: collapse; background-color: #FFFFFF !important; color: #1E293B !important; font-size: 0.87rem; }
table.finor-table thead tr { background-color: #0F4C81 !important; color: #FFFFFF !important; text-align: right; }
table.finor-table thead th:first-child { text-align: left; }
table.finor-table th { padding: 10px 12px; font-weight: 700; color: #FFFFFF !important; white-space: nowrap; }
table.finor-table td { padding: 9px 12px; border-bottom: 1px solid #EDF2F7; text-align: right; color: #1E293B !important; white-space: nowrap; }
table.finor-table td:first-child { text-align: left; font-weight: 600; color: #0F172A !important; }
table.finor-table tbody tr:nth-child(even) { background-color: #F8FAFC !important; }
table.finor-table tbody tr.best-row { background-color: #ECFDF5 !important; font-weight: 700 !important; }
.badge-best { background-color: #1B4D3E; color: #FFFFFF !important; padding: 2px 7px; border-radius: 4px; font-size: 0.73rem; font-weight: 700; }
</style>"""

st.markdown(CSS_GLOBAL, unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO LEVE DE DADOS E NORMAS
# ==============================================================================
DATA_CORTE_TREINO = '2024-12-31'

def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace('.SA', '')

def read_csv_flexible(filename_base: str) -> pd.DataFrame:
    if os.path.exists(filename_base):
        return pd.read_csv(filename_base)
    elif os.path.exists(f"{filename_base}.csv"):
        return pd.read_csv(f"{filename_base}.csv")
    else:
        raise FileNotFoundError(f"Arquivo não localizado: '{filename_base}' ou '{filename_base}.csv'")

@st.cache_data
def carregar_mercado_completo(mercado: str):
    if mercado == 'S&P 100 (EUA)':
        precos = read_csv_flexible('sp100_precos')
        indice_df = read_csv_flexible('sp100_indice')
        meta = read_csv_flexible('SP100_Acoes') if os.path.exists('SP100_Acoes.csv') or os.path.exists('SP100_Acoes') else read_csv_flexible('SP100_wikipedia_21set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Symbol'].apply(normalize_ticker) if 'Symbol' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        if 'Sector' in meta.columns:
            meta['Sector'] = meta['Sector'].astype(str).map(lambda x: TRADUCAO_SP100.get(x, x))
        
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Sector'] if 'Sector' in meta.columns else meta['Setor']))
        col_nome = 'Name' if 'Name' in meta.columns else ('Security' if 'Security' in meta.columns else 'Ticker')
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        
        # Trata os pesos garantindo uma pd.Series iterável
        if 'Weight' in meta.columns:
            pesos_serie = meta['Weight']
        elif 'Peso_Indice' in meta.columns:
            pesos_serie = meta['Peso_Indice']
        else:
            pesos_serie = pd.Series(1.0 / len(meta), index=meta.index)
        
        mapa_pesos = dict(zip(meta['Ticker_Norm'], pesos_serie))

    else:
        precos = read_csv_flexible('ibov_precos')
        indice_df = read_csv_flexible('ibov_indice')
        meta = read_csv_flexible('Ibov_Acoes') if os.path.exists('Ibov_Acoes.csv') or os.path.exists('Ibov_Acoes') else read_csv_flexible('IBOVESPA_B3_24set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Codigo'].apply(normalize_ticker) if 'Codigo' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Setor'].astype(str)))
        col_nome = 'Acao' if 'Acao' in meta.columns else 'Ticker'
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        
        # Trata os pesos garantindo uma pd.Series iterável
        if 'Part. (%)' in meta.columns:
            pesos_serie = meta['Part. (%)']
        elif 'Peso_Indice' in meta.columns:
            pesos_serie = meta['Peso_Indice']
        else:
            pesos_serie = pd.Series(1.0 / len(meta), index=meta.index)
            
        mapa_pesos = dict(zip(meta['Ticker_Norm'], pesos_serie))

    indice = indice_df.iloc[:, 0].dropna()
    
    col_mapping = {col: normalize_ticker(col) for col in precos.columns}
    precos = precos.rename(columns=col_mapping)
    precos = precos.loc[:, ~precos.columns.duplicated()]

    pct_nulos = precos.loc[:DATA_CORTE_TREINO].isna().mean() * 100.0
    validas = [str(c) for c in pct_nulos[pct_nulos <= 5.0].index.tolist()]

    precos_clean = precos[validas].ffill().dropna(how='any')
    indice_clean = indice.ffill().dropna()

    if precos_clean.abs().max().max() > 2.0:
        ret_acoes = np.log(precos_clean / precos_clean.shift(1)).dropna(how='any')
    else:
        ret_acoes = precos_clean.copy()

    if indice_clean.abs().max() > 2.0:
        ret_idx = np.log(indice_clean / indice_clean.shift(1)).dropna()
    else:
        ret_idx = indice_clean.copy()

    datas_comuns = ret_acoes.index.intersection(ret_idx.index)
    ret_acoes = ret_acoes.loc[datas_comuns]
    ret_idx = ret_idx.loc[datas_comuns]
    precos_clean = precos_clean.loc[datas_comuns]

    setores_map = {str(c): mapa_set.get(str(c), 'Outros') for c in ret_acoes.columns}
    nomes_map = {str(c): mapa_nomes.get(str(c), str(c)) for c in ret_acoes.columns}
    pesos_map = pd.Series({str(c): mapa_pesos.get(str(c), 1.0 / len(ret_acoes.columns)) for c in ret_acoes.columns})
    
    ultimas_cotacoes = precos_clean.iloc[-1]
    data_ultima_cot = pd.Timestamp(precos_clean.index[-1])

    return ret_acoes, ret_idx, setores_map, nomes_map, pesos_map, ultimas_cotacoes, data_ultima_cot
    if mercado == 'S&P 100 (EUA)':
        precos = read_csv_flexible('sp100_precos')
        indice_df = read_csv_flexible('sp100_indice')
        meta = read_csv_flexible('SP100_Acoes') if os.path.exists('SP100_Acoes.csv') or os.path.exists('SP100_Acoes') else read_csv_flexible('SP100_wikipedia_21set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Symbol'].apply(normalize_ticker) if 'Symbol' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        if 'Sector' in meta.columns:
            meta['Sector'] = meta['Sector'].astype(str).map(lambda x: TRADUCAO_SP100.get(x, x))
        
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Sector'] if 'Sector' in meta.columns else meta['Setor']))
        col_nome = 'Name' if 'Name' in meta.columns else ('Security' if 'Security' in meta.columns else 'Ticker')
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        
        # --- CORREÇÃO DE PESOS (S&P 100) ---
        if 'Weight' in meta.columns:
            pesos_serie = meta['Weight']
        elif 'Peso_Indice' in meta.columns:
            pesos_serie = meta['Peso_Indice']
        else:
            pesos_serie = pd.Series(1.0 / len(meta), index=meta.index)
        
        mapa_pesos = dict(zip(meta['Ticker_Norm'], pesos_serie))

    else:
        precos = read_csv_flexible('ibov_precos')
        indice_df = read_csv_flexible('ibov_indice')
        meta = read_csv_flexible('Ibov_Acoes') if os.path.exists('Ibov_Acoes.csv') or os.path.exists('Ibov_Acoes') else read_csv_flexible('IBOVESPA_B3_24set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Codigo'].apply(normalize_ticker) if 'Codigo' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Setor'].astype(str)))
        col_nome = 'Acao' if 'Acao' in meta.columns else 'Ticker'
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        
        # --- CORREÇÃO DE PESOS (IBOVESPA) ---
        if 'Part. (%)' in meta.columns:
            pesos_serie = meta['Part. (%)']
        elif 'Peso_Indice' in meta.columns:
            pesos_serie = meta['Peso_Indice']
        else:
            pesos_serie = pd.Series(1.0 / len(meta), index=meta.index)
            
        mapa_pesos = dict(zip(meta['Ticker_Norm'], pesos_serie))
    if mercado == 'S&P 100 (EUA)':
        precos = read_csv_flexible('sp100_precos')
        indice_df = read_csv_flexible('sp100_indice')
        meta = read_csv_flexible('SP100_Acoes') if os.path.exists('SP100_Acoes.csv') or os.path.exists('SP100_Acoes') else read_csv_flexible('SP100_wikipedia_21set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Symbol'].apply(normalize_ticker) if 'Symbol' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        if 'Sector' in meta.columns:
            meta['Sector'] = meta['Sector'].astype(str).map(lambda x: TRADUCAO_SP100.get(x, x))
        
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Sector'] if 'Sector' in meta.columns else meta['Setor']))
        col_nome = 'Name' if 'Name' in meta.columns else ('Security' if 'Security' in meta.columns else 'Ticker')
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        mapa_pesos = dict(zip(meta['Ticker_Norm'], meta['Weight'] if 'Weight' in meta.columns else meta.get('Peso_Indice', 1.0 / len(meta))))

    else:
        precos = read_csv_flexible('ibov_precos')
        indice_df = read_csv_flexible('ibov_indice')
        meta = read_csv_flexible('Ibov_Acoes') if os.path.exists('Ibov_Acoes.csv') or os.path.exists('Ibov_Acoes') else read_csv_flexible('IBOVESPA_B3_24set2026')
        
        date_col_p = [c for c in precos.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        precos[date_col_p] = pd.to_datetime(precos[date_col_p])
        precos = precos.set_index(date_col_p).sort_index()

        date_col_i = [c for c in indice_df.columns if str(c).lower() in ['date', 'data', 'datetime']][0]
        indice_df[date_col_i] = pd.to_datetime(indice_df[date_col_i])
        indice_df = indice_df.set_index(date_col_i).sort_index()

        meta['Ticker_Norm'] = meta['Codigo'].apply(normalize_ticker) if 'Codigo' in meta.columns else meta['Ticker'].apply(normalize_ticker)
        mapa_set = dict(zip(meta['Ticker_Norm'], meta['Setor'].astype(str)))
        col_nome = 'Acao' if 'Acao' in meta.columns else 'Ticker'
        mapa_nomes = dict(zip(meta['Ticker_Norm'], meta[col_nome].astype(str)))
        mapa_pesos = dict(zip(meta['Ticker_Norm'], meta['Part. (%)'] if 'Part. (%)' in meta.columns else meta.get('Peso_Indice', 1.0 / len(meta))))

    indice = indice_df.iloc[:, 0].dropna()
    
    col_mapping = {col: normalize_ticker(col) for col in precos.columns}
    precos = precos.rename(columns=col_mapping)
    precos = precos.loc[:, ~precos.columns.duplicated()]

    pct_nulos = precos.loc[:DATA_CORTE_TREINO].isna().mean() * 100.0
    validas = [str(c) for c in pct_nulos[pct_nulos <= 5.0].index.tolist()]

    precos_clean = precos[validas].ffill().dropna(how='any')
    indice_clean = indice.ffill().dropna()

    if precos_clean.abs().max().max() > 2.0:
        ret_acoes = np.log(precos_clean / precos_clean.shift(1)).dropna(how='any')
    else:
        ret_acoes = precos_clean.copy()

    if indice_clean.abs().max() > 2.0:
        ret_idx = np.log(indice_clean / indice_clean.shift(1)).dropna()
    else:
        ret_idx = indice_clean.copy()

    datas_comuns = ret_acoes.index.intersection(ret_idx.index)
    ret_acoes = ret_acoes.loc[datas_comuns]
    ret_idx = ret_idx.loc[datas_comuns]
    precos_clean = precos_clean.loc[datas_comuns]

    setores_map = {str(c): mapa_set.get(str(c), 'Outros') for c in ret_acoes.columns}
    nomes_map = {str(c): mapa_nomes.get(str(c), str(c)) for c in ret_acoes.columns}
    pesos_map = pd.Series({str(c): mapa_pesos.get(str(c), 1.0 / len(ret_acoes.columns)) for c in ret_acoes.columns})
    
    ultimas_cotacoes = precos_clean.iloc[-1]
    data_ultima_cot = pd.Timestamp(precos_clean.index[-1])

    return ret_acoes, ret_idx, setores_map, nomes_map, pesos_map, ultimas_cotacoes, data_ultima_cot

# ==============================================================================
# 3. MOTOR COMPLETO DE ALGORITMOS E OTIMIZADORES UNIFICADOS
# ==============================================================================
MODELO_SLSQP = 'SLSQP + Ajuste de Volatilidade'
MODELO_ENET = 'Elastic Net Positivo'
MODELO_BASE_PESO = 'Baseline Top Peso'
MODELO_BASE_CORR = 'Baseline Top Corr'
MODELO_TOP3_SETOR = 'Top 3 por Setor'
MODELO_LASSO_OTIM = 'Lasso + Otimização'
MODELO_CLUSTER_OTIM = 'Cluster + Otimização'
MODELO_MIQP_GUROBI = 'MIQP (Gurobi)'

MODELOS = [
    MODELO_SLSQP, MODELO_ENET, MODELO_BASE_PESO, MODELO_BASE_CORR, 
    MODELO_TOP3_SETOR, MODELO_LASSO_OTIM, MODELO_CLUSTER_OTIM, MODELO_MIQP_GUROBI
]

ESCOLHA_AUTO = 'Campeão (menor TE)'
K_PADRAO = 15
PESO_MAX_PADRAO = 0.30
LAMBDA_VOL_PADRAO = 0.25
L1_RATIO_PADRAO = 0.5
KS_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 35]

def construir_features_treino(treino_X: pd.DataFrame, treino_y: pd.Series, setores_map: Dict[str, str]) -> pd.DataFrame:
    var_idx = float(treino_y.var())
    feats = pd.DataFrame(index=treino_X.columns)
    feats['corr_idx'] = treino_X.corrwith(treino_y).fillna(0)
    feats['beta'] = treino_X.apply(lambda s: float(s.cov(treino_y)) / var_idx if var_idx > 0 else 0.0)
    feats['vol_anual'] = treino_X.std() * np.sqrt(252.0)
    feats['te_ind'] = treino_X.sub(treino_y, axis=0).std() * np.sqrt(252.0)
    feats['setor'] = [setores_map.get(str(idx), 'Outros') for idx in feats.index]
    return feats.sort_values('corr_idx', ascending=False)

def otimizar_pesos_slsqp_puro(ret_acoes: pd.DataFrame, ret_indice: pd.Series, acoes: List[str], peso_max: float = 1.0) -> pd.Series:
    R = ret_acoes[acoes].values
    y = ret_indice.values
    n = len(acoes)
    def erro(w): return np.mean((y - R @ w) ** 2)
    res = minimize(erro, x0=np.ones(n) / n, method='SLSQP', bounds=[(0, peso_max)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}])
    return pd.Series(res.x, index=acoes)

def resolver_slsqp_avancado(treino_X: pd.DataFrame, treino_y: pd.Series, acoes: List[str], peso_max: float = 0.30, lambda_vol: float = 0.0, w_ant: Optional[pd.Series] = None, lambda_turn: float = 0.0) -> pd.Series:
    R = treino_X[acoes].to_numpy(dtype=float) * 100.0
    y = treino_y.to_numpy(dtype=float) * 100.0
    n = len(acoes)
    cov_anual = np.cov(R, rowvar=False) * 252.0
    vol_idx = float(np.std(y, ddof=1) * np.sqrt(252.0))
    w_prev_vec = w_ant.reindex(acoes, fill_value=0.0).to_numpy(dtype=float) if (w_ant is not None and not w_ant.empty and lambda_turn > 0) else None

    def func_obj(w: np.ndarray) -> float:
        mse = float(np.mean((y - R @ w) ** 2))
        obj = mse
        if lambda_vol > 0:
            v_c = float(np.sqrt(max(float(w.T @ cov_anual @ w), 1e-12)))
            obj += lambda_vol * (((v_c - vol_idx) ** 2) / 252.0)
        if w_prev_vec is not None and lambda_turn > 0:
            obj += (lambda_turn * 10000.0) * float(np.sum((w - w_prev_vec) ** 2))
        return obj

    w0 = w_prev_vec if (w_prev_vec is not None and float(w_prev_vec.sum()) > 0.5) else np.ones(n) / n
    w0 = w0 / float(w0.sum())
    ub = min(1.0, max(peso_max, 1.0 / n + 1e-3))
    res = minimize(func_obj, x0=w0, method='SLSQP', bounds=[(0.0, ub)] * n, constraints=[{'type': 'eq', 'fun': lambda w: float(np.sum(w)) - 1.0}], options={'ftol': 1e-11, 'maxiter': 750})
    w_s = pd.Series(np.maximum(res.x, 0.0), index=acoes)
    w_s = w_s[w_s > 1e-5]
    return w_s / float(w_s.sum())

def selecionar_top3_por_setor_ajustado(features: pd.DataFrame, k: int) -> List[str]:
    """
    Seleciona exatamente K ações respeitando a regra de poda setorial:
    1. Coleta até 3 ações por setor (ordenadas por maior corr_idx).
    2. Se a soma ultrapassar K, elimina a 3ª posição dos piores setores, 
       depois a 2ª posição, e por fim a 1ª posição dos piores setores.
    3. Preserva prioritariamente as ações dos setores com maior correlação média.
    """
    if 'setor' not in features.columns or 'corr_idx' not in features.columns:
        return list(features.sort_values(by='corr_idx', ascending=False).head(k).index)
    
    # Ranquear setores pela correlação média (do menor para o maior)
    setores_rank = (
        features.groupby('setor')['corr_idx']
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    
    # Pegar até 3 ações por setor
    setores_dict = {}
    for setor in setores_rank:
        acoes_setor = (
            features[features['setor'] == setor]
            .sort_values(by='corr_idx', ascending=False)
            .head(3)
            .index.tolist()
        )
        setores_dict[setor] = acoes_setor

    # Montar fila de descarte gradual: nível 2 (3ª ação) -> 1 (2ª ação) -> 0 (1ª ação)
    # Sempre do setor menos correlacionado para o mais correlacionado
    fila_descarte = []
    for nivel in [2, 1, 0]:
        for setor in setores_rank:
            if len(setores_dict[setor]) > nivel:
                fila_descarte.append(setores_dict[setor][nivel])

    # Inverter a fila para obter a ordem de preservação (Top prioridade primeiro)
    ordem_preservacao = fila_descarte[::-1]

    # Retorna EXATAMENTE K ações
    return ordem_preservacao[:k]
    """
    Seleciona K ações com base na regra de Top 3 por setor com poda dinâmica:
    1. Inicia com as 3 melhores ações de cada setor (ordenadas por corr_idx).
    2. Se K for menor que o total inicial, remove gradualmente as ações com menor 
       correlação de cada setor (rodada a rodada do pior setor para o melhor).
    3. Quando um setor estiver prestes a ser desativado, dá preferência a manter 
       as ações dos setores com maior correlação média com o índice.
    """
    if 'setor' not in features.columns or 'corr_idx' not in features.columns:
        return list(features.sort_values(by='corr_idx', ascending=False).head(k).index)
    
    # 1. Ranquear os setores pela correlação média de suas ações com o índice
    setores_rank = (
        features.groupby('setor')['corr_idx']
        .mean()
        .sort_values(ascending=True) # Do setor menos correlacionado para o mais correlacionado
        .index.tolist()
    )
    
    # 2. Agrupar até 3 ações por setor, ordenadas da maior para a menor correlação
    setores_dict = {}
    for setor in setores_rank:
        acoes_setor = (
            features[features['setor'] == setor]
            .sort_values(by='corr_idx', ascending=False)
            .head(3)
            .index.tolist()
        )
        setores_dict[setor] = acoes_setor  # [pos1 (melhor), pos2, pos3 (pior)]

    # 3. Fila de descarte gradual (elimina pos3 de todos os setores, depois pos2, depois pos1)
    # Sempre iterando do pior setor para o melhor setor na ordem de descarte
    fila_prioridade = []
    
    # Posições de descarte: 3ª ação -> 2ª ação -> 1ª ação
    for nivel in [2, 1, 0]:  # índices 2 (3ª posição), 1 (2ª posição), 0 (1ª posição)
        for setor in setores_rank:
            if len(setores_dict[setor]) > nivel:
                fila_prioridade.append(setores_dict[setor][nivel])

    # 4. Inverter a fila de descarte para obter a ordem de PRESERVAÇÃO
    # As últimas a serem descartadas são as primeiras a serem mantidas no K
    ordem_preservacao = fila_prioridade[::-1]

    # Retorna exatamente as K ações de maior prioridade
    return ordem_preservacao[:k]

def selecionar_por_cluster(ret_acoes: pd.DataFrame, features: pd.DataFrame, k: int) -> List[str]:
    corr = ret_acoes.corr().fillna(0)
    dist = np.sqrt(0.5 * (1 - corr).clip(lower=0))
    cl = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    rotulos = cl.fit_predict(dist.values)
    escolhidas = []
    for g in range(k):
        membros = ret_acoes.columns[rotulos == g]
        melhor = features.loc[membros, 'corr_idx'].idxmax()
        escolhidas.append(melhor)
    return escolhidas

def optimize_miqp_gurobi(ret_acoes: pd.DataFrame, ret_indice: pd.Series, k_max: int = 20, w_max: float = 0.30) -> pd.Series:
    tickers = ret_acoes.columns.tolist()
    N = len(tickers)
    excess_matrix = ret_acoes.sub(ret_indice, axis=0)
    Q = excess_matrix.cov().values * 252.0
    try:
        model = gp.Model("Index_Tracking_MIQP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 10)
        model.setParam('MIPGap', 0.01)
        w = model.addVars(N, lb=0.0, ub=w_max, vtype=GRB.CONTINUOUS, name="w")
        z = model.addVars(N, vtype=GRB.BINARY, name="z")
        model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1.0, name="budget")
        for i in range(N): model.addConstr(w[i] <= z[i] * w_max, name=f"link_{i}")
        model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k_max, name="cardinality")
        obj = gp.QuadExpr()
        for i in range(N):
            for j in range(N): obj += w[i] * Q[i, j] * w[j]
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()
        if model.SolCount > 0:
            weights = pd.Series([w[i].X for i in range(N)], index=tickers)
            weights = weights[weights > 1e-5]
            return weights / weights.sum()
    except Exception:
        pass
    top_corr = ret_acoes.corrwith(ret_indice).abs().sort_values(ascending=False).head(k_max).index
    return otimizar_pesos_slsqp_puro(ret_acoes, ret_indice, list(top_corr), peso_max=w_max)

def modelo_lasso(ret_acoes: pd.DataFrame, ret_indice: pd.Series, alpha: float = 1e-4, peso_max: float = 0.30) -> pd.Series:
    m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=50000)
    m.fit(ret_acoes.values, ret_indice.values)
    w = pd.Series(m.coef_, index=ret_acoes.columns)
    w = w[w > 1e-6]
    if len(w) == 0: return pd.Series(1.0 / len(ret_acoes.columns), index=ret_acoes.columns)
    return otimizar_pesos_slsqp_puro(ret_acoes, ret_indice, list(w.index), peso_max=peso_max)

def calcular_caminho_enet(treino_X: pd.DataFrame, treino_y: pd.Series, l1_ratio: float = L1_RATIO_PADRAO, n_alphas: int = 100, n_splits: int = 5):
    X = treino_X.to_numpy(dtype=float) * 100.0
    y = treino_y.to_numpy(dtype=float) * 100.0
    n_obs = X.shape[0]
    alpha_max = float(np.max(X.T @ y)) / (n_obs * l1_ratio)
    if not np.isfinite(alpha_max) or alpha_max <= 0.0: return np.array([1.0]), np.zeros((X.shape[1], 1)), 1.0
    alphas = alpha_max * np.logspace(0.0, -4.0, n_alphas)
    cv = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, positive=True, fit_intercept=False, cv=TimeSeriesSplit(n_splits=n_splits), max_iter=5000, tol=1e-6).fit(X, y)
    alphas_out, coefs, _ = enet_path(X, y, l1_ratio=l1_ratio, alphas=alphas, positive=True, max_iter=5000, tol=1e-6)
    return np.asarray(alphas_out), np.asarray(coefs), float(cv.alpha_)

def resolver_enet_positivo(treino_X: pd.DataFrame, treino_y: pd.Series, k: int = 15, peso_max: float = 0.30, l1_ratio: float = L1_RATIO_PADRAO, caminho=None) -> pd.Series:
    acoes = [str(c) for c in treino_X.columns]
    k = int(max(1, min(k, len(acoes))))
    if caminho is None: caminho = calcular_caminho_enet(treino_X, treino_y, l1_ratio)
    alphas, coefs, alpha_cv = caminho
    n_ativos = (coefs > 1e-10).sum(axis=0)
    elegiveis = np.where((alphas <= alpha_cv * (1.0 + 1e-9)) & (n_ativos >= k))[0]
    pos = int(elegiveis[0]) if len(elegiveis) > 0 else len(alphas) - 1
    alpha_sel = float(alphas[pos])
    coef_rank = coefs[:, pos]
    ordem = np.argsort(coef_rank)[::-1][:k]
    sel = ordem[coef_rank[ordem] > 1e-10]
    if len(sel) == 0:
        sel = np.argsort(treino_X.corrwith(treino_y).to_numpy(dtype=float))[::-1][:k]
        coef_sel = np.ones(len(sel))
    else:
        modelo = ElasticNet(alpha=alpha_sel, l1_ratio=l1_ratio, positive=True, fit_intercept=False, max_iter=10000, tol=1e-6)
        modelo.fit(treino_X.to_numpy(dtype=float)[:, sel] * 100.0, treino_y.to_numpy(dtype=float) * 100.0)
        coef_sel = np.asarray(modelo.coef_, dtype=float)
        if float(coef_sel.sum()) <= 1e-12: coef_sel = coef_rank[sel]
    ub = min(1.0, max(peso_max, 1.0 / len(sel) + 1e-3))
    w_s = pd.Series(coef_sel, index=[acoes[i] for i in sel])
    w_s = w_s[w_s > 1e-5]
    return w_s / float(w_s.sum())

# RUNNER UNIFICADO DE MODELOS
def treinar_modelo_especifico(modelo_nome: str, treino_X: pd.DataFrame, treino_y: pd.Series, setores_map: Dict[str, str], pesos_map: pd.Series, k: int, peso_max: float, lambda_vol: float, l1_ratio: float, w_ant=None) -> pd.Series:
    feats = construir_features_treino(treino_X, treino_y, setores_map)
    
    if modelo_nome == MODELO_BASE_PESO:
        top = pesos_map.reindex(treino_X.columns).sort_values(ascending=False).head(k)
        return top / top.sum()
    elif modelo_nome == MODELO_BASE_CORR:
        top = feats.sort_values('corr_idx', ascending=False).head(k).index
        return pd.Series(1.0 / k, index=top)
    elif modelo_nome == MODELO_TOP3_SETOR:
        acoes_top3 = selecionar_top3_por_setor_ajustado(feats, k=k)
        return otimizar_pesos_slsqp_puro(treino_X, treino_y, acoes_top3, peso_max=peso_max)
    elif modelo_nome == MODELO_LASSO_OTIM:
        return modelo_lasso(treino_X, treino_y, alpha=1e-4, peso_max=peso_max)
    elif modelo_nome == MODELO_CLUSTER_OTIM:
        acoes_cl = selecionar_por_cluster(treino_X, feats, k=k)
        return otimizar_pesos_slsqp_puro(treino_X, treino_y, acoes_cl, peso_max=peso_max)
    elif modelo_nome == MODELO_MIQP_GUROBI:
        return optimize_miqp_gurobi(treino_X, treino_y, k_max=k, w_max=peso_max)
    elif modelo_nome == MODELO_ENET:
        return resolver_enet_positivo(treino_X, treino_y, k=k, peso_max=peso_max, l1_ratio=l1_ratio)
    else: # MODELO_SLSQP
        cands = feats.sort_values('corr_idx', ascending=False).head(k * 2).index.tolist()
        return resolver_slsqp_avancado(treino_X, treino_y, cands[:k], peso_max=peso_max, lambda_vol=lambda_vol, w_ant=w_ant, lambda_turn=1e-4)

@st.cache_data(show_spinner=False)
def rodar_walk_forward_completo(
    ret_acoes: pd.DataFrame, ret_idx: pd.Series, setores_map: Dict[str, str], pesos_map: pd.Series,
    k: int = 15, peso_max: float = 0.30, lambda_vol: float = 0.25, lambda_turn: float = 1e-4,
    custo_bps: int = 10, modelo: str = MODELO_SLSQP, l1_ratio: float = L1_RATIO_PADRAO
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    inicio = pd.to_datetime('2025-01-01')
    pontos = pd.date_range(inicio, ret_acoes.index[-1], freq='3MS')
    if pontos[-1] < ret_acoes.index[-1]:
        pontos = pontos.union([ret_acoes.index[-1] + pd.Timedelta(days=1)])

    ret_oos_bruto, ret_oos_liq, turnovers, janelas = [], [], [], {}
    w_ant = pd.Series(dtype=float)

    for t0, t1 in zip(pontos[:-1], pontos[1:]):
        tr_x = ret_acoes.loc[t0 - pd.DateOffset(months=24) : t0 - pd.Timedelta(days=1)]
        tr_y = ret_idx.loc[tr_x.index]

        w_novo = treinar_modelo_especifico(modelo, tr_x, tr_y, setores_map, pesos_map, k, peso_max, lambda_vol, l1_ratio, w_ant)

        if not w_ant.empty:
            uniao = w_novo.index.union(w_ant.index)
            diff_w = w_novo.reindex(uniao, fill_value=0.0) - w_ant.reindex(uniao, fill_value=0.0)
            turn = 0.5 * float(diff_w.abs().sum())
            turnovers.append(turn)
        else:
            turn = 1.0
        w_ant = w_novo

        cols_w = [str(c) for c in w_novo.index.tolist() if str(c) in ret_acoes.columns]
        bloco_x = ret_acoes.loc[t0 : t1 - pd.Timedelta(days=1), cols_w]
        if len(bloco_x) > 0:
            rb: pd.Series = bloco_x @ w_novo.loc[cols_w]
            rl: pd.Series = rb.copy()
            rl.iloc[0] = float(rl.iloc[0]) - turn * (custo_bps / 10000.0)
            ret_oos_bruto.append(rb)
            ret_oos_liq.append(rl)

            yb = ret_idx.loc[bloco_x.index]
            r_cart = (float(np.exp(rl.sum())) - 1.0) * 100.0
            r_idx = (float(np.exp(yb.sum())) - 1.0) * 100.0
            te_j = float((rb - yb).std() * np.sqrt(252.0) * 100.0) if len(bloco_x) > 1 else np.nan
            
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

    df_wf = pd.DataFrame([
        {
            'K Efetivo': int((w_ant > 1e-5).sum()),
            'TE Walk-Forward (% a.a.)': round(te, 2),
            'Vol. Carteira (%)': round(vol_c, 2),
            'Gap Vol (p.p.)': round(abs(vol_c - vol_i), 2),
            'Correlação (r)': round(float(sb.corr(si)), 4),
            'R²': round(1.0 - ss_res / ss_tot, 4),
            'Turnover Médio Trim. (%)': round(float(np.mean(turnovers)) * 100.0, 1) if len(turnovers) > 0 else 0.0,
            'TD Bruta (%)': round(td_b, 2),
            'TD Líquida (c/ custos %)': round(td_l, 2),
            'Carteiras OOS': len(turnovers) + 1,
        }
    ], index=pd.Index([modelo], name='Modelo'))
    
    df_janelas = pd.DataFrame.from_dict(janelas, orient='index')
    return df_wf, sl, w_ant, si, df_janelas

@st.cache_data(show_spinner=False)
def calcular_curva_k(ret_acoes: pd.DataFrame, ret_idx: pd.Series, setores_map: Dict[str, str], pesos_map: pd.Series, peso_max: float = 0.30, lambda_vol: float = 0.25, modelo: str = MODELO_SLSQP, l1_ratio: float = L1_RATIO_PADRAO) -> pd.DataFrame:
    tr_x = ret_acoes.loc[:DATA_CORTE_TREINO]
    tr_y = ret_idx.loc[:DATA_CORTE_TREINO]
    oos_x = ret_acoes.loc[DATA_CORTE_TREINO:].iloc[1:]
    oos_y = ret_idx.loc[oos_x.index]

    linhas = []
    for k_i in KS_GRID:
        w_i = treinar_modelo_especifico(modelo, tr_x, tr_y, setores_map, pesos_map, k_i, peso_max, lambda_vol, l1_ratio)
        cols_i = [str(c) for c in w_i.index.tolist() if str(c) in oos_x.columns]
        te_i = float((oos_x[cols_i] @ w_i.loc[cols_i] - oos_y).std() * np.sqrt(252.0) * 100.0)
        linhas.append({'K': k_i, 'TE': round(te_i, 2)})
    return pd.DataFrame(linhas)

def rodar_todos_modelos(ret_acoes, ret_idx, setores_map, pesos_map, k, peso_max, lambda_vol, l1_ratio):
    res = {}
    for m in MODELOS:
        res[m] = rodar_walk_forward_completo(ret_acoes, ret_idx, setores_map, pesos_map, k=k, peso_max=peso_max, lambda_vol=lambda_vol, modelo=m, l1_ratio=l1_ratio)
    return res

def calcular_curvas_k_todos(ret_acoes, ret_idx, setores_map, pesos_map, peso_max, lambda_vol, l1_ratio):
    res = {}
    for m in MODELOS:
        res[m] = calcular_curva_k(ret_acoes, ret_idx, setores_map, pesos_map, peso_max=peso_max, lambda_vol=lambda_vol, modelo=m, l1_ratio=l1_ratio)
    return res

# ==============================================================================
# 4. AUXILIARES VISUAIS E TABELAS
# ==============================================================================
def obter_url_logo(ticker: str, is_ibov: bool) -> str:
    t_limpo = ticker.replace('.SA', '').strip()
    dom = DOMINIOS_IBOV.get(t_limpo) if is_ibov else DOMINIOS_SP100.get(t_limpo)
    if dom: return f'https://www.google.com/s2/favicons?domain={dom}&sz=128'
    return ''

def renderizar_cards_carteira(pesos: pd.Series, nomes_map: Dict[str, str], setores_map: Dict[str, str], cotacoes: pd.Series, is_ibov: bool, cor_barra: str) -> str:
    moeda = 'R$' if is_ibov else 'US$'
    w_ord = pesos.sort_values(ascending=False)
    max_w = float(w_ord.max()) if len(w_ord) > 0 else 1.0
    cards_html = ['<div class="stocks-grid">']
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

        img_tag = f"<img src='{url_logo}' class='logo-img' onerror=\"this.style.display='none';\" />" if url_logo else ''
        card_item = (
            f'<div class="stock-card">'
            f'<div class="logo-wrap"><span class="logo-fallback">{sigla_curta}</span>{img_tag}</div>'
            f'<div class="stock-info">'
            f'<div class="stock-top"><span class="stock-ticker">{t_clean}</span>'
            f'<span class="stock-weight" style="color: {cor_barra} !important;">{w * 100:.2f}%</span></div>'
            f'<div class="stock-name" title="{nome_emp}">{nome_emp}</div>'
            f'<div class="stock-sector" title="{setor_emp}">{setor_emp}</div>'
            f'<div class="stock-price">{preco_txt}</div>'
            f'<div class="weight-bar-bg"><div style="width: {larg_pct:.1f}%; height: 100%; background-color: {cor_barra};"></div></div>'
            f'</div></div>'
        )
        cards_html.append(card_item)
    cards_html.append('</div>')
    return ''.join(cards_html)

def renderizar_tabela_html(df: pd.DataFrame, col_destaque: str = 'TE Walk-Forward (% a.a.)', cor_header: str = '#0F4C81', nome_indice_col: str = 'Família de Modelo') -> str:
    melhor_idx = str(df[col_destaque].idxmin()) if col_destaque in df.columns else ''
    html = ['<div class="finor-table-container"><table class="finor-table">']
    html.append(f'<thead style="background-color: {cor_header} !important;"><tr><th>{nome_indice_col}</th>')
    for c in df.columns: html.append(f'<th style="background-color: {cor_header} !important; color: #FFFFFF !important;">{c}</th>')
    html.append('</tr></thead><tbody>')
    for idx_raw, row in df.iterrows():
        idx_str = str(idx_raw)
        is_best = idx_str == melhor_idx
        tr_class = ' class="best-row"' if is_best else ''
        nome_exib = f'{idx_str} <span class="badge-best">Campeão</span>' if is_best else idx_str
        html.append(f'<tr{tr_class}><td>{nome_exib}</td>')
        for c in df.columns:
            val = row[c]
            if isinstance(val, (int, np.integer)) or c in ['K Efetivo', 'Carteiras OOS', 'Qtd. Ações (Inteiras)']: txt = f'{int(val):,}'.replace(',', '.')
            elif c in ['Correlação (r)', 'R²']: txt = f'{float(val):.4f}'
            elif isinstance(val, (float, np.floating)): txt = f'{float(val):,.2f}'
            else: txt = str(val)
            html.append(f'<td>{txt}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return ''.join(html)

def configurar_grafico_padrao(fig: go.Figure, altura: int = 460, y_title: str = '', x_title: str = 'Data') -> go.Figure:
    fig.update_layout(
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', height=altura,
        font=dict(family='sans-serif', size=12, color='#222222'), hovermode='x unified',
        hoverlabel=dict(bgcolor='#FFFFFF', font_size=12, font_color='#1A202C', bordercolor='#CBD5E1'),
        margin=dict(l=65, r=25, t=55, b=55),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, bgcolor='rgba(255,255,255,0.95)', bordercolor='#E2E8F0', borderwidth=1, font=dict(size=11.5, color='#1A202C')),
        xaxis=dict(title=dict(text=x_title, font=dict(size=12.5, color='#1A202C')), tickfont=dict(size=11, color='#333333'), showgrid=False, showline=True, linecolor='#CCCCCC', linewidth=1, ticks='outside', tickcolor='#CCCCCC'),
        yaxis=dict(title=dict(text=y_title, font=dict(size=12.5, color='#1A202C')), tickfont=dict(size=11, color='#333333'), showgrid=True, gridcolor='#E5E5E5', gridwidth=1, griddash='dash', showline=True, linecolor='#CCCCCC', linewidth=1, zeroline=False)
    )
    return fig

# ==============================================================================
# 5. SIDEBAR: SELETOR DE ÍNDICE E PARÂMETROS DA CARTEIRA
# ==============================================================================
st.sidebar.markdown('### >>>>> ÍNDICE & PARÂMETROS')
mercado_sel = st.sidebar.radio('Selecione o Índice (Benchmark)', ['S&P 100 (EUA)', 'IBOVESPA (Brasil)'], index=0)

escolha_modelo = st.sidebar.radio(
    'Carteira exibida nas abas 1 e 4',
    [ESCOLHA_AUTO] + MODELOS, index=0,
    help='As abas 2 e 3 sempre comparam todos os algoritmos. "Campeão" exibe o de menor tracking error walk-forward.'
)

is_ibov = 'IBOV' in mercado_sel
cor_primaria = CORES['ibov'] if is_ibov else CORES['sp100']
cor_secundaria = CORES['ibov_sec'] if is_ibov else CORES['sp100_sec']
moeda_simb = 'R$' if is_ibov else 'US$'

k_val = st.sidebar.slider('Cardinalidade da Carteira (K ações)', min_value=5, max_value=35, value=K_PADRAO, step=1)
peso_max_val = st.sidebar.slider('Peso Máximo por Ação (%)', min_value=7, max_value=30, value=int(PESO_MAX_PADRAO * 100), step=1) / 100.0
lambda_vol_val = st.sidebar.slider('Ajuste de Volatilidade (λ_vol)', min_value=0.0, max_value=1.0, value=LAMBDA_VOL_PADRAO, step=0.05)
l1_ratio_val = float(st.sidebar.slider('Mix L1/L2 do Elastic Net (ρ)', min_value=0.1, max_value=1.0, value=L1_RATIO_PADRAO, step=0.1))

st.sidebar.markdown('---')
st.sidebar.markdown('### >>>>> PROTOCOLO DE VALIDAÇÃO')
st.sidebar.info('**Validação Walk-Forward Estrita**\n\n• **Janela Móvel:** 24 meses\n• **Rebalanceamento:** Trimestral OOS\n• **Custos de Transação:** 0,10% (10 bps)')

ret_acoes_full, ret_idx_full, setores_map, nomes_map, pesos_map, ultimas_cotacoes, data_ultima_cot = carregar_mercado_completo(mercado_sel)

with st.spinner('Reprocessando as carteiras com os novos parâmetros...'):
    resultados = rodar_todos_modelos(ret_acoes_full, ret_idx_full, setores_map, pesos_map, k_val, peso_max_val, lambda_vol_val, l1_ratio_val)
    curvas_k = calcular_curvas_k_todos(ret_acoes_full, ret_idx_full, setores_map, pesos_map, peso_max_val, lambda_vol_val, l1_ratio_val)

df_wf_comp = pd.concat([resultados[m][0] for m in MODELOS])
if escolha_modelo == ESCOLHA_AUTO:
    modelo_sel = str(df_wf_comp['TE Walk-Forward (% a.a.)'].idxmin())
else:
    modelo_sel = escolha_modelo

_, _, pesos_atuais, idx_oos, df_janelas = resultados[modelo_sel]

CORES_MODELOS = {
    MODELO_SLSQP: cor_primaria, MODELO_ENET: CORES['laranja'], MODELO_BASE_PESO: CORES['neutro'],
    MODELO_BASE_CORR: '#4A5568', MODELO_TOP3_SETOR: '#2B6999', MODELO_LASSO_OTIM: CORES['roxo'],
    MODELO_CLUSTER_OTIM: '#D97706', MODELO_MIQP_GUROBI: '#1B4D3E'
}

metricas = df_wf_comp.loc[modelo_sel]
te_melhor = float(metricas['TE Walk-Forward (% a.a.)'])
td_liquida = float(metricas['TD Líquida (c/ custos %)'])
vol_cart_oos = float(metricas['Vol. Carteira (%)'])
vol_idx_oos = float(idx_oos.std() * np.sqrt(252.0) * 100.0)

# ==============================================================================
# 6. CABEÇALHO E KPIS
# ==============================================================================
emblema_html = (
    '<div class="idx-emblem" style="background: linear-gradient(135deg, #1B4D3E 0%, #2E7D67 100%);">'
    '<span style="font-size: 1.15rem; line-height: 1;">[B]³</span>'
    '<span style="font-size: 0.65rem; font-weight: 700; margin-top: 3px; letter-spacing: 0.05em;">IBOV</span></div>'
    if is_ibov else
    '<div class="idx-emblem" style="background: linear-gradient(135deg, #0F4C81 0%, #2B6999 100%);">'
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
    f'<div class="kpi-label">MODELO SELECIONADO</div>'
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
# 7. ABAS DO DASHBOARD FINAL
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '1. Carteira Campeã & Simulador',
    '2. Desempenho Walk-Forward',
    '3. Curva Desvio × Nº de Ações (K)',
    '4. Equilíbrio Setorial & Risco',
    '5. Exportar Relatório Executivo'
])

# ------------------------------------------------------------------------------
# ABA 1: VITRINE DA CARTEIRA E SIMULADOR
# ------------------------------------------------------------------------------
with tab1:
    col_aporte, _col_vazia = st.columns([1.0, 1.3])
    with col_aporte:
        valor_padrao = 100000.0 if is_ibov else 50000.0
        passo_val = 5000.0 if is_ibov else 2500.0
        valor_investir = float(st.number_input(f'Valor do aporte para simulação de compra ({moeda_simb}):', min_value=1000.0, max_value=100000000.0, value=valor_padrao, step=passo_val, format='%.2f'))

    w_atual = pesos_atuais.sort_values(ascending=False)

    st.markdown(f'<div class="sec-title">>>>>> Composição da Carteira Recomendada: {modelo_sel} ({len(w_atual)} Ações)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-sub">Pesos alocados no último rebalanceamento e cotação de fechamento em <b>{data_ultima_cot.strftime("%d/%m/%Y")}</b>.</div>', unsafe_allow_html=True)

    st.markdown(renderizar_cards_carteira(w_atual, nomes_map, setores_map, ultimas_cotacoes, is_ibov, cor_primaria), unsafe_allow_html=True)

    cotacoes_cart = ultimas_cotacoes.reindex(w_atual.index).fillna(1.0)
    orcamento_alvo = w_atual * valor_investir
    qtd_inteira = np.floor(orcamento_alvo / cotacoes_cart).astype(int)
    valor_efetivo = qtd_inteira * cotacoes_cart
    peso_efetivo = (valor_efetivo / valor_investir) * 100.0
    caixa_residual = valor_investir - float(valor_efetivo.sum())

    st.markdown(f'<div class="sec-title">>>>>> Calculadora de Execução: Quantidade de Ações para {moeda_simb} {valor_investir:,.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-sub">Lotes inteiros calculados pela última cotação — <b>Total Alocado: {moeda_simb} {float(valor_efetivo.sum()):,.2f} ({float(peso_efetivo.sum()):.2f}%)</b> | <b>Saldo Residual em Caixa: {moeda_simb} {caixa_residual:,.2f}</b></div>', unsafe_allow_html=True)

    df_boleta = pd.DataFrame({
        'Empresa': [nomes_map.get(str(t), str(t).replace('.SA', '')) for t in w_atual.index],
        'Setor': [setores_map.get(str(t), 'Outros') for t in w_atual.index],
        'Peso Modelo (%)': (w_atual.to_numpy(dtype=float) * 100.0).round(2),
        f'Cotação ({moeda_simb})': cotacoes_cart.to_numpy(dtype=float).round(2),
        f'Orçamento Alvo ({moeda_simb})': orcamento_alvo.to_numpy(dtype=float).round(2),
        'Qtd. Ações (Inteiras)': qtd_inteira.to_numpy(dtype=int),
        f'Valor Comprado ({moeda_simb})': valor_efetivo.to_numpy(dtype=float).round(2),
        'Peso Realizado (%)': peso_efetivo.to_numpy(dtype=float).round(2),
    }, index=[str(t).replace('.SA', '') for t in w_atual.index])

    st.markdown(renderizar_tabela_html(df_boleta, col_destaque='', cor_header=cor_primaria, nome_indice_col='Ticker'), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# ABA 2: DESEMPENHO WALK-FORWARD
# ------------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="sec-title">>>>>> Validação Walk-Forward (Carteiras Trimestrais Out-of-Sample)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Trajetória acumulada fora da amostra (Base 100 em Jan/2025) descontando custos de transação (0,10%)</div>', unsafe_allow_html=True)

    idx_acum = np.exp(idx_oos.cumsum()) * 100.0
    fig_wf = go.Figure()
    fig_wf.add_hline(y=100, line_dash='dot', line_color='#A0AEC0', line_width=1.2)
    fig_wf.add_trace(go.Scatter(x=idx_acum.index, y=idx_acum, mode='lines', name=f'Benchmark ({mercado_sel})', line=dict(color=CORES['grafite'], width=3.3)))
    
    selected_models_plot = st.multiselect("Selecione os modelos para exibir no gráfico:", options=MODELOS, default=[MODELO_SLSQP, MODELO_ENET, MODELO_TOP3_SETOR, MODELO_MIQP_GUROBI])

    for m_nome in selected_models_plot:
        cart_acum_m = np.exp(resultados[m_nome][1].cumsum()) * 100.0
        te_m = float(df_wf_comp.loc[m_nome, 'TE Walk-Forward (% a.a.)'])
        fig_wf.add_trace(go.Scatter(x=cart_acum_m.index, y=cart_acum_m, mode='lines', name=f'{m_nome} (TE: {te_m:.2f}%)', line=dict(color=CORES_MODELOS.get(m_nome, '#888888'), width=2.5)))

    fig_wf = configurar_grafico_padrao(fig_wf, altura=470, y_title=f'Evolução do Patrimônio (Base 100 = {moeda_simb} 100)', x_title='Data')
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown('<div class="sec-title">>>>>> Resumo Consolidado de Todos os Modelos</div>', unsafe_allow_html=True)
    st.markdown(renderizar_tabela_html(df_wf_comp, col_destaque='TE Walk-Forward (% a.a.)', cor_header=cor_primaria, nome_indice_col='Modelo'), unsafe_allow_html=True)

    st.markdown(f'<div class="sec-title">>>>>> Detalhe por Carteira Trimestral: {modelo_sel}</div>', unsafe_allow_html=True)
    st.markdown(renderizar_tabela_html(df_janelas, col_destaque='', cor_header=cor_primaria, nome_indice_col='Carteira'), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# ABA 3: CURVA DE TRACKING ERROR X CARDINALIDADE (K)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="sec-title">>>>>> Curva de Trade-Off: Tracking Error × Número de Ações (K)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Avaliação da redução do desvio acumulado fora da amostra de K = 5 a K = 35 ações</div>', unsafe_allow_html=True)

    fig_k = go.Figure()
    for m_nome in selected_models_plot:
        fig_k.add_trace(go.Scatter(x=curvas_k[m_nome]['K'], y=curvas_k[m_nome]['TE'], mode='lines+markers', name=m_nome, line=dict(color=CORES_MODELOS.get(m_nome, '#888888'), width=2.5), marker=dict(size=7)))
    
    fig_k.add_vline(x=k_val, line_dash='dash', line_color=CORES['alerta'], line_width=1.5)
    fig_k = configurar_grafico_padrao(fig_k, altura=450, x_title='Número de Ações na Carteira (K)', y_title='Tracking Error Fora da Amostra (% a.a.)')
    st.plotly_chart(fig_k, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 4: EQUILÍBRIO SETORIAL E RISCO
# ------------------------------------------------------------------------------
with tab4:
    col_p1, col_p2 = st.columns([1.05, 1.0])
    with col_p1:
        st.markdown('<div class="sec-title">>>>>> Aderência de Volatilidade: Carteira vs. Índice</div>', unsafe_allow_html=True)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(y=['Índice (benchmark)', 'Carteira'], x=[vol_idx_oos, vol_cart_oos], orientation='h', marker_color=[CORES['grafite'], cor_primaria], text=[f'{vol_idx_oos:.2f}%', f'{vol_cart_oos:.2f}%'], textposition='outside'))
        fig_vol = configurar_grafico_padrao(fig_vol, altura=420, x_title='Volatilidade Anualizada (% a.a.)', y_title='')
        fig_vol.update_xaxes(range=[0, max(vol_idx_oos, vol_cart_oos) * 1.28])
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_p2:
        st.markdown('<div class="sec-title">>>>>> Equilíbrio Setorial: Carteira vs. Índice</div>', unsafe_allow_html=True)
        s_set_cart = pd.Series(pesos_atuais.to_numpy(dtype=float) * 100.0, index=[setores_map.get(str(t), 'Outros') for t in pesos_atuais.index]).groupby(level=0).sum()
        s_set_univ = pd.Series(1.0, index=[setores_map.get(str(t), 'Outros') for t in ret_acoes_full.columns]).groupby(level=0).sum()
        s_set_univ = (s_set_univ / float(s_set_univ.sum())) * 100.0

        df_set_comp = pd.DataFrame({'Carteira Campeã (%)': s_set_cart, 'Universo do Índice (%)': s_set_univ}).fillna(0.0).sort_values('Carteira Campeã (%)', ascending=True)

        fig_set = go.Figure()
        fig_set.add_trace(go.Bar(y=df_set_comp.index, x=df_set_comp['Universo do Índice (%)'], name='Proporção no Índice (%)', orientation='h', marker_color=CORES['neutro']))
        fig_set.add_trace(go.Bar(y=df_set_comp.index, x=df_set_comp['Carteira Campeã (%)'], name='Peso na Carteira (%)', orientation='h', marker_color=cor_primaria))
        fig_set.update_layout(barmode='group')
        fig_set = configurar_grafico_padrao(fig_set, altura=420, x_title='Participação Setorial (%)', y_title='Setor Econômico')
        st.plotly_chart(fig_set, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 5: EXPORTAÇÃO COMPLETA DE RELATÓRIO EXECUTIVO
# ------------------------------------------------------------------------------
with tab5:
    st.markdown('<div class="sec-title">>>>>> Exportação de Relatórios e Resultados</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Baixe os dados compilados em formato CSV ou gere um relatório impresso completo contendo as métricas de todos os 8 modelos e a boleta de compra recomendada.</div>', unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_data = df_wf_comp.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📄 Baixar Tabela de Métricas Comparativas (CSV)",
            data=csv_data,
            file_name=f"metricas_index_tracking_{mercado_sel}_K{k_val}.csv",
            mime="text/csv"
        )

    with col_exp2:
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Relatório Index Tracking - {mercado_sel}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #1A202C; }}
                h1 {{ color: {cor_primaria}; border-bottom: 2px solid {cor_primaria}; padding-bottom: 8px; }}
                h2 {{ color: {cor_secundaria}; margin-top: 30px; }}
                .highlight {{ background-color: #F8FAFC; padding: 15px; border-left: 5px solid {cor_primaria}; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 15px; margin-bottom: 25px; }}
                th, td {{ border: 1px solid #CBD5E1; text-align: left; padding: 10px; }}
                th {{ background-color: #F1F5F9; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #F8FAFC; }}
            </style>
        </head>
        <body>
            <h1>Relatório Executivo de Index Tracking: {mercado_sel}</h1>
            <p><b>Período Walk-Forward:</b> {dt_ini_str} até {dt_fim_str} | <b>Cardinalidade (K):</b> {k_val}</p>
            
            <div class="highlight">
                <h3>🏆 Modelo Selecionado: {modelo_sel}</h3>
                <ul>
                    <li><b>Tracking Error Anualizado:</b> {te_melhor:.2f}% a.a.</li>
                    <li><b>Diferença Acumulada Líquida:</b> {td_liquida:+.2f} p.p.</li>
                    <li><b>Correlação com o Índice:</b> {float(metricas['Correlação (r)']):.4f}</li>
                    <li><b>Coeficiente R²:</b> {float(metricas['R²']):.4f}</li>
                    <li><b>Total de Ações Alocadas:</b> {len(df_boleta)}</li>
                </ul>
            </div>

            <h2>1. Métricas Comparativas de Todos os 8 Modelos</h2>
            {df_wf_comp.to_html()}

            <h2>2. Calculadora de Compra e Alocação da Carteira ({modelo_sel})</h2>
            <p>Aporte Simulado: <b>{moeda_simb} {valor_investir:,.2f}</b> | Saldo Residual: <b>{moeda_simb} {caixa_residual:,.2f}</b></p>
            {df_boleta.to_html()}
        </body>
        </html>
        """
        st.download_button(
            label="🖨️ Baixar Relatório Executivo Completo (HTML / PDF)",
            data=html_report.encode('utf-8'),
            file_name=f"relatorio_index_tracking_{mercado_sel}_K{k_val}.html",
            mime="text/html"
        )