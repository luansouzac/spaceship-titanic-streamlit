# ============================================================
# App Streamlit - Spaceship Titanic
# Carrega os modelos treinados (.pkl), recebe os dados de um passageiro
# e mostra se ele foi transportado para outra dimensão.
# ============================================================
import os
import joblib            # lê os arquivos .pkl (modelos, colunas e padronizador)
import altair as alt     # usado no gráfico da aba "Gráficos"
import pandas as pd
import streamlit as st

# Título e ícone da aba do navegador. Sempre deve ser o primeiro comando do Streamlit.
st.set_page_config(page_title='Spaceship Titanic', page_icon=':material/rocket_launch:', layout='centered')

# Pasta onde estão os .pkl: a pasta "modelos" ao lado deste arquivo (funciona no Colab e no computador)
PASTA_MODELOS = os.environ.get('PASTA_MODELOS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos'))

# Nome da coluna no dataset -> nome que aparece na tela
COLUNAS_GASTOS = {'RoomService': 'Serviço de quarto', 'FoodCourt': 'Restaurante', 'ShoppingMall': 'Shopping',
          'Spa': 'Spa', 'VRDeck': 'Realidade virtual'}

# Nome que aparece no seletor -> arquivo .pkl do modelo treinado
ARQUIVOS_MODELOS = {
    'CatBoost (melhor)': 'modelo_catboost.pkl',
    'Random Forest': 'modelo_random_forest.pkl',
    'KNN': 'modelo_knn.pkl',
    'Regressão Logística': 'modelo_regressao_logistica.pkl',
}

# ---------------- Visual (CSS) ----------------
# O Streamlit permite injetar CSS. Aqui deixamos o app com o estilo da NASA
# (cantos retos, títulos em maiúsculas, faixa no topo) nas cores roxas.
# Nada daqui interfere na previsão, é só aparência.
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top, #1b1f4b 0%, #0b0d24 60%); }
html, body, [class*="css"], .stApp { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
h1, h2, h3, p, label, span, div { color: #fff; }
header[data-testid="stHeader"] { background: transparent; }
.topo { border-top: 6px solid #c77dff; padding: 1.6rem 0 1.2rem; margin-bottom: 0.5rem; border-bottom: 1px solid #3a3f7a; }
.topo small { color: #c77dff; letter-spacing: 0.25em; font-weight: 700; font-size: 0.8rem; }
.topo h1 { background: linear-gradient(90deg, #7c9cff, #c77dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.2rem; line-height: 1; margin: 0.4rem 0 0.6rem; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 800; }
.topo p { color: #b8b8b8; margin: 0; font-size: 1.05rem; }
/* Cartão de resultado: .sim (roxo) quando transportado, .nao (vinho) quando não */
.resultado { border-radius: 0; padding: 1.4rem 1.6rem; margin: 1.2rem 0; font-size: 1.8rem; font-weight: 800;
             text-transform: uppercase; letter-spacing: 0.06em; border-left: 10px solid; }
.sim { background: linear-gradient(90deg, #5a3fd0, #9d4edd); border-color: #fff; }
.nao { background: linear-gradient(90deg, #7a2740, #c0485f); border-color: #fff; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border: 1px solid #3a3f7a; border-radius: 0; padding: 1rem 1.2rem; }
div[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem; color: #b8b8b8; }
div[data-testid="stMetricValue"] { font-weight: 800; }
section[data-testid="stSidebar"] { background: #0f1230; border-right: 1px solid #3a3f7a; }
section[data-testid="stSidebar"] h2 { text-transform: uppercase; letter-spacing: 0.08em; font-size: 1.1rem; }
button[data-baseweb="tab"] { text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
div[data-baseweb="tab-highlight"] { background-color: #c77dff; }
div[data-baseweb="select"] > div, input { border-radius: 0 !important; }
.stButton > button { border-radius: 0; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; padding: 0.8rem; border: none; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #5a3fd0, #9d4edd); color: #fff; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.15); }
</style>
<div class="topo"><small>MISSÃO DE ANÁLISE DE DADOS</small><h1>Spaceship Titanic</h1><p>Preencha os dados do passageiro e veja se ele foi transportado para outra dimensão.</p></div>
""", unsafe_allow_html=True)


# ---------------- Carregar arquivos ----------------
@st.cache_resource   # lê o .pkl do disco uma vez só e guarda na memória (o app fica mais rápido)
def carregar(arquivo):
    return joblib.load(os.path.join(PASTA_MODELOS, arquivo))


# ---------------- Preparar os dados do passageiro ----------------
def montar_linha(passageiro, colunas_treino):
    """Transforma o que o usuário digitou numa linha igual às do treino.

    O modelo só entende a tabela no formato em que foi treinado. Por isso repetimos
    aqui o mesmo tratamento do notebook de treino, para um único passageiro.
    """
    # Regra descoberta na análise: quem está em CryoSleep ou tem menos de 13 anos nunca gasta
    sem_gasto = passageiro['sono_criogenico'] or passageiro['idade'] < 13
    gastos = {nome_gasto: 0 if sem_gasto else passageiro[nome_gasto] for nome_gasto in COLUNAS_GASTOS}

    linha_passageiro = {
        'CryoSleep': int(passageiro['sono_criogenico']), 'Age': passageiro['idade'], 'VIP': int(passageiro['passageiro_vip']), **gastos,
        'TamanhoGrupo': passageiro['tamanho_grupo'], 'NumCabine': passageiro['numero_cabine'],
        # Variáveis criadas no treino: soma de todos os gastos, gastos de luxo e gastos básicos
        'GastoTotal': sum(gastos.values()),
        'GastoLuxo': gastos['RoomService'] + gastos['Spa'] + gastos['VRDeck'],
        'GastoBasico': gastos['FoodCourt'] + gastos['ShoppingMall'],
    }
    # One-hot encoding: no treino cada categoria virou uma coluna 0/1 (ex.: HomePlanet_Europa).
    # Aqui marcamos 1 só na categoria escolhida; as outras colunas ficam 0 no reindex abaixo.
    for prefixo, valor in [('HomePlanet', passageiro['planeta']), ('Destination', passageiro['destino']),
                           ('Deck', passageiro['deck']), ('Lado', passageiro['lado'])]:
        linha_passageiro[f'{prefixo}_{valor}'] = 1
    # reindex coloca as colunas na MESMA ORDEM do treino (lista salva em colunas_treino.pkl)
    return pd.DataFrame([linha_passageiro]).reindex(columns=colunas_treino, fill_value=0)


# ---------------- Barra lateral (st.sidebar) ----------------
with st.sidebar:
    st.header(':material/tune: Modelo')
    nome_modelo = st.selectbox('Escolha o modelo', list(ARQUIVOS_MODELOS))
    st.caption('O CatBoost teve a melhor acurácia na validação cruzada (81,3%).')

# ---------------- Abas (st.tabs) ----------------
aba_prever, aba_graficos, aba_sobre = st.tabs([':material/query_stats: Prever', ':material/bar_chart: Gráficos', ':material/info: Como funciona'])

# ===== Aba 1: formulário + previsão =====
with aba_prever:
    # Dados principais do passageiro
    coluna_idade, coluna_planeta, coluna_destino = st.columns(3)
    idade = coluna_idade.number_input('Idade', 0, 90, 28)
    planeta = coluna_planeta.selectbox('Planeta de origem', ['Earth', 'Europa', 'Mars'])
    destino = coluna_destino.selectbox('Destino', ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])

    coluna_sono, coluna_vip = st.columns(2)
    sono_criogenico = coluna_sono.toggle('Em sono criogênico')
    passageiro_vip = coluna_vip.toggle('É VIP')

    # Os campos de gasto ficam desativados quando a regra "não gasta" se aplica
    sem_gasto = sono_criogenico or idade < 13
    st.markdown('**Gastos a bordo**')
    if sem_gasto:
        st.caption('Quem está em sono criogênico ou tem menos de 13 anos não gasta nada.')
    gastos = {}
    for coluna_da_tela, (nome_coluna, rotulo_gasto) in zip(st.columns(5), COLUNAS_GASTOS.items()):
        gastos[nome_coluna] = coluna_da_tela.number_input(rotulo_gasto, 0, 30000, 0, step=100, disabled=sem_gasto)

    # Campos que o modelo usa, mas que o usuário comum não precisa mexer (têm valor padrão)
    with st.expander('Detalhes da cabine e do grupo (opcional)'):
        coluna_deck, coluna_lado, coluna_cabine, coluna_grupo = st.columns(4)
        deck = coluna_deck.selectbox('Deck', list('ABCDEFGT'), index=6)
        lado = coluna_lado.selectbox('Lado', ['P', 'S'])
        numero_cabine = coluna_cabine.number_input('Nº da cabine', 0, 2000, 300)
        tamanho_grupo = coluna_grupo.number_input('Pessoas no grupo', 1, 8, 1)

    # A previsão só roda quando o botão é clicado
    if st.button('Prever', icon=':material/rocket_launch:', type='primary', width='stretch'):
        # 1) Carregar do disco o modelo escolhido, a lista de colunas e o padronizador
        try:
            modelo = carregar(ARQUIVOS_MODELOS[nome_modelo])
            colunas_treino = carregar('colunas_treino.pkl')
            padronizador = carregar('padronizador.pkl')
        except FileNotFoundError as erro_arquivo:
            st.error(f'Arquivo não encontrado: {erro_arquivo.filename}. Faça o upload dos .pkl no Colab.')
            st.stop()

        # 2) Montar a linha do passageiro no formato do treino
        passageiro = dict(idade=idade, planeta=planeta, destino=destino, sono_criogenico=sono_criogenico, passageiro_vip=passageiro_vip, deck=deck,
                          lado=lado, numero_cabine=numero_cabine, tamanho_grupo=tamanho_grupo, **gastos)
        linha_passageiro = montar_linha(passageiro, colunas_treino)

        # 3) Padronizar com o MESMO StandardScaler do treino (os modelos foram treinados com dados padronizados)
        dados_padronizados = pd.DataFrame(padronizador.transform(linha_passageiro), columns=colunas_treino)

        # 4) Cada modelo usa as colunas com que foi treinado: o CatBoost usa todas (29),
        #    os outros usam só as selecionadas pela importância (26)
        colunas_modelo = list(modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else modelo.feature_names_)

        # 5) predict_proba devolve [chance de NÃO ser transportado, chance de SER transportado]; pegamos a segunda
        probabilidade = float(modelo.predict_proba(dados_padronizados[colunas_modelo])[0][1])

        # 6) Mostrar o resultado: 50% ou mais = transportado
        if probabilidade >= 0.5:
            st.markdown('<div class="resultado sim">Foi transportado!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="resultado nao">Não foi transportado</div>', unsafe_allow_html=True)
        # Totalizadores (st.metric)
        metrica_chance, metrica_gasto, metrica_modelo = st.columns(3)
        metrica_chance.metric('Chance de ser transportado', f'{probabilidade:.0%}')
        metrica_gasto.metric('Gasto total', f'{0 if sem_gasto else sum(gastos.values()):,}')
        metrica_modelo.metric('Modelo', nome_modelo.split(' (')[0])

# ===== Aba 2: gráficos =====
# Taxas de transporte observadas no treino (8.693 passageiros), calculadas na análise do relatório.
# São números fixos: assim o app não precisa do train.csv.
TAXAS_TRANSPORTE = {
    'Sono criogênico': {'Em CryoSleep': 81.8, 'Acordado': 32.9},
    'Planeta de origem': {'Europa': 65.9, 'Marte': 52.3, 'Terra': 42.4},
    'Idade': {'0 anos': 80.9, '1 a 12': 66.9, '13 a 17': 55.3, '18 a 25': 45.8,
              '26 a 40': 48.0, '41 a 60': 49.5, 'Mais de 60': 47.3},
    'Deck da cabine': {'B': 73.4, 'C': 68.0, 'G': 51.6, 'F': 44.0, 'D': 43.3, 'E': 35.7},
    'Tamanho do grupo': {'Sozinho': 45.2, '2 pessoas': 53.8, '3 pessoas': 59.3, '4 pessoas': 64.1},
}

with aba_graficos:
    fator_escolhido = st.selectbox('Ver a chance de ser transportado por:', list(TAXAS_TRANSPORTE))
    dados_grafico = pd.DataFrame({'Grupo': list(TAXAS_TRANSPORTE[fator_escolhido]), 'Transportados (%)': list(TAXAS_TRANSPORTE[fator_escolhido].values())})
    # sort=None mantém as barras na ordem em que escrevemos (senão o Altair ordena por letra)
    grafico_barras = alt.Chart(dados_grafico).mark_bar(color='#9d4edd').encode(
        x=alt.X('Grupo', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('Transportados (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['Grupo', 'Transportados (%)'])
    st.altair_chart(grafico_barras, width='stretch')
    st.caption('Taxa geral de transportados no treino: 50,4%. São taxas observadas e mostram associação, não causa.')

# ===== Aba 3: explicação =====
with aba_sobre:
    st.markdown("""
Este app usa modelos de Machine Learning treinados com o dataset **Spaceship Titanic** (Kaggle).

1. Você preenche os dados de um passageiro.
2. O app monta as mesmas variáveis usadas no treino (gasto total, gasto de luxo e básico, etc.).
3. O modelo escolhido calcula a **chance de o passageiro ter sido transportado**.

**O que os dados mostram:** quem estava em sono criogênico foi transportado em 82% dos casos
(contra 33% de quem estava acordado), e passageiros de Europa tiveram mais chance (66%) que os da Terra (42%).
""")
