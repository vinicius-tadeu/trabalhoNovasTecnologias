"""
TRABALHO FINAL - AT2 N2
NOVAS TECNOLOGIAS - GPE17N50289

Projeto:
Análise de vídeos em alta no YouTube

Autor:
Vinícius Tadeu Soares Silva

Ideia principal:
A coluna "views" do dataset é acumulada. Se o mesmo vídeo aparece em alta
em vários dias, não faz sentido somar a coluna views diretamente.

Por isso, este projeto cria a coluna "views_novas_estimadas", calculando a
diferença entre a medição atual e a medição anterior do mesmo vídeo.
"""

from pathlib import Path
import json

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="YouTube Trending Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análise de Vídeos em Alta no YouTube")


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    """Formata números grandes para aparecer no dashboard."""
    if valor is None or pd.isna(valor):
        return "0"

    valor = float(valor)

    if valor >= 1_000_000_000:
        return f"{valor / 1_000_000_000:.2f} bi"

    if valor >= 1_000_000:
        return f"{valor / 1_000_000:.2f} mi"

    if valor >= 1_000:
        return f"{valor / 1_000:.2f} mil"

    return f"{valor:.0f}"


def normalizar_colunas(df):
    """Padroniza nomes de colunas para facilitar o uso."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    df = df.rename(
        columns={
            "publish_time": "publish_date",
            "published_at": "publish_date",
            "comments": "comment_count",
            "comments_count": "comment_count",
            "country": "publish_country",
            "category": "category_id"
        }
    )

    return df


def inferir_pais_pelo_arquivo(arquivo):
    """
    Quando o dataset vem separado por país, o nome costuma ser:
    USvideos.csv, BRvideos.csv, GBvideos.csv etc.
    """
    nome = arquivo.stem.upper()

    if len(nome) >= 2 and nome[:2].isalpha():
        return nome[:2]

    return "Não informado"


def converter_data(coluna):
    """Converte datas normais e também o formato antigo yy.dd.mm."""
    data = pd.to_datetime(coluna, errors="coerce", utc=True)

    if data.isna().mean() > 0.5:
        data_alternativa = pd.to_datetime(
            coluna,
            format="%y.%d.%m",
            errors="coerce",
            utc=True
        )

        data = data.fillna(data_alternativa)

    return data.dt.tz_convert(None)


def corrigir_texto(texto):
    """
    Corrige erros comuns de encoding.

    Exemplo:
    'La RelÃ¨ve #2' vira 'La Relève #2'
    """
    if pd.isna(texto):
        return ""

    texto = str(texto)

    sinais_de_erro = ["Ã", "Â", "�"]

    if not any(sinal in texto for sinal in sinais_de_erro):
        return texto

    try:
        return texto.encode("latin1").decode("utf-8")
    except Exception:
        return texto


def dividir_sem_erro(numerador, denominador):
    """Evita divisão por zero."""
    return np.where(denominador > 0, numerador / denominador, 0)


def criar_id_video(df):
    """
    Cria um identificador para o vídeo.

    Se video_id estiver correto, usa video_id.
    Se vier com erro de planilha, como #NAME?, usa canal + título.
    """
    df = df.copy()

    id_alternativo = (
        df["channel_title"].astype(str).str.strip()
        + " - "
        + df["title"].astype(str).str.strip()
    )

    if "video_id" not in df.columns:
        df["id_video"] = id_alternativo
        return df

    id_original = df["video_id"].astype(str).str.strip()
    id_maiusculo = id_original.str.upper()

    id_invalido = (
        id_original.isna()
        | id_original.eq("")
        | id_maiusculo.isin(["#NAME?", "#NOME?", "#VALUE!", "#VALOR!", "NAN", "NONE", "NULL"])
        | id_maiusculo.str.startswith("#")
    )

    df["id_video"] = np.where(
        id_invalido,
        id_alternativo,
        id_original
    )

    return df


def mapa_categorias_padrao():
    """
    Mapa padrão das categorias mais comuns do YouTube.

    Ele é usado quando o dataset não traz os arquivos JSON de categoria.
    """
    return {
        "1": "Film & Animation",
        "2": "Autos & Vehicles",
        "10": "Music",
        "15": "Pets & Animals",
        "17": "Sports",
        "19": "Travel & Events",
        "20": "Gaming",
        "22": "People & Blogs",
        "23": "Comedy",
        "24": "Entertainment",
        "25": "News & Politics",
        "26": "Howto & Style",
        "27": "Education",
        "28": "Science & Technology",
        "29": "Nonprofits & Activism",
        "30": "Movies",
        "43": "Shows",
        "44": "Trailers"
    }


def carregar_mapa_categorias(pastas):
    """
    Tenta carregar nomes das categorias a partir dos arquivos JSON do dataset.

    Alguns datasets do YouTube Trending têm arquivos como:
    US_category_id.json, BR_category_id.json, GB_category_id.json etc.
    """
    mapa = {}

    for pasta in pastas:
        for arquivo_json in Path(pasta).glob("*.json"):
            try:
                with open(arquivo_json, "r", encoding="utf-8") as arquivo:
                    conteudo = json.load(arquivo)

                for item in conteudo.get("items", []):
                    id_categoria = str(item.get("id"))
                    nome_categoria = item.get("snippet", {}).get("title")

                    if id_categoria and nome_categoria:
                        mapa[id_categoria] = corrigir_texto(nome_categoria)

            except Exception:
                pass

    if not mapa:
        mapa = mapa_categorias_padrao()

    return mapa


def adicionar_nome_categoria(df, mapa_categorias):
    """Cria a coluna category_name a partir da coluna category_id."""
    df = df.copy()

    if "category_name" in df.columns:
        df["category_name"] = df["category_name"].apply(corrigir_texto)
        return df

    df["category_id"] = df["category_id"].astype(str)

    df["category_name"] = df["category_id"].map(mapa_categorias)

    df["category_name"] = df["category_name"].fillna(
        "Categoria " + df["category_id"]
    )

    return df


def ajustar_escala_para_grafico(serie):
    """
    Evita aparecer 1e9, 1e10 ou números confusos no gráfico.
    O gráfico mostra o número já convertido em mil, milhões ou bilhões.
    """
    if serie.empty:
        return serie, "views"

    maior_valor = serie.max()

    if pd.isna(maior_valor):
        return serie, "views"

    if maior_valor >= 1_000_000_000:
        return serie / 1_000_000_000, "bilhões"

    if maior_valor >= 1_000_000:
        return serie / 1_000_000, "milhões"

    if maior_valor >= 1_000:
        return serie / 1_000, "milhares"

    return serie, "views"


def grafico_barras(serie, titulo):
    """Cria gráfico de barras simples e com eixo fácil de entender."""
    serie_plot, unidade = ajustar_escala_para_grafico(serie)

    fig, ax = plt.subplots(figsize=(10, 5))

    serie_plot.plot(kind="bar", ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel(f"Views novas estimadas ({unidade})")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        f"Neste gráfico, 1.0 significa 1.0 {unidade}. "
        f"Se a escala for milhões, 1.0 = 1 milhão. "
        f"Se for bilhões, 1.0 = 1 bilhão."
    )


def grafico_barras_horizontal(serie, titulo):
    """Cria gráfico horizontal para rankings de views."""
    serie_plot, unidade = ajustar_escala_para_grafico(serie)

    fig, ax = plt.subplots(figsize=(10, 5))

    serie_plot.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel(f"Views novas estimadas ({unidade})")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        f"Neste gráfico, 1.0 significa 1.0 {unidade}. "
        f"Se a escala for milhões, 1.0 = 1 milhão. "
        f"Se for bilhões, 1.0 = 1 bilhão."
    )


def grafico_importancia_modelo(serie, titulo):
    """Cria gráfico específico para importância das variáveis do modelo."""
    fig, ax = plt.subplots(figsize=(10, 5))

    serie.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel("Importância da variável")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Quanto maior a barra, mais essa variável ajudou o modelo a explicar as views novas estimadas."
    )




def grafico_percentual_horizontal(serie, titulo):
    """Cria gráfico horizontal para percentuais, como taxa de engajamento."""
    fig, ax = plt.subplots(figsize=(10, 5))

    serie.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel("Engajamento estimado (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "O engajamento foi calculado como: "
        "(likes novos + comentários novos) / views novas estimadas."
    )


def calcular_engajamento_geral(df):
    """
    Calcula o engajamento geral usando soma de likes e comentários novos
    dividida pela soma de views novas.
    """
    total_interacoes = (
        df["likes_novos_estimados"].sum()
        + df["comentarios_novos_estimados"].sum()
    )

    total_views_novas = df["views_novas_estimadas"].sum()

    if total_views_novas <= 0:
        return 0

    return (total_interacoes / total_views_novas) * 100


# =====================================================
# CARREGAMENTO DOS DADOS
# =====================================================

@st.cache_data(show_spinner="Carregando dados...")
def carregar_dados():
    """
    Primeiro o app procura arquivos CSV locais.
    Se não encontrar, tenta baixar pelo KaggleHub.
    """
    pastas_dados = [Path("data"), Path(".")]

    arquivos_csv = []

    for pasta in pastas_dados:
        arquivos_csv.extend(list(pasta.glob("*.csv")))

    if not arquivos_csv:
        caminho_kaggle = Path(
            kagglehub.dataset_download("thedevastator/youtube-trending-videos-dataset")
        )

        pastas_dados.append(caminho_kaggle)
        arquivos_csv = list(caminho_kaggle.glob("*.csv"))

    mapa_categorias = carregar_mapa_categorias(pastas_dados)

    bases = []

    for arquivo in arquivos_csv:
        try:
            # latin1 ajuda a evitar que a leitura quebre em alguns arquivos antigos
            df_temp = pd.read_csv(arquivo, low_memory=False, encoding="latin1")
            df_temp = normalizar_colunas(df_temp)

            if "views" not in df_temp.columns:
                continue

            if "publish_country" not in df_temp.columns:
                df_temp["publish_country"] = inferir_pais_pelo_arquivo(arquivo)

            df_temp = adicionar_nome_categoria(df_temp, mapa_categorias)

            bases.append(df_temp)

        except Exception:
            try:
                df_temp = pd.read_csv(arquivo, low_memory=False)
                df_temp = normalizar_colunas(df_temp)

                if "views" not in df_temp.columns:
                    continue

                if "publish_country" not in df_temp.columns:
                    df_temp["publish_country"] = inferir_pais_pelo_arquivo(arquivo)

                df_temp = adicionar_nome_categoria(df_temp, mapa_categorias)

                bases.append(df_temp)

            except Exception:
                pass

    if not bases:
        st.error("Nenhum arquivo CSV válido foi encontrado.")
        st.stop()

    return pd.concat(bases, ignore_index=True)


# =====================================================
# TRATAMENTO DOS DADOS
# =====================================================

@st.cache_data(show_spinner="Tratando dados...")
def tratar_dados(df):
    df = df.copy()
    df = normalizar_colunas(df)

    # Remove registros duplicados
    df = df.drop_duplicates()

    # Garante colunas numéricas obrigatórias
    colunas_numericas = ["views", "likes", "dislikes", "comment_count"]

    for coluna in colunas_numericas:
        if coluna not in df.columns:
            df[coluna] = 0

        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0)

    # Remove registros sem views
    df = df[df["views"] > 0]

    # Datas
    if "publish_date" in df.columns:
        df["publish_date"] = converter_data(df["publish_date"])
    else:
        df["publish_date"] = pd.NaT

    if "trending_date" in df.columns:
        df["trending_date"] = converter_data(df["trending_date"])
    else:
        df["trending_date"] = pd.NaT

    # Colunas de texto e categorias
    if "publish_country" not in df.columns:
        df["publish_country"] = "Não informado"

    if "channel_title" not in df.columns:
        df["channel_title"] = "Canal não informado"

    if "category_id" not in df.columns:
        df["category_id"] = "Não informado"

    if "category_name" not in df.columns:
        df["category_name"] = "Categoria " + df["category_id"].astype(str)

    if "title" not in df.columns:
        df["title"] = ""

    df["publish_country"] = df["publish_country"].fillna("Não informado")
    df["channel_title"] = df["channel_title"].fillna("Canal não informado").apply(corrigir_texto)
    df["category_id"] = df["category_id"].astype(str).fillna("Não informado")
    df["category_name"] = df["category_name"].fillna("Categoria " + df["category_id"]).apply(corrigir_texto)
    df["title"] = df["title"].fillna("").apply(corrigir_texto)

    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("").apply(corrigir_texto)

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").apply(corrigir_texto)

    # Remove vídeos removidos ou com erro
    if "video_error_or_removed" in df.columns:
        removido = (
            df["video_error_or_removed"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "sim"])
        )

        df = df[~removido]

    # Cria identificador do vídeo, corrigindo #NAME? e outros erros de planilha
    df = criar_id_video(df)

    # Remove duplicidade do mesmo vídeo no mesmo país e no mesmo dia
    df = df.drop_duplicates(
        subset=["id_video", "publish_country", "trending_date"],
        keep="last"
    )

    # Ordena para calcular diferença entre as aparições
    df = df.sort_values(["id_video", "publish_country", "trending_date"])

    grupo = df.groupby(["id_video", "publish_country"], dropna=False)

    # =====================================================
    # PONTO MAIS IMPORTANTE DO TRABALHO
    # =====================================================
    # views = total acumulado no momento da coleta
    # views_novas_estimadas = diferença entre uma coleta e a anterior

    df["views_novas_estimadas"] = grupo["views"].diff()
    df["likes_novos_estimados"] = grupo["likes"].diff()
    df["comentarios_novos_estimados"] = grupo["comment_count"].diff()

    # A primeira aparição não tem dia anterior para comparar, então fica 0
    df["views_novas_estimadas"] = df["views_novas_estimadas"].clip(lower=0).fillna(0)
    df["likes_novos_estimados"] = df["likes_novos_estimados"].clip(lower=0).fillna(0)
    df["comentarios_novos_estimados"] = df["comentarios_novos_estimados"].clip(lower=0).fillna(0)

    # Novas métricas
    df["engajamento_estimado"] = dividir_sem_erro(
        df["likes_novos_estimados"] + df["comentarios_novos_estimados"],
        df["views_novas_estimadas"]
    )

    df["dias_ate_trending"] = (
        df["trending_date"] - df["publish_date"]
    ).dt.days.clip(lower=0)

    df["tamanho_titulo"] = df["title"].str.len()

    if "tags" in df.columns:
        df["quantidade_tags"] = df["tags"].apply(
            lambda x: 0 if x in ["", "[none]"] else len(str(x).split("|"))
        )
    else:
        df["quantidade_tags"] = 0

    dias = {
        0: "Segunda",
        1: "Terça",
        2: "Quarta",
        3: "Quinta",
        4: "Sexta",
        5: "Sábado",
        6: "Domingo"
    }

    df["dia_em_alta"] = df["trending_date"].dt.dayofweek.map(dias)

    df = df.replace([np.inf, -np.inf], 0)

    return df


# =====================================================
# MODELO SIMPLES COM SCIKIT-LEARN
# =====================================================

def treinar_modelo(df):
    """
    Modelo simples para demonstrar uso de Scikit-learn.
    O objetivo é estimar views novas com base em algumas variáveis numéricas.
    """
    df_modelo = df[df["views_novas_estimadas"] > 0].copy()

    colunas_modelo = [
        "likes_novos_estimados",
        "comentarios_novos_estimados",
        "engajamento_estimado",
        "dias_ate_trending",
        "tamanho_titulo",
        "quantidade_tags"
    ]

    df_modelo = df_modelo[colunas_modelo + ["views_novas_estimadas"]].dropna()

    if len(df_modelo) < 200:
        return None

    X = df_modelo[colunas_modelo]
    y = np.log1p(df_modelo["views_novas_estimadas"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    modelo = RandomForestRegressor(
        n_estimators=80,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    mae = mean_absolute_error(
        np.expm1(y_test),
        np.expm1(previsoes)
    )

    r2 = r2_score(y_test, previsoes)

    importancias = pd.DataFrame(
        {
            "variavel": colunas_modelo,
            "importancia": modelo.feature_importances_
        }
    ).sort_values("importancia", ascending=False)

    return mae, r2, importancias


# =====================================================
# EXECUÇÃO DO APP
# =====================================================

df_original = carregar_dados()
df = tratar_dados(df_original)

st.markdown(
    """
    Este dashboard analisa vídeos em alta no YouTube.

    A pergunta principal é:

    **Quais fatores estão relacionados ao crescimento de visualizações dos vídeos
    enquanto eles aparecem em alta?**
    """
)

st.info(
    """
    A coluna original `views` é acumulada. Por isso, o projeto usa a métrica
    `views_novas_estimadas`, que calcula a diferença de views entre uma aparição
    e outra do mesmo vídeo.
    """
)


# =====================================================
# FILTROS
# =====================================================

st.sidebar.header("Filtros")

paises = sorted(df["publish_country"].dropna().unique())

pais = st.sidebar.selectbox(
    "País",
    ["Todos"] + paises
)

categorias = sorted(df["category_name"].dropna().astype(str).unique())

categoria = st.sidebar.selectbox(
    "Categoria",
    ["Todas"] + categorias
)

df_filtrado = df.copy()

if pais != "Todos":
    df_filtrado = df_filtrado[df_filtrado["publish_country"] == pais]

if categoria != "Todas":
    df_filtrado = df_filtrado[df_filtrado["category_name"].astype(str) == categoria]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()


# =====================================================
# KPIS
# =====================================================

registros = len(df_filtrado)
videos_unicos = df_filtrado["id_video"].nunique()
views_novas = df_filtrado["views_novas_estimadas"].sum()

df_com_crescimento = df_filtrado[df_filtrado["views_novas_estimadas"] > 0]

if df_com_crescimento.empty:
    mediana_views_novas = 0
else:
    mediana_views_novas = df_com_crescimento["views_novas_estimadas"].median()

engajamento_medio = calcular_engajamento_geral(df_filtrado)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Registros", formatar_numero(registros))
col2.metric("Vídeos únicos", formatar_numero(videos_unicos))
col3.metric("Views novas estimadas", formatar_numero(views_novas))
col4.metric("Mediana de views novas", formatar_numero(mediana_views_novas))
col5.metric("Engajamento médio", f"{engajamento_medio:.2f}%")


# =====================================================
# ABAS
# =====================================================

aba1, aba2, aba3, aba4 = st.tabs(
    [
        "Resumo",
        "Gráficos",
        "Modelo",
        "Dados"
    ]
)


# =====================================================
# ABA RESUMO
# =====================================================

with aba1:
    st.subheader("Resumo do tratamento")

    views_acumuladas_somadas = df_filtrado["views"].sum()

    df_ultima_aparicao = (
        df_filtrado
        .sort_values(["id_video", "publish_country", "trending_date"])
        .drop_duplicates(subset=["id_video", "publish_country"], keep="last")
    )

    views_acumuladas_videos_unicos = df_ultima_aparicao["views"].sum()

    st.write(
        f"""
        - Registros na base original: **{formatar_numero(len(df_original))}**
        - Registros após tratamento: **{formatar_numero(len(df))}**
        - Registros após filtros: **{formatar_numero(len(df_filtrado))}**
        - Vídeos únicos após filtros: **{formatar_numero(videos_unicos)}**
        - Soma direta de views acumuladas: **{formatar_numero(views_acumuladas_somadas)}**
        - Views acumuladas considerando cada vídeo uma vez por país: **{formatar_numero(views_acumuladas_videos_unicos)}**
        - Views novas estimadas entre aparições: **{formatar_numero(views_novas)}**
        """
    )

    st.success(
        """
        A métrica mais adequada para evitar distorção é "views novas estimadas",
        porque ela não soma repetidamente o total acumulado do mesmo vídeo.
        """
    )


# =====================================================
# ABA GRÁFICOS
# =====================================================

with aba2:
    st.subheader("1. Crescimento estimado por dia em alta")

    ordem_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

    views_por_dia = (
        df_filtrado
        .groupby("dia_em_alta")["views_novas_estimadas"]
        .sum()
        .reindex(ordem_dias)
        .fillna(0)
    )

    grafico_barras(
        views_por_dia,
        "Views novas estimadas por dia da semana"
    )

    st.write(
        """
        Este gráfico mostra em quais dias da semana os vídeos tiveram maior crescimento
        de visualizações enquanto estavam em alta.
        """
    )

    st.subheader("2. Top 10 categorias por crescimento de views")

    top_categorias = (
        df_filtrado
        .groupby("category_name")["views_novas_estimadas"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    grafico_barras_horizontal(
        top_categorias,
        "Categorias com maior crescimento estimado de views"
    )

    st.subheader("3. Top 10 canais por crescimento de views")

    top_canais = (
        df_filtrado
        .groupby("channel_title")["views_novas_estimadas"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    grafico_barras_horizontal(
        top_canais,
        "Canais com maior crescimento estimado de views"
    )

    st.subheader("4. Engajamento médio por categoria")

    st.write(
        """
        Este gráfico é mais fácil de interpretar do que a relação direta entre views e engajamento.
        Ele mostra, por categoria, qual foi a proporção média de interações em relação às
        views novas estimadas.
        """
    )

    base_engajamento = df_filtrado[df_filtrado["views_novas_estimadas"] > 0].copy()

    engajamento_categoria = (
        base_engajamento
        .groupby("category_name")
        .agg(
            views_novas=("views_novas_estimadas", "sum"),
            likes_novos=("likes_novos_estimados", "sum"),
            comentarios_novos=("comentarios_novos_estimados", "sum"),
            quantidade_registros=("views_novas_estimadas", "count")
        )
        .reset_index()
    )

    # Evita categorias com poucos dados, porque elas podem gerar percentuais muito distorcidos
    engajamento_categoria = engajamento_categoria[
        (engajamento_categoria["views_novas"] >= 1_000_000)
        & (engajamento_categoria["quantidade_registros"] >= 20)
    ]

    if engajamento_categoria.empty:
        st.warning("Não há dados suficientes para calcular engajamento por categoria com segurança.")

    else:
        engajamento_categoria["engajamento_percentual"] = (
            (
                engajamento_categoria["likes_novos"]
                + engajamento_categoria["comentarios_novos"]
            )
            / engajamento_categoria["views_novas"]
        ) * 100

        top_engajamento = (
            engajamento_categoria
            .set_index("category_name")["engajamento_percentual"]
            .sort_values(ascending=False)
            .head(10)
        )

        grafico_percentual_horizontal(
            top_engajamento,
            "Categorias com maior engajamento médio estimado"
        )


# =====================================================
# ABA MODELO
# =====================================================

with aba3:
    st.subheader("Modelo simples com Scikit-learn")

    st.write(
        """
        Foi criado um modelo simples de Random Forest para estimar as views novas
        dos vídeos com base em likes novos, comentários novos, engajamento,
        tempo até entrar em alta, tamanho do título e quantidade de tags.
        """
    )

    resultado = treinar_modelo(df_filtrado)

    if resultado is None:
        st.warning("Não há dados suficientes para treinar o modelo com os filtros atuais.")

    else:
        mae, r2, importancias = resultado

        c1, c2 = st.columns(2)

        c1.metric("Erro médio absoluto", formatar_numero(mae))
        c2.metric("R² em escala log", f"{r2:.3f}")

        st.write(
            """
            O modelo não deve ser entendido como uma previsão perfeita de viralização.
            O R² não é taxa de acerto; ele indica quanto da variação dos dados o modelo
            conseguiu explicar.
            """
        )

        grafico_importancia_modelo(
            importancias.set_index("variavel")["importancia"],
            "Importância das variáveis no modelo"
        )


# =====================================================
# ABA DADOS
# =====================================================

with aba4:
    st.subheader("Amostra dos dados tratados")

    colunas = [
        "title",
        "publish_country",
        "channel_title",
        "category_id",
        "category_name",
        "trending_date",
        "views",
        "views_novas_estimadas",
        "likes",
        "likes_novos_estimados",
        "comment_count",
        "comentarios_novos_estimados",
        "engajamento_estimado",
        "dia_em_alta"
    ]

    colunas = [coluna for coluna in colunas if coluna in df_filtrado.columns]

    st.dataframe(
        df_filtrado[colunas].head(100),
        use_container_width=True
    )

    st.subheader("Exemplo da diferença entre views acumuladas e views novas")

    exemplo = (
        df_filtrado
        .sort_values(["id_video", "publish_country", "trending_date"])
        [
            [
                "title",
                "publish_country",
                "category_name",
                "trending_date",
                "views",
                "views_novas_estimadas"
            ]
        ]
        .head(30)
    )

    st.dataframe(exemplo, use_container_width=True)
