"""
TRABALHO FINAL - AT2 N2
NOVAS TECNOLOGIAS - GPE17N50289

Tema:
Análise de Dados e Solução de Problemas com Python

Projeto:
Análise de vídeos em alta no YouTube

Autor:
Vinícius Tadeu Soares Silva
"""

# =====================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =====================================================

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="YouTube Trending Analytics",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# CONSTANTES
# =====================================================

DATASET_KAGGLE = "thedevastator/youtube-trending-videos-dataset"

COLUNAS_NUMERICAS_BASE = [
    "views",
    "likes",
    "dislikes",
    "comment_count"
]

ORDEM_DIAS = [
    "Segunda-feira",
    "Terça-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sábado",
    "Domingo"
]


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    """Formata números grandes para leitura em tela."""
    if pd.isna(valor):
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
    """Padroniza os nomes das colunas."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    mapa = {
        "publish_time": "publish_date",
        "published_at": "publish_date",
        "comments": "comment_count",
        "comments_count": "comment_count",
        "country": "publish_country",
        "category": "category_id"
    }

    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})

    return df


def inferir_pais_pelo_nome_arquivo(caminho):
    """
    Alguns datasets do YouTube vêm separados por país:
    USvideos.csv, BRvideos.csv, GBvideos.csv etc.
    Essa função tenta inferir o país pelo nome do arquivo.
    """
    nome = caminho.stem.upper()

    if len(nome) >= 2:
        possivel_pais = nome[:2]

        if possivel_pais.isalpha():
            return possivel_pais

    return "Não informado"


def parse_data(serie):
    """Converte datas em diferentes formatos possíveis."""
    data = pd.to_datetime(serie, errors="coerce", utc=True)

    # Formato comum em alguns datasets antigos do YouTube: 17.14.11
    if data.isna().mean() > 0.5:
        data_alt = pd.to_datetime(
            serie,
            format="%y.%d.%m",
            errors="coerce",
            utc=True
        )

        data = data.fillna(data_alt)

    return data.dt.tz_convert(None)


@st.cache_data(show_spinner="Carregando base de dados...")
def carregar_dados():
    """
    Carrega o dataset.

    Para a entrega no AVA, recomenda-se anexar também os arquivos CSV
    dentro de uma pasta chamada 'data'.

    Caso os arquivos não estejam localmente, o código tenta baixar via KaggleHub.
    """

    arquivos_locais = list(Path("data").glob("*.csv")) + list(Path(".").glob("*.csv"))

    if not arquivos_locais:
        caminho_kaggle = Path(
            kagglehub.dataset_download(DATASET_KAGGLE)
        )

        arquivos_locais = list(caminho_kaggle.glob("*.csv"))

    bases = []

    for arquivo in arquivos_locais:
        try:
            amostra = pd.read_csv(arquivo, nrows=5, low_memory=False)
            amostra = normalizar_colunas(amostra)

            if "views" not in amostra.columns:
                continue

            df_temp = pd.read_csv(arquivo, low_memory=False)
            df_temp = normalizar_colunas(df_temp)

            if "publish_country" not in df_temp.columns:
                df_temp["publish_country"] = inferir_pais_pelo_nome_arquivo(arquivo)

            df_temp["arquivo_origem"] = arquivo.name
            bases.append(df_temp)

        except Exception:
            continue

    if not bases:
        st.error(
            "Nenhum CSV válido foi encontrado. "
            "Verifique se o arquivo do dataset está na pasta data/."
        )
        st.stop()

    df = pd.concat(bases, ignore_index=True)

    return df


@st.cache_data(show_spinner="Tratando e preparando os dados...")
def preparar_dados(df):
    """Executa limpeza, tratamento e engenharia de atributos."""

    df = df.copy()
    df = normalizar_colunas(df)

    # Remove duplicatas completas
    df = df.drop_duplicates()

    # Remove duplicatas mais relevantes, caso existam IDs de vídeo
    if {"video_id", "trending_date"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["video_id", "trending_date"])

    # Garante colunas numéricas
    for coluna in COLUNAS_NUMERICAS_BASE:
        if coluna not in df.columns:
            df[coluna] = 0

        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0)

    # Views precisa ser maior que zero para cálculo de taxas
    df = df[df["views"] > 0]

    # Conversão de datas
    if "publish_date" in df.columns:
        df["publish_date"] = parse_data(df["publish_date"])
    else:
        df["publish_date"] = pd.NaT

    if "trending_date" in df.columns:
        df["trending_date"] = parse_data(df["trending_date"])
    else:
        df["trending_date"] = pd.NaT

    # Remove vídeos indisponíveis, caso a coluna exista
    if "video_error_or_removed" in df.columns:
        removido = (
            df["video_error_or_removed"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "sim"])
        )

        df = df[~removido]

    # Preenchimento de campos categóricos
    if "publish_country" not in df.columns:
        df["publish_country"] = "Não informado"

    df["publish_country"] = df["publish_country"].fillna("Não informado")

    if "channel_title" not in df.columns:
        df["channel_title"] = "Canal não informado"

    df["channel_title"] = df["channel_title"].fillna("Canal não informado")

    if "category_id" not in df.columns:
        df["category_id"] = "Não informado"

    df["category_id"] = df["category_id"].astype(str).fillna("Não informado")

    # =====================================================
    # ENGENHARIA DE ATRIBUTOS
    # =====================================================

    df["like_rate"] = df["likes"] / df["views"]
    df["comment_rate"] = df["comment_count"] / df["views"]

    df["engagement_rate"] = (
        df["likes"] + df["dislikes"] + df["comment_count"]
    ) / df["views"]

    df["log_views"] = np.log1p(df["views"])
    df["log_likes"] = np.log1p(df["likes"])
    df["log_comments"] = np.log1p(df["comment_count"])

    df["publish_hour"] = df["publish_date"].dt.hour

    df["dias_ate_trending"] = (
        df["trending_date"] - df["publish_date"]
    ).dt.days

    df["dias_ate_trending"] = df["dias_ate_trending"].clip(lower=0)

    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
        df["title_length"] = df["title"].str.len()
    else:
        df["title_length"] = 0

    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("")
        df["tag_count"] = df["tags"].apply(
            lambda texto: 0 if texto in ["", "[none]"] else len(str(texto).split("|"))
        )
    else:
        df["tag_count"] = 0

    # Dia da semana traduzido
    dias_map = {
        0: "Segunda-feira",
        1: "Terça-feira",
        2: "Quarta-feira",
        3: "Quinta-feira",
        4: "Sexta-feira",
        5: "Sábado",
        6: "Domingo"
    }

    df["dia_semana_publicacao"] = df["publish_date"].dt.dayofweek.map(dias_map)

    # Remove infinitos gerados por divisões
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove registros sem informação mínima para análise
    df = df.dropna(subset=["views", "likes", "comment_count"])

    return df


@st.cache_data(show_spinner="Treinando modelo preditivo...")
def treinar_modelo(df):
    """
    Treina um modelo simples para estimar visualizações.

    O objetivo aqui é demonstrar aplicação de Scikit-learn.
    O modelo tem caráter analítico/descritivo, não deve ser interpretado
    como previsão perfeita de viralização.
    """

    df_modelo = df.copy()

    colunas_numericas = [
        "likes",
        "dislikes",
        "comment_count",
        "like_rate",
        "comment_rate",
        "engagement_rate",
        "publish_hour",
        "dias_ate_trending",
        "title_length",
        "tag_count"
    ]

    colunas_categoricas = [
        "publish_country",
        "category_id",
        "dia_semana_publicacao"
    ]

    colunas_numericas = [c for c in colunas_numericas if c in df_modelo.columns]
    colunas_categoricas = [c for c in colunas_categoricas if c in df_modelo.columns]

    colunas_modelo = colunas_numericas + colunas_categoricas + ["views"]

    df_modelo = df_modelo[colunas_modelo].dropna(subset=["views"])

    if len(df_modelo) < 200:
        return None

    X = df_modelo.drop(columns=["views"])
    y = np.log1p(df_modelo["views"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre_processamento = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median"))
                    ]
                ),
                colunas_numericas
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder)
                    ]
                ),
                colunas_categoricas
            )
        ]
    )

    modelo = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", pre_processamento),
            ("modelo", modelo)
        ]
    )

    pipeline.fit(X_train, y_train)

    previsoes_log = pipeline.predict(X_test)

    y_test_original = np.expm1(y_test)
    previsoes_original = np.expm1(previsoes_log)

    mae = mean_absolute_error(y_test_original, previsoes_original)
    r2 = r2_score(y_test, previsoes_log)

    # Importância das variáveis
    nomes_features = list(colunas_numericas)

    if colunas_categoricas:
        encoder_treinado = (
            pipeline
            .named_steps["preprocessamento"]
            .named_transformers_["cat"]
            .named_steps["encoder"]
        )

        nomes_cat = encoder_treinado.get_feature_names_out(colunas_categoricas)
        nomes_features.extend(nomes_cat)

    importancias = pipeline.named_steps["modelo"].feature_importances_

    if len(nomes_features) != len(importancias):
        nomes_features = [f"feature_{i}" for i in range(len(importancias))]

    df_importancias = pd.DataFrame(
        {
            "feature": nomes_features,
            "importancia": importancias
        }
    ).sort_values("importancia", ascending=False)

    return {
        "modelo": pipeline,
        "mae": mae,
        "r2": r2,
        "importancias": df_importancias,
        "linhas_treino": len(X_train),
        "linhas_teste": len(X_test)
    }


def plotar_barra_horizontal(serie, titulo, xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))

    serie.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)


def plotar_histograma_views(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["log_views"].dropna(), bins=35)

    ax.set_title("Distribuição das Visualizações em Escala Logarítmica")
    ax.set_xlabel("Log das visualizações")
    ax.set_ylabel("Quantidade de vídeos")
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)


def plotar_dispersao_engajamento(df):
    amostra = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        amostra["log_views"],
        amostra["engagement_rate"] * 100,
        alpha=0.35
    )

    ax.set_title("Relação entre Visualizações e Taxa de Engajamento")
    ax.set_xlabel("Log das visualizações")
    ax.set_ylabel("Taxa de engajamento (%)")
    ax.grid(alpha=0.3)

    st.pyplot(fig)


def plotar_correlacao(df):
    colunas = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "like_rate",
        "comment_rate",
        "engagement_rate",
        "dias_ate_trending",
        "title_length",
        "tag_count"
    ]

    colunas = [c for c in colunas if c in df.columns]

    corr = df[colunas].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    imagem = ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    ax.set_title("Matriz de Correlação entre Variáveis Numéricas")

    fig.colorbar(imagem, ax=ax)

    st.pyplot(fig)


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

df_raw = carregar_dados()
df = preparar_dados(df_raw)

st.title("📊 YouTube Trending Analytics")

st.markdown(
    """
    Este projeto analisa vídeos que entraram em alta no YouTube com o objetivo de
    entender quais fatores estão associados ao alcance e ao engajamento dos conteúdos.

    **Pergunta de negócio:**  
    Quais características estão mais relacionadas ao sucesso de vídeos em alta no YouTube
    e como os dados podem ajudar criadores, marcas e equipes de marketing a entenderem
    padrões de audiência?
    """
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Filtros")

paises = sorted(df["publish_country"].dropna().unique())

paises_selecionados = st.sidebar.multiselect(
    "País de publicação",
    options=paises,
    default=paises
)

categorias = sorted(df["category_id"].dropna().astype(str).unique())

categorias_selecionadas = st.sidebar.multiselect(
    "Categoria",
    options=categorias,
    default=categorias
)

min_views = int(df["views"].quantile(0.01))
max_views = int(df["views"].quantile(0.99))

faixa_views = st.sidebar.slider(
    "Faixa de visualizações",
    min_value=min_views,
    max_value=max_views,
    value=(min_views, max_views)
)

df_filtrado = df.copy()

if paises_selecionados:
    df_filtrado = df_filtrado[
        df_filtrado["publish_country"].isin(paises_selecionados)
    ]

if categorias_selecionadas:
    df_filtrado = df_filtrado[
        df_filtrado["category_id"].astype(str).isin(categorias_selecionadas)
    ]

df_filtrado = df_filtrado[
    (df_filtrado["views"] >= faixa_views[0]) &
    (df_filtrado["views"] <= faixa_views[1])
]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()


# =====================================================
# KPIs
# =====================================================

total_videos = len(df_filtrado)
total_views = df_filtrado["views"].sum()
mediana_views = df_filtrado["views"].median()
media_likes = df_filtrado["likes"].mean()
engajamento_medio = df_filtrado["engagement_rate"].mean() * 100
dias_trending_mediana = df_filtrado["dias_ate_trending"].median()

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Vídeos analisados", formatar_numero(total_videos))
c2.metric("Views totais", formatar_numero(total_views))
c3.metric("Mediana de views", formatar_numero(mediana_views))
c4.metric("Likes médios", formatar_numero(media_likes))
c5.metric("Engajamento médio", f"{engajamento_medio:.2f}%")


# =====================================================
# ABAS
# =====================================================

aba1, aba2, aba3, aba4, aba5 = st.tabs(
    [
        "Visão Geral",
        "Exploração Visual",
        "Modelo Preditivo",
        "Dados Tratados",
        "Conclusão"
    ]
)


# =====================================================
# ABA 1 - VISÃO GERAL
# =====================================================

with aba1:
    st.subheader("Resumo da base tratada")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            **Quantidade inicial de registros:** {formatar_numero(len(df_raw))}  
            **Quantidade após tratamento:** {formatar_numero(len(df))}  
            **Quantidade após filtros:** {formatar_numero(len(df_filtrado))}  
            **Países analisados:** {df_filtrado["publish_country"].nunique()}  
            **Canais analisados:** {df_filtrado["channel_title"].nunique()}  
            """
        )

    with col2:
        st.markdown(
            f"""
            **Média de views:** {formatar_numero(df_filtrado["views"].mean())}  
            **Mediana de views:** {formatar_numero(df_filtrado["views"].median())}  
            **Maior vídeo da amostra:** {formatar_numero(df_filtrado["views"].max())} views  
            **Tempo mediano até entrar em alta:** {dias_trending_mediana:.1f} dias  
            """
        )

    st.info(
        """
        A mediana foi usada em conjunto com a média porque dados de audiência costumam
        ter valores extremos. Poucos vídeos muito grandes podem distorcer a média.
        """
    )


# =====================================================
# ABA 2 - EXPLORAÇÃO VISUAL
# =====================================================

with aba2:
    st.subheader("Distribuição das visualizações")

    st.write(
        """
        O gráfico abaixo usa escala logarítmica para facilitar a leitura,
        pois vídeos em plataformas digitais costumam apresentar distribuição desigual:
        muitos vídeos possuem alcance moderado e poucos concentram uma audiência muito alta.
        """
    )

    plotar_histograma_views(df_filtrado)

    st.subheader("Top 10 países por quantidade de vídeos em alta")

    top_paises = (
        df_filtrado["publish_country"]
        .value_counts()
        .head(10)
    )

    plotar_barra_horizontal(
        top_paises,
        "Países com Mais Vídeos em Alta",
        "Quantidade de vídeos"
    )

    st.subheader("Publicações por dia da semana")

    dias = (
        df_filtrado["dia_semana_publicacao"]
        .value_counts()
        .reindex(ORDEM_DIAS)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    dias.plot(kind="bar", ax=ax)

    ax.set_title("Quantidade de Publicações por Dia da Semana")
    ax.set_xlabel("Dia da semana")
    ax.set_ylabel("Quantidade de vídeos")
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

    st.subheader("Visualizações x Engajamento")

    st.write(
        """
        Este gráfico ajuda a verificar se vídeos com muitas visualizações também possuem
        alto engajamento proporcional. Isso é importante porque um vídeo pode ter muitas
        views, mas uma taxa baixa de interação.
        """
    )

    plotar_dispersao_engajamento(df_filtrado)

    st.subheader("Top 10 canais por mediana de visualizações")

    top_canais = (
        df_filtrado
        .groupby("channel_title")
        .agg(
            mediana_views=("views", "median"),
            quantidade_videos=("views", "count")
        )
        .query("quantidade_videos >= 3")
        .sort_values("mediana_views", ascending=False)
        .head(10)
    )

    if not top_canais.empty:
        plotar_barra_horizontal(
            top_canais["mediana_views"],
            "Canais com Maior Mediana de Views",
            "Mediana de visualizações"
        )
    else:
        st.warning(
            "Não há canais com pelo menos 3 vídeos para gerar este ranking com segurança."
        )

    st.subheader("Correlação entre variáveis")

    st.write(
        """
        A matriz de correlação mostra quais variáveis numéricas se movimentam de forma parecida.
        Por exemplo, é esperado que likes e comentários tenham relação positiva com visualizações.
        """
    )

    plotar_correlacao(df_filtrado)


# =====================================================
# ABA 3 - MODELO PREDITIVO
# =====================================================

with aba3:
    st.subheader("Modelo preditivo com Scikit-learn")

    st.markdown(
        """
        Foi treinado um modelo de **Random Forest Regressor** para estimar o volume de
        visualizações dos vídeos com base em variáveis como likes, comentários,
        taxa de engajamento, país, categoria, dia da semana e características do título.

        O objetivo do modelo é demonstrar o uso de Scikit-learn dentro do ciclo de análise.
        Ele não deve ser interpretado como uma previsão perfeita de viralização.
        """
    )

    resultado_modelo = treinar_modelo(df_filtrado)

    if resultado_modelo is None:
        st.warning(
            "A quantidade de dados filtrados é insuficiente para treinar o modelo com segurança. "
            "Remova alguns filtros e tente novamente."
        )

    else:
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Linhas de treino", formatar_numero(resultado_modelo["linhas_treino"]))
        m2.metric("Linhas de teste", formatar_numero(resultado_modelo["linhas_teste"]))
        m3.metric("Erro médio absoluto", formatar_numero(resultado_modelo["mae"]))
        m4.metric("R² em escala log", f"{resultado_modelo['r2']:.3f}")

        st.info(
            """
            O erro médio absoluto indica, em média, a diferença entre a previsão e o valor real.
            O R² mostra o quanto o modelo consegue explicar da variação dos dados na base de teste.
            """
        )

        st.subheader("Variáveis mais importantes para o modelo")

        importancias = resultado_modelo["importancias"].head(12)

        fig, ax = plt.subplots(figsize=(10, 6))

        importancias.sort_values("importancia").plot(
            kind="barh",
            x="feature",
            y="importancia",
            legend=False,
            ax=ax
        )

        ax.set_title("Importância das Variáveis no Modelo")
        ax.set_xlabel("Importância")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.3)

        st.pyplot(fig)

        st.warning(
            """
            Limitação importante: algumas variáveis, como likes e comentários, acontecem
            junto com as visualizações. Por isso, o modelo é mais útil para análise do
            comportamento dos vídeos em alta do que para prever o sucesso antes da publicação.
            """
        )


# =====================================================
# ABA 4 - DADOS TRATADOS
# =====================================================

with aba4:
    st.subheader("Amostra dos dados após tratamento")

    colunas_exibir = [
        "publish_country",
        "channel_title",
        "category_id",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "like_rate",
        "comment_rate",
        "engagement_rate",
        "dias_ate_trending",
        "title_length",
        "tag_count"
    ]

    colunas_exibir = [c for c in colunas_exibir if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado[colunas_exibir].head(100),
        use_container_width=True
    )

    st.subheader("Qualidade dos dados")

    nulos = (
        df_filtrado[colunas_exibir]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    st.dataframe(
        nulos.rename("Quantidade de nulos"),
        use_container_width=True
    )


# =====================================================
# ABA 5 - CONCLUSÃO
# =====================================================

with aba5:
    st.subheader("Conclusão executiva")

    pais_mais_frequente = df_filtrado["publish_country"].value_counts().idxmax()
    canal_maior_mediana = (
        df_filtrado
        .groupby("channel_title")["views"]
        .median()
        .sort_values(ascending=False)
        .index[0]
    )

    st.markdown(
        f"""
        A análise mostra que os vídeos em alta no YouTube possuem forte desigualdade
        na distribuição de visualizações. Poucos vídeos concentram volumes muito altos
        de audiência, enquanto a maior parte apresenta alcance mais moderado.

        Dentro dos filtros selecionados, o país com maior quantidade de vídeos analisados foi
        **{pais_mais_frequente}**. O canal com maior mediana de visualizações foi
        **{canal_maior_mediana}**.

        Também foi observado que métricas de interação, como likes e comentários,
        possuem relação relevante com o volume de visualizações. Porém, a taxa de engajamento
        deve ser analisada separadamente, pois um vídeo muito visto nem sempre é o vídeo
        proporcionalmente mais engajado.

        O modelo de Machine Learning criado com Scikit-learn serviu como apoio para identificar
        quais variáveis ajudam a explicar o alcance dos vídeos. Ainda assim, ele possui limitações,
        já que algumas informações usadas no treinamento, como likes e comentários, só são
        conhecidas depois que o vídeo já começou a performar.

        Portanto, a solução ajuda criadores de conteúdo, equipes de marketing e analistas digitais
        a entender padrões de audiência, comparar países, canais e categorias, além de identificar
        fatores associados ao sucesso de vídeos em alta.
        """
    )

    st.success(
        """
        Projeto concluído com uso de Pandas, Numpy, Matplotlib, Streamlit e Scikit-learn.
        """
    )