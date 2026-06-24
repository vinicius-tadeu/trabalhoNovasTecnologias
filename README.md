# YouTube Trending Analytics

## Análise de vídeos em alta no YouTube

Este projeto foi desenvolvido para a disciplina **Novas Tecnologias - GPE17N50289**, como parte do **Trabalho Final - AT2 N2**.

**Autor:** Vinícius Tadeu Soares Silva

## Ideia principal

Atualmente, as visualizações são uma das métricas mais importantes para empresas, influenciadores e criadores de conteúdo. Elas indicam alcance, interesse do público e potencial de visibilidade dentro das plataformas digitais.

Com base nisso, este projeto analisa vídeos em alta no YouTube para entender quais categorias, canais e características estão mais relacionadas ao crescimento de visualizações. A análise permite observar tendências de conteúdo e identificar quais tipos de vídeos podem gerar mais atenção do público.

## Problema analisado

Para empresas e influenciadores, entender o que está em alta é importante para tomar decisões mais estratégicas sobre produção de conteúdo, campanhas, posicionamento e comunicação com o público.

Ao analisar vídeos em alta por categoria, é possível observar quais temas concentram maior volume de visualizações, quais canais possuem maior alcance e quais características dos vídeos aparecem com mais frequência entre os conteúdos de destaque.

## Metodologia

O projeto utiliza dados de vídeos em alta no YouTube e realiza as seguintes etapas:

- carregamento dos dados a partir de arquivos CSV locais ou do KaggleHub;
- padronização dos nomes das colunas;
- tratamento de datas, textos, categorias e valores numéricos;
- correção de problemas comuns de encoding em títulos, canais e descrições;
- criação de identificadores alternativos para vídeos com `video_id` inválido;
- remoção de registros duplicados e vídeos removidos ou com erro;
- criação de métricas para análise de visualizações, engajamento e desempenho por categoria;
- geração de gráficos interativos em um dashboard com Streamlit;
- aplicação de um modelo simples de Machine Learning com Random Forest.

## Observação sobre a coluna `views`

Durante o tratamento dos dados, foi necessário considerar uma limitação do dataset: a coluna `views` representa um valor acumulado. Quando o mesmo vídeo aparece em alta por vários dias, somar diretamente essa coluna pode gerar uma distorção nos resultados.

Por isso, o projeto cria a coluna `views_novas_estimadas`, calculando a diferença entre a medição atual e a medição anterior do mesmo vídeo. Essa métrica foi usada para tornar a análise mais coerente e evitar a repetição de visualizações acumuladas.

## Resultados visuais

O dashboard apresenta gráficos que ajudam a interpretar os dados de forma mais simples, como:

- crescimento estimado de visualizações por dia da semana;
- top 10 categorias com maior crescimento de views;
- top 10 canais com maior crescimento de views;
- categorias com maior engajamento médio estimado;
- importância das variáveis utilizadas no modelo preditivo.

Esses resultados permitem visualizar quais categorias e canais concentram maior alcance, ajudando a entender melhor o comportamento dos vídeos em alta.

## Modelo de Machine Learning

Foi criado um modelo simples de **Random Forest Regressor** com a biblioteca Scikit-learn. O objetivo do modelo é estimar o crescimento de visualizações dos vídeos com base em variáveis como:

- likes novos estimados;
- comentários novos estimados;
- engajamento estimado;
- dias até o vídeo entrar em alta;
- tamanho do título;
- quantidade de tags.

O modelo não deve ser interpretado como uma previsão perfeita de viralização. Ele serve como uma demonstração de uso de Machine Learning para identificar padrões nos dados e observar quais variáveis ajudam a explicar melhor o crescimento de visualizações.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Scikit-learn
- KaggleHub

## Como executar o projeto

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o aplicativo Streamlit:

```bash
streamlit run app.py
```

## Conclusão

A análise mostra como dados de vídeos em alta podem ser usados para apoiar decisões estratégicas de conteúdo. Ao observar categorias, canais, visualizações e engajamento, empresas e influenciadores conseguem entender melhor quais tipos de conteúdo geram mais atenção e podem direcionar suas estratégias de forma mais eficiente.

Além disso, o uso de tratamento de dados, visualizações e Machine Learning demonstra como ferramentas de análise podem transformar grandes volumes de dados em informações úteis para tomada de decisão.
