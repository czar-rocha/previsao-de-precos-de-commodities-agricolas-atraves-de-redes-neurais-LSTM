# BI MASTER - ICA - PUC-Rio
## Previsão de preços de commodities agrícolas através de redes neurais LSTM

### Índice
  
- [Resumo](#resumo)
- [Introdução](#introducao)
- [Objetivos do trabalho](#objetivos)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Conclusões](#conclusoes)
- [Referências](#referencias)

<h2 id="resumo">Resumo</h2>
Este trabalho apresenta um estudo de aplicação de redes neurais para inferência de preços do milho, uma das principais commodities brasileiras. O setor de commodities agrícolas tem grande relevância na economia brasileira, pois está inserido no comércio global de mercadorias e possui uma parcela significativa de valor nas suas negociações. Na literatura, há diversas pesquisas no campo específico da previsão de commodities, seja utilizando métodos clássicos, modelos de aprendizagem de máquina ou redes neurais. Métodos clássicos, por serem técnicas generalizadas, estão mais suscetíveis a falhas ao tentar se adaptar às características voláteis e não estacionárias do mercado de commodities. Já as redes neurais profundas tem se mostrado uma alternativa eficiente para a previsão de comportamentos voláteis de forma adaptável e dinâmica. O método proposto neste trabalho consiste em utilizar redes neurais recorrentes do tipo Long Short-Term Memory (LSTM) para implementação de modelos univariado e multivariado para previsão de preços do milho. A partir de dados do CEPEA (Centro de Estudos Avançados em Economia Aplicada) foram obtidas as series históricas de três commodities do setor agropecuário brasileiro: milho, soja e boi gordo. Também foi incluído no estudo os dados de um indicador global de commodities, o Commodity Research Bureau Index (CRB). No processo de treinamento da rede buscou-se minimizar a função de perda através da otimização dos hiper-parâmetros. Os resultados de previsão sobre as cotações do milho mostraram uma boa performance nos dados de teste.

<h2 id="introducao">Introdução</h2>

<h2 id="objetivos">Objetivos do trabalho</h2>

<h2 id="metodologia">Metodologia</h2>
A metodologia utilizada neste trabalho inclui a realização de cinco etapas envolvidas no desenvolvimento do tema proposto: i) preparação e análise dos dados; ii) implementação da rede neural LSTM; iii) treinamento dos modelos Univariado e Multivariado; e iv) avaliação de métricas e geração de resultados.

### Preparação e análise dos dados

Para a aplicação do modelo proposto, foram selecionadas três commodities ligadas ao setor agropecuário brasileiro: milho, soja e boi. Os dados consistem nas series históricas dos indicadores diários de preços fornecidos pela Escola Superior de Agricultura Luiz de Queiroz (Esalq), por meio do banco de dados do Centro de Estudos Avançados em Economia Aplicada [(CEPEA)](https://www.cepea.esalq.usp.br/br/consultas-ao-banco-de-dados-do-site.aspx).

Também foi incluído no estudo os dados de um indicador global de commodities, o Commodity Research Bureau Index (CRB), que é um índice que atua como um indicador representativo dos mercados globais de commodities. Este índice é calculado como uma média dos preços de commodities individuais, formado por 19 commodities, classificadas em 4 grupos, com diferentes pesos: Energia: 39%, Agricultura: 41%, Metais preciosos: 7%, Metais básicos e industriais: 13% [(https://tradingeconomics.com/commodity/crb)](https://tradingeconomics.com/commodity/crb). O CRB busca medir a direção do preço agregado de vários setores de commodities e é utilizado para projetar movimentos direcionais dos preços de comercialização de commodities globais [(https://www.investopedia.com/terms/c/crb.asp)](https://www.investopedia.com/terms/c/crb.asp). A série histórica do CRB foi adquirida a partir do site [Br Investing.com](https://br.investing.com/indices/thomson-reuters---jefferies-crb).

Os datasets utilizados neste trabalho estão disponíveis em formato csv no diretório data.

O CEPEA fornece os preços das commodities cotados em real (brl) e dólar (usd). Dado que as commodities brasileiras tem forte influência do dólar, por estarem inseridas no mercado global de commodities, e dado que o CRB é um indicador também referenciado em dólar, constatou-se uma maior correlação entre as cotações das commodities cotadas em dólar e o CRB. O notebook modelo-lstm-multivariado.ipynb ilustra uma análise das correlações entre as séries estudadas.
Dessa forma, por haver maior correlação entre as variáveis, foram consideradas somente as cotações em dólar para as abordagens de previsão univarida e multivariada.
Na preparação dos dados foi realizada ainda a verificação e exclusão de dados nulos, bem como foi adicionada uma feature contendo uma versão dos dados normalizados para valores entre 0 e 1, facilitando assim algumas análises sobre as variáveis.


### Implementação da rede neural

A implementação da rede LSTM foi realizada utilizando a biblioteca Keras, que é um framework prático e intuitivo para construir e treinar modelos de redes neurais. A rede foi configurada com 3 camadas totalmente conectadas. Foram adicionadas também camadas de eliminação (dropout) para evitar o problema de overfitting devido à rede densa, portanto, após cada camada LSTM oculta, foi criada uma camada dropout que garante que a rede neural não dependa inteiramente de um determinado neurônio.

Após definir as camadas da rede, foram especificadas as configurações de aprendizagem. Foi definido o otimizador adam com taxa de aprendizado de 0,01, que é o valor padrão para o otimizador adam, e o erro médio quadrático (MSE, do inglês Mean Squared Error) foi definido como função de perda.


### Treinamento do modelo LSTM Univariado

O notebook modelo-lstm-univariado.ipynb mostra o processo de desenvolvimento e teste de performance do modelo univariado. Este modelo foi treinado utilizando apenas a feature de cotação do milho. A série temporal foi dividida em 60-20-20 subconjuntos, onde 60% dos dados foram utilizados para treinamento, 20% para validação  e otimização de parâmetros, e os 20% restantes foram usados para teste.

A otimização dos parâmetros foi realizada a partir dos dados de treino e validação. A Tabela 1 mostra os parâmetros utilizados na otimização e os resultados alcançados após todas as iterações.

| Parâmetros | Configuração inicial | Configuração após otimização |
|---|---|---|
| Window size | 9 | 12 |
| Units | 140 | 100 |
| Dropout | 0.2 | 0.2 |
| Epochs | 120 | 180 |
| MSE | 0.023 | 0.021 |
| rMSE | 0.152 | 0.147 |
| MAPE | 16.65 | 16.33 |


- **Treinamento do modelo LSTM Multivariado:**



<h2 id="resultados">Resultados</h2>

<h2 id="conclusoes">Conclusões</h2>

<h2 id="referencias">Referências</h2>
