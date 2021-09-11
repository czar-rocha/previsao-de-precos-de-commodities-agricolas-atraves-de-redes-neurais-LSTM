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

O objetivo geral deste trabalho é empregar técnicas de redes neurais LSTM para propor uma abordagem de previsão para as cotações do milho, utilizando-se modelos univariados e multivariados com base nos dados divulgados pelo CEPEA.
Como objetivos específicos, tem-se os seguintes itens: (i) revisão bibliográfica de trabalhos relacionados a aplicação de redes neurais para inferência de preços de commodities agrícolas; (ii) preparação dos dados; (iii) criação de uma rede neural LSTM e treinamento de modelos univariado e multivariado; e (iv) avaliação de métricas, geração de resultados e conclusões.

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

O notebook modelo-lstm-univariado.ipynb mostra o desenvolvimento e teste do modelo univariado. Este modelo foi treinado utilizando apenas a feature de cotação do milho. A série temporal foi dividida em 60-20-20 subconjuntos, onde 60% dos dados foram utilizados para treinamento, 20% para validação  e otimização de parâmetros, e os 20% restantes foram usados para teste.

A otimização dos parâmetros foi realizada a partir dos dados de treino e validação. A Tabela 1 mostra os parâmetros utilizados na otimização e os resultados alcançados após todas as iterações.

| Parâmetros | Configuração inicial | Configuração após otimização |
|---|---|---|
| Window size | 9 | 12 |
| Units | 140 | 100 |
| Dropout | 0.2 | 0.2 |
| Epochs | 120 | 180 |
| MSE | 0.023 | 0.021 |
| RMSE | 0.152 | 0.147 |
| MAPE | 16.65 | 16.33 |

Tabela 1: Resultados da otimização do modelo univariado realizado com os dados de treino e validação

A configuração inicial dos parâmetros foi realizada de forma empírica através de iterações experimentais. Após isso, foi definido também de forma empírica, um range de valores em torno da configuração inicial, sendo estes valores testados iterativamente para identificação dos hiper-parâmetros ótimos, através do menor MSE gerado ao fim do processo.

### Treinamento do modelo LSTM Multivariado

O notebook modelo-lstm-multivariado.ipynb mostra o desenvolvimento e teste do modelo multivariado. O modelo multivariado foi treinado a partir das features do milho, soja, boi e do CRB. Para o treinamento, cada série temporal foi dividida em 60-20-20 subconjuntos, onde 60% dos dados foram usados para treinamento, 20% para validação e otimização de parâmetros, e os 20% restantes foram usados para teste.

Para ajustar os parâmetros de treinamento e diminuir o erro do modelo, foram realizadas otimizações de hiper-parâmetros através de combinações de diferentes valores. A Tabela 2 mostra os resultados da otimização realizada com os dados de treino e validação.

| Parâmetros | Configuração inicial | Configuração após otimização |
|---|---|---|
| Window size | 9 | 15 |
| Units | 140 | 120 |
| Dropout | 0.2 | 0.2 |
| Epochs | 120 | 160 |
| MSE | 0.026 | 0.021 |
| RMSE | 0.163 | 0.147 |
| MAPE | 16.78 | 16.35 |

Tabela 2: Resultados da otimização do modelo multivariado realizado com os dados de treino e validação

<h2 id="resultados">Resultados</h2>

O desempenho de previsão dos modelos LSTM foi verificado usando os dados do conjunto de teste. O erro quadrático médio (MSE) foi selecionado como a principal medida para avaliar o desempenho dos modelos.  O MSE dos modelos LSTM univariado e multivariado foi de 0,036 e 0,037, respectivamente. Para fins de verificação das métricas, também foi gerado a raiz do erro médio quadrado (RMSE) e o erro percentual médio absoluto (MAPE) para ambos os modelos.

| Modelo | MSE | RMSE | MAPE |
|---|---|---|---|
| LSTM-Univariado | 0.038 | 0.194 | 28.17 |
| LSTM-Multivariado | 0.038 | 0.195 | 28.04% |

Tabela 3: Performance de previsão dos modelos aplicado aos dados de teste.

Conforme pode ser visto na Tabela 3, o modelo LSTM univariado teve um desempenho levemente melhor do que o modelo multivariado. Dessa forma, a inclusão de outras variáveis relacionadas a cotação do milho não teve relevância para a performance do modelo. Por outro lado, o modelo multivariado permitiu observar relações entre as variáveis, que se mostraram interessantes.

<h2 id="conclusoes">Conclusões</h2>

Este trabalho propôs uma abordagem de rede neural LSTM univariada e multivariada para previsão dos preços do milho, uma das principais commodities do setor agropecuário brasileiro. Os modelos propostos passaram por ajustes de hiper parâmetros e apresentaram um bom desempenho nos dados de teste. Os modelos univariado e multivariado apresentaram performances semelhantes, dessa forma, a inclusão de outras variáveis ao estudo não teve relevância na performance do modelo para a previsão das cotações do milho, no entanto a abordagem multivariada agregou conhecimento ao permitir a observância das relações entre as variáveis.

Há espaço para que a abordagem sugerida neste trabalho possa ser testificada de forma mais detalhada ao se fazer análises comparativas com outros métodos de inferência em redes neurais. Dessa forma sugere-se como trabalhos futuros a inclusão de outros métodos de inferência aplicados à dinâmica de preços do milho. 


<h2 id="referencias">Referências</h2>


