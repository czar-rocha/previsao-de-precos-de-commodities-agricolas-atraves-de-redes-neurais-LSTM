# Previsão de preços de commodities agrícolas através de redes neurais LSTM
#### Aluno: [Junior Rocha](https://github.com/czar-rocha)
#### Orientadora: [Manoela Kohler](https://github.com/manoelakohler)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

### Resumo

Este trabalho apresenta um estudo de aplicação de redes neurais para inferência de preços do milho, uma das principais commodities brasileiras. O setor de commodities agrícolas tem grande relevância na economia brasileira, pois está inserido no comércio global de mercadorias e possui uma parcela significativa de valor nas suas negociações. Na literatura, há diversas pesquisas no campo específico da previsão de commodities, seja utilizando métodos clássicos, modelos de aprendizagem de máquina ou redes neurais. Métodos clássicos, por serem técnicas generalizadas, estão mais suscetíveis a falhas ao tentar se adaptar às características voláteis e não estacionárias do mercado de commodities. Já as redes neurais profundas tem se mostrado uma alternativa eficiente para a previsão de comportamentos voláteis de forma adaptável e dinâmica. O método proposto neste trabalho consiste em utilizar redes neurais recorrentes do tipo Long Short-Term Memory (LSTM) para implementação de modelos univariado e multivariado para previsão de preços do milho. A partir de dados do CEPEA (Centro de Estudos Avançados em Economia Aplicada), foram obtidas as series históricas de três commodities do setor agropecuário brasileiro: milho, soja e boi gordo. Também foi incluído no estudo os dados de um indicador global de commodities, o Commodity Research Bureau Index (CRB). No processo de treinamento da rede buscou-se minimizar a função de perda através da otimização dos hiper-parâmetros. Os resultados de previsão sobre as cotações do milho mostraram uma boa performance nos dados de teste.

### 1. Introdução

#### i) Motivação

O mercado de commodities em geral compõem um setor de grande relevância para a economia brasileira. O comércio de commodities do setor agrícola corresponde a uma significativa parcela de valor nas negociações de mercadorias, que nas últimas décadas tem apresentado altas taxas de crescimento e hoje representa uma parcela significativa do PIB brasileiro. Dessa forma, este é um mercado que tem passado por constante evolução e aperfeiçoamento em seus processos de gestão, com o objetivo de se tornar cada vez mais competitivo e eficiente frente a economia global (CORSINI; RIBEIRO, 2008).

De acordo com FERREIRA et al. (2011) nas últimas décadas este setor se inseriu na economia global e se tornou uma cadeia produtiva que busca cada vez mais uma administração estratégica, centrada na eficácia e eficiência, uma vez os meios de negociação de commodities é considerado complexo por apresentar processos estocásticos nas suas formações de preços. Portanto, as estratégias definidas nos processos de planejamento e tomadas de decisão neste setor devem estar em constante aprimoramento. Neste sentido, a previsão de preços no setor agrícola é um tema de grande relevância por fornecer informações que possam reduzir incertezas e riscos e, consequentemente, apoiar tomadas de decisão de mercado (OUYANG; WEI; WU, 2019).

Segundo CLEMEN e REILLY (2001), toda decisão possui quatro pontos de dificuldade: complexidade, incerteza, objetivos múltiplos que se inter-relacionam e possibilidade de diferentes perspectivas levarem a diferentes conclusões de análise. Dessa forma, as decisões baseadas em análises de incertezas estão sempre presentes no dia-dia dos tomadores de decisão.

No Brasil, existem solidas fontes de dados que realizam a divulgação de series históricos de indicadores de preços de commodities agropecuárias, a exemplo do CEPEA - Centro de Estudos Avançados em Economia Aplicada, que desde 1993 elabora a divulgação de indicadores de diversas commodities agropecuárias brasileiras, tais como Boi Gordo, Milho, Etanol, Açúcar, Soja, Algodão, dentre outras.

No entanto, apesar de haver amplas bases de dados históricos, a previsão de qualquer mercado de ativos em geral, como de commodities, acionário, derivativos ou futuros, é geralmente considerada uma das questões mais desafiadoras entre as previsões de séries temporais, devido ao seu ruído e características voláteis. Isso ocorre porque o comportamento dos preços de qualquer ativo pode mudar drasticamente em diferentes períodos, não havendo como mapear todas as variáveis que influenciam diretamente o comportamento de preços de um determinado ativo (XIONG et al., 2015; LI et al., 2020).

Existe na literatura diversos métodos convencionais voltados para previsão de séries temporais, como por exemplo, modelos de médias móveis, lineares, auto regressivo (AR) e auto-regressivo integrado de médias móveis (ARIMA). No entanto, nos últimos anos houve um avanço bastante significativo no desenvolvimento de métodos de machine learning (Support Vector Machine, Randon Forest etc.) e deep learning (como as redes neurais artificiais e suas variantes, RNN, LSTM, GRU, etc.), tornando esses métodos mais atrativos para previsões de séries temporais, devido a sua capacidade de desempenhar bons resultados de previsão.

De acordo com LI et al. (2020) e SAINI et al. (2020), quando comparado às técnicas tradicionais, os modelos de deep learning desempenham funções mais flexíveis e oferecem uma melhor possibilidade de modelar um comportamento temporal altamente complexo e não linear, pois o aprendizado profundo geralmente elimina a necessidade de estacionariedade no pré-processamento de dados.

Existe na literatura uma diversidade de pesquisas no campo específico da previsão de commodities, seja utilizando métodos clássicos, modelos de aprendizagem de máquina, redes neurais ou até mesmo modelos híbridos ou compostos. Podemos citar alguns trabalhos relacionados a este trabalho, tais como LIU et al. (2017) e CARRASCO et al. (2018), que propuseram modelos de previsão dos preços futuros do cobre a partir de aprendizagem em Random Forest e Suport Vector Machine, respectivamente. Já MALLIARIS e MALLIARIS (2009) empregou uma rede neural artificial para prever os preços do ouro, do petróleo e do euro. No estudo conduzido por SIVALINGAM et a. (2016) foi proposto um modelo de redes neurais para previsão dos preços futuros do ouro a partir dos dados históricos de outras quatro variáveis, composta pelos preços da prata, petróleo bruto, S&P 500 e taxa de câmbio. GARGANO e TIMMERMANN (2014) utilizou modelos de aprendizagem de máquina para previsão do principal índice global de commodities, o Commodity Research Bureau Index (CRB), utilizando em seu modelo um conjunto de variáveis macroeconômicas que poderiam influenciar a dinâmica de precificação do indicador CRB.

Embora as redes neurais tradicionais tenham uma boa capacidade de previsão, a sua capacidade preditiva em séries temporais ainda é desafiadora devido a dinâmica volátil e não lineares. Na tentativa de superar esses desafios, as redes neurais do tipo Long-short Term Memory (LSTM) vem apresentando resultados promissores para o processamento de dados de sequência, como séries temporais.

As redes LSTM são um tipo de RNN com modificações na estrutura de controle de suas camadas. Em uma rede LSTM as estruturas de controle trabalham de forma cooperativa com o objetivo de controlar o fluxo de informação, operando como uma memória e definindo o que a rede deve guardar, e o que deve esquecer ao longo do tempo. Especificamente, Uma célula LSTM contém três tipos de portões de controle: (i) Forget Gate, portão para decidir qual informação não é relevante e deve ser esquecida; (ii) Input Gate, portão para definir quais entradas serão utilizadas para atualizar as células de memória; e (iii) Output Gate, portão que decide quais serão as saídas da célula, levando em conta qual a entrada e qual o estado da célula de memória (ALMEIDA, 2019).

Com o avanço no uso de estruturas de deep learning, as redes neurais LSTM tem sido amplamente empregadas em diversas áreas do conhecimento, tais como no processamento de linguagem natural (GRAVES et al., 2013), previsão de séries temporais (WANG et al., 2018) e análise de emoções (WOLLMER et al., 2013). Esses trabalhos indicaram que redes LSTM tem uma boa capacidade de aprender as dependências entre séries de dados complexas e não lineares.

#### ii) Objetivos do trabalho

O objetivo geral deste trabalho é empregar técnicas de redes neurais LSTM para propor uma abordagem de previsão para as cotações do milho, utilizando-se modelos univariado e multivariado com base nos dados divulgados pelo CEPEA.
Como objetivos específicos, tem-se os seguintes itens: (I) revisão bibliográfica de trabalhos relacionados a aplicação de redes neurais para inferência de preços de commodities agrícolas; (II) preparação dos dados; (III) criação de uma rede neural LSTM e treinamento de modelos univariado e multivariado; e (IV) avaliação de métricas, geração de resultados e conclusões.

### 2. Modelagem

A modelagem deste projeto está organizada em uma metodologia de trabalho composta por cinco etapas: i) Preparação e análise dos dados; ii) Implementação da rede neural LSTM; iii) Treinamento do modelo LSTM Univariado; e iv) Treinamento do modelo LSTM Multivariado.

#### i) Preparação e análise dos dados

Para a aplicação do modelo proposto, foram selecionadas três commodities ligadas ao setor agropecuário brasileiro: milho, soja e boi. Os dados consistem nas series históricas dos indicadores diários de preços fornecidos pela Escola Superior de Agricultura Luiz de Queiroz (Esalq), por meio do banco de dados do Centro de Estudos Avançados em Economia Aplicada [(CEPEA)](https://www.cepea.esalq.usp.br/br/consultas-ao-banco-de-dados-do-site.aspx).

Também foi incluído no estudo os dados de um indicador global de commodities, o Commodity Research Bureau Index (CRB), que é um índice que atua como um indicador representativo dos mercados globais de commodities. Este índice é calculado como uma média dos preços de commodities individuais, formado por 19 commodities, classificadas em 4 grupos, com diferentes pesos: Energia: 39%, Agricultura: 41%, Metais preciosos: 7%, Metais básicos e industriais: 13% [(https://tradingeconomics.com/commodity/crb)](https://tradingeconomics.com/commodity/crb). O CRB busca medir a direção do preço agregado de vários setores de commodities e é utilizado para projetar movimentos direcionais dos preços de comercialização de commodities globais [(https://www.investopedia.com/terms/c/crb.asp)](https://www.investopedia.com/terms/c/crb.asp). A série histórica do CRB foi adquirida a partir do site [Br Investing.com](https://br.investing.com/indices/thomson-reuters---jefferies-crb).

Os datasets utilizados neste trabalho estão disponíveis em formato csv no diretório [dados](dados).

O CEPEA fornece os preços das commodities cotados em real (brl) e dólar (usd). Dado que as commodities brasileiras tem forte influência do dólar, por estarem inseridas no mercado global de commodities, e dado que o CRB é um indicador também referenciado em dólar, constatou-se uma maior correlação entre as cotações das commodities cotadas em dólar e o CRB. O notebook modelo-lstm-multivariado.ipynb ilustra uma análise das correlações entre as séries estudadas.
Dessa forma, por haver maior correlação entre as variáveis, foram consideradas somente as cotações em dólar para as abordagens de previsão univarida e multivariada.
Na preparação dos dados foi realizada ainda a verificação e exclusão de dados nulos, bem como foi adicionada uma feature contendo uma versão dos dados normalizados para valores entre 0 e 1, facilitando assim algumas análises sobre as variáveis.

#### ii) Implementação da rede neural

A implementação da rede LSTM foi realizada utilizando a biblioteca Keras, que é um framework prático e intuitivo para construir e treinar modelos de redes neurais. A rede foi configurada com 3 camadas totalmente conectadas. Foram adicionadas também camadas de eliminação (dropout) após cada camada LSTM oculta, para evitar o problema de overfitting devido à rede densa.

Após definir as camadas da rede, foram especificadas as configurações de aprendizagem. Foi definido o otimizador adam com taxa de aprendizado de 0,01, que é o valor padrão para o otimizador adam, e o erro médio quadrático (MSE, do inglês Mean Squared Error) foi definido como função de perda.

Foram definidos ainda os valores de hiper-parâmetros que foram utilizados para dar início ao processo de treinamento e otimização da rede. Os hiper-parâmetros iniciais foram definidos de forma empírica através de tentativa e erro, com os seguintes valores:
- Window size: 9
- Units: 140
- Dropout: 0.2
- Epochs: 160
- Batch size: 16

As etapas a seguir referentes ao treinamento dos modelos univariado e multivariado realizaram uma otimização dos hiper-parâmetros através da técnica de validação cruzados, onde a combinação de diferentes parâmetros foram testados para identificação da menor função de perda do modelo.

#### iii) Treinamento do modelo LSTM Univariado

O notebook [modelo_lstm_univariado.ipynb](modelo-lstm-univariado/modelo_lstm_univariado.ipynb) mostra o desenvolvimento e teste do modelo univariado. Este modelo foi treinado utilizando apenas a feature de cotação do milho. A série temporal foi dividida em 60-20-20 subconjuntos, onde 60% dos dados foram utilizados para treinamento, 20% para validação  e otimização de parâmetros, e os 20% restantes foram usados para teste.

A otimização dos parâmetros foi realizada a partir dos dados de treino e validação. A Tabela 1 mostra os parâmetros utilizados na otimização e os resultados alcançados após todas as iterações.

| Parâmetros/Métricas | Configuração inicial | Configuração após otimização |
|---|---|---|
| Window size | 9 | 12 |
| Units | 140 | 100 |
| Dropout | 0.2 | 0.2 |
| Epochs | 120 | 180 |
| RMSE | 0.152 | 0.147 |
| MAPE | 16.65 | 16.33 |

Tabela 1: Resultados da otimização do modelo univariado realizado com os dados de treino e validação

A configuração inicial dos parâmetros foi realizada de forma empírica através de iterações experimentais. Após isso, foi definido também de forma empírica, um range de valores em torno da configuração inicial, sendo estes valores testados iterativamente para identificação dos hiper-parâmetros ótimos, através do menor RMSE gerado ao fim do processo.

#### iv) Treinamento do modelo LSTM Multivariado

O notebook [modelo_lstm_multivariado.ipynb](modelo-lstm-multivariado/modelo_lstm_multivariado.ipynb) mostra o desenvolvimento e teste do modelo multivariado. O modelo multivariado foi treinado a partir das features do milho, soja, boi e do CRB. Para o treinamento, cada série temporal foi dividida em 60-20-20 subconjuntos, onde 60% dos dados foram usados para treinamento, 20% para validação e otimização de parâmetros, e os 20% restantes foram usados para teste.

Para ajustar os parâmetros de treinamento e diminuir o erro do modelo, foram realizadas otimizações de hiper-parâmetros através de combinações de diferentes valores. A Tabela 2 mostra os resultados da otimização realizada com os dados de treino e validação.

| Parâmetros/Métricas | Configuração inicial | Configuração após otimização |
|---|---|---|
| Window size | 9 | 15 |
| Units | 140 | 120 |
| Dropout | 0.2 | 0.2 |
| Epochs | 120 | 160 |
| RMSE | 0.163 | 0.147 |
| MAPE | 16.78 | 16.35 |

Tabela 2: Resultados da otimização do modelo multivariado realizado com os dados de treino e validação

### 3. Resultados

O desempenho de previsão dos modelos LSTM foi verificado usando os dados do conjunto de teste. O erro quadrático médio (MSE) foi selecionado como a principal medida para avaliar o desempenho dos modelos.  O MSE dos modelos LSTM univariado e multivariado foi de 0,036 e 0,037, respectivamente. Para fins de verificação das métricas, também foi gerado a raiz do erro médio quadrado (RMSE) e o erro percentual médio absoluto (MAPE) para ambos os modelos.

| Modelo | RMSE | MAPE |
|---|---|---|
| LSTM-Univariado | 0.194 | 28.17 |
| LSTM-Multivariado | 0.195 | 28.04% |

Tabela 3: Performance de previsão dos modelos Univariado e Multivariado aplicado aos dados de teste.

Conforme pode ser visto na Tabela 3, o modelo LSTM univariado teve um desempenho levemente melhor do que o modelo multivariado. Dessa forma, a inclusão de outras variáveis relacionadas a cotação do milho não teve relevância para a performance do modelo. Por outro lado, o modelo multivariado permitiu observar relações entre as variáveis, que se mostraram interessantes.

### 4. Conlusões

Este trabalho propôs uma abordagem de rede neural LSTM univariada e multivariada para previsão de preços do milho. Os modelos propostos passaram por ajustes de hiper parâmetros e apresentaram um bom desempenho nos dados de teste. Os modelos univariado e multivariado apresentaram performances semelhantes, dessa forma, a inclusão de outras variáveis ao estudo não teve relevância na performance do modelo para a previsão das cotações do milho, no entanto a abordagem multivariada agregou conhecimento ao permitir a observância das relações entre as variáveis.

Com o avanço no uso de estruturas de deep learning, as redes neurais LSTM tem sido amplamente empregadas em diversas áreas do conhecimento. Pesquisas recentes tem abordado arquiteturas baseadas em redes autoenconders e em métodos hibridos de decomposição, como o Variational Mode Decomposition (VMD).

Este trabalho propôs uma arquitetura de rede LSTM simples, portanto há espaço para que esta abordagem possa ser testificada em outras arquiteturas de redes neurais. Dessa forma sugere-se como trabalhos futuros a inclusão de outros métodos de inferência aplicados à dinâmica de preços do milho. 

### Referências

CARRASCO, R.; GABRIEL, A.; ISMAEL, S.; MÁX, C. et al. Forecast of copper price series using vector support machines. In: 2018 7th International Conference on Industrial Technology and Management (ICITM), 2018, p. 380-384. DOI: 10.1109/ICITM.2018.8333979.

CLEMEN , R.; REILLY , T. Making Hard Decisions with Decision Tools. 31, p. 127-129, 2001.

CORSINI, F.; RIBEIRO, C. Dinâmica e previsão de preços de commodities agrícolas com o filtro de Kalman. In: XXVIII ENCONTRO NACIONAL DE ENGENHARIA DE PRODUÇÃO, 2008.

FERREIRA, L.; MOURA, G.; BORENSTEIN, D.; FISCHMANN, A. Utilização de redes neurais artificiais como estratégia de previsão de preços no contexto de agronegócio. RAI – Revista de Administração e Inovação.  2011.

GARGANO, A.; TIMMERMANN, A. Forecasting commodity price indexes using macroeconomic and financial predictors. International journal of forecasting, 30, n. 3, p. 825-843, 2014.

GRAVES, A.; ABDEL-RAHMAN, M.; HINTON, G. Speech Recognition with Deep Recurrent Neural Networks. arXiv.org, 2013.

LI, H.; CUI, Y.; WANG, S.; LIU, J. et al. Multivariate Financial Time- Series Prediction with Certified Robustness. IEEE access, 8, p. 1-1, 2020.

LIU, C.; HU, Z.; LI, Y.; LIU, S. Forecasting copper prices by decision tree learning. Resources policy, 52, p. 427-434, 2017.

MALLIARIS, A. G.; MALLIARIS, M., 2009, Time series and neural networks comparison on gold, oil and the euro. 1961-1967. Disponível em: https://www.scopus.com/inward/record.uri?eid=2-s2.0-70449390002&doi=10.1109%2fIJCNN.2009.5178780&partnerID=40&md5=3e473257422a25c3f0829d5d85b66cdc.

OUYANG, H.; WEI, X.; WU, Q. Agricultural commodity futures prices prediction via long- and short- term time series network. Journal of applied economics, 22, n. 1, p. 468-483, 2019.

SAINI, U.; RAJESH, K.; VIPIN, J.; U., K. M. Univariant Time Series forecasting of Agriculture load by using LSTM and GRU RNNs. In: 2020 IEEE Students Conference on Engineering & Systems (SCES), 2020, p. 1-6. DOI: 10.1109/SCES50439.2020.9236695.

SIVALINGAM, K. C.; MAHENDRAN, S.; NATARAJAN, S. Forecasting gold prices based on extreme learning machine. Int. J. Comput. Control, 11, 2016.

WANG, H.; YANG, Z.; YU, Q.; HONG, T. et al. Online reliability time series prediction via convolutional neural network and long short term memory for service- oriented systems. Knowledge-Based Systems, 159, p. 132, 2018.

WOLLMER, M.; KAISER, M.; EYBEN, F.; SCHULLER, B. et al. LSTM- Modeling of continuous emotions in an audiovisual affect recognition framework. Image and Vision Computing, 31, n. 2, p. 153-163, 2013.

XIONG, T.; LI, C.; BAO, Y.; HU, Z. et al. A combination method for interval forecasting of agricultural commodity futures prices. Knowledge-based systems, 77, p. 92-102, 2015.

---

Matrícula: 182.671.010

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação **Business Intelligence Master**
