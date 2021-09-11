# BI MASTER - ICA - PUC-Rio
## Previsão de preços de commodities agrícolas através de redes neurais LSTM

### Índice
  
- [Resumo](#resumo)
- [Introdução](#introducao)
- [Motivação](#motivacao)
- [Objetivos do trabalho](#objetivos)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Conclusões](#conclusoes)

<h2 id="resumo">Resumo</h2>
Este trabalho apresenta um estudo da aplicação de redes neurais para inferência de preços do milho, uma das principais commodities brasileiras. O setor de commodities agrícolas tem grande relevância na economia brasileira, está inserido no comercio global de mercadorias e possui uma parcela significativa de valor nas suas negociações. Na literatura existente, há diversas pesquisas no campo específico da previsão de commodities, seja utilizando métodos clássicos, modelos de aprendizagem de máquina ou redes neurais. Métodos clássicos, por serem técnicas generalizadas, estão mais suscetíveis a falhas ao tentar se adaptar às características voláteis e não estacionárias do mercado de commodities. Já as redes neurais profundas tem se mostrado uma alternativa eficiente para a previsão de comportamentos voláteis de forma adaptável e dinâmica. O método proposto neste trabalho consiste em utilizar redes neurais recorrentes do tipo Long Short-Term Memory (LSTM) para implementação de modelos univariado e multivariado para previsão de preços do milho com base na série histórica divulgada pelo CEPEA (Centro de Estudos Avançados em Economia Aplicada). Os dados foram obtidos a partir da base do CEPEA, onde foram obtidas as series históricas de três commodities do setor agropecuário brasileiro: milho, soja e boi gordo. O estudo incluiu ainda os dados de um indicador global de commodities, o Commodity Research Bureau Index (CRB). Buscou-se minimizar a função de perda através da otimização dos hiper-parâmetros. Os resultados mostraram uma boa performance nos dados de teste.

<h2 id="introducao">Introdução</h2>
