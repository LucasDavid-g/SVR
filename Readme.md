# Previsão de Valores de Imóveis - Documentação Técnica

## 📋 Visão Geral

O código tem como objetivo prever os preços de imóveis no Rio de Janeiro, utilizando a técnica de Support Vector Regression (SVR) para prever os valores de imóveis com diferentes números de dormitórios, considerando características temporais e inflação acumulada.

## 🛠 Dependências

- pandas
- numpy
- scikit-learn
- matplotlib
- datetime

## 🔍 Etapas do Processamento

### 1. Carregamento e Preparação dos Dados

- Carrega dois arquivos Excel: 
  - `dataframe.xlsx`: Conjunto de dados principal dos imóveis
  - `inflacao.xlsx`: Dados de inflação acumulada

- Processamento de datas:
  - Converte a coluna 'data' para datetime
  - Mapeia o ano para cada registro
  - Incorpora dados de inflação acumulada ao DataFrame principal

### 2. Criação de Características

Criação de características para melhorar a precisão do modelo:
- `data_ord`: Data convertida para ordinal
- `data_ord_quadrado`: Quadrado da data ordinal
- `data_ord_cubo`: Cubo da data ordinal
- `inflacao_acumulada`: Percentual de inflação do ano correspondente

### 3. Escalonamento de Dados

Utilização do StandardScaler para:
- Escalonar variáveis independentes (X)
- Escalonar variáveis dependentes (y) para cada número de dormitórios

### 4. Divisão de Dados

- Divisão dos dados em conjuntos de treino e teste
  - 80% para treinamento
  - 20% para teste
- Uso de `train_test_split` com seed fixo (42) para reprodutibilidade

### 5. Treinamento de Modelos

- Implementação de Support Vector Regression (SVR) com kernel RBF
- Treinamento de modelos separados para imóveis com 1, 2, 3 e 4 dormitórios
- Hiperparâmetros fixos:
  - `C = 1000`
  - `epsilon = 0.01`
  - `gamma = 1`

### 6. Função de Previsão

Função `prever_imovel(data_prevista)`:
- Recebe uma data futura
- Obtém inflação acumulada para o ano
- Escala os dados de entrada
- Retorna previsões para imóveis de 1 a 4 dormitórios

### 7. Visualização

- Gráfico comparativo com:
  - Dados históricos reais
  - Previsões futuras
- Marcadores e estilos de linha diferenciados
- Formatação de datas no eixo x
- Legenda e grade para melhor compreensão

## 🚀 Uso Exemplo

```python
from datetime import datetime

mes = 5
ano = 2025
data_futura = datetime(ano, mes, 1)
previsao = prever_imovel(data_futura)
```

## ⚠️ Limitações e Considerações

* Depende da qualidade dos dados de entrada
* Utiliza hiperparâmetros fixos (sem busca exaustiva)
* Previsões podem ser afetadas por mudanças não capturadas no modelo

## 🔧 Recomendações para Melhorias

1. Implementar validação cruzada
2. Realizar busca de hiperparâmetros mais abrangente
3. Adicionar mais features (se disponíveis)
4. Implementar tratamento de outliers

## 📈 Métricas de Desempenho

O modelo avalia o desempenho usando:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Coeficiente de Determinação (R²)