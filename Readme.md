# 📄 Documentação do Código: Previsão de Preços de Imóveis com SVR

## ✍ Descrição Geral
Este código utiliza a Regressão de Vetores de Suporte (SVR) para prever os preços de imóveis em função de características temporais. Os dados são carregados de um arquivo Excel e processados para gerar previsões para imóveis com 1 a 4 dormitórios.

##📚 Bibliotecas Utilizadas
- `pandas`: Para manipulação de dados.
- `numpy`: Para operações numéricas.
- `sklearn`: Para implementação do modelo SVR, pré-processamento e avaliação.
- `matplotlib`: Para visualização dos dados e previsões.
- `datetime`: Para manipulação de datas.

## 🖥️ Estrutura do Código

### 1.Carregamento de Dados
```
df = pd.read_excel('dataframe.xlsx')
```

### 2.Pré-processamento de Dados
- Conversão das datas para formato ordinal.
- Criação de características polinomiais (quadrado e cubo das datas).
```
df['data'] = pd.to_datetime(df['data'])
df['data_ord'] = df['data'].map(datetime.toordinal)
df['data_ord_quadrado'] = df['data_ord'] ** 2
df['data_ord_cubo'] = df['data_ord'] ** 3
```

### 3.Definição de Variáveis
Definição das variáveis independentes (X) e dependentes (y1, y2, y3, y4).
```
X = df[['data_ord', 'data_ord_quadrado', 'data_ord_cubo']].values
y1 = df['1d'].values
y2 = df['2d'].values
y3 = df['3d'].values
y4 = df['4d'].values
```
### 4. Escalonamento dos Dados
Os dados são escalonados para melhorar o desempenho do modelo SVR.
```
escalonador_X = StandardScaler()
escalonador_y1 = StandardScaler()
escalonador_y2 = StandardScaler()
escalonador_y3 = StandardScaler()
escalonador_y4 = StandardScaler()

X_escalonado = escalonador_X.fit_transform(X)
y1_escalonado = escalonador_y1.fit_transform(y1.reshape(-1, 1))
y2_escalonado = escalonador_y2.fit_transform(y2.reshape(-1, 1))
y3_escalonado = escalonador_y3.fit_transform(y3.reshape(-1, 1))
y4_escalonado = escalonador_y4.fit_transform(y4.reshape(-1, 1))
```
### 5. Busca de Parâmetros com GridSearchCV
Configuração do GridSearchCV para encontrar os melhores parâmetros para o modelo SVR.
```
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto']
}

def melhor_modelo_svr(X, y):
    busca_grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
    busca_grid.fit(X, y.ravel())
    print(f"Melhores parâmetros: {busca_grid.best_params_}")
    return busca_grid.best_estimator_
```
### 6. Divisão em Conjuntos de Treino e Teste
Os dados são divididos em conjuntos de treino (80%) e teste (20%).
```
X_treino, X_teste, y1_treino, y1_teste = train_test_split(X_escalonado, y1_escalonado, test_size=0.2, random_state=42)
[Repetido para y2, y3, e y4...]
```
### 7. Treinamento dos Modelos
Os modelos SVR são treinados usando os dados de treino.
```
modelo1 = melhor_modelo_svr(X_treino, y1_treino)
[Repetido para modelo2, modelo3, e modelo4...]
```
### 8. Previsões e Avaliação
As previsões são realizadas sobre o conjunto de teste e as métricas de desempenho são calculadas.
```
y1_teste_pred = escalonador_y1.inverse_transform(modelo1.predict(X_teste).reshape(-1, 1))
[# Cálculo de MSE, MAE, e R² para cada modelo...]
```
### 9. Função de Previsão
```
def prever_imovel(data_prevista):
[Lógica de previsão...]
```
### 10. Visualização
Resultados e previsões são visualizados em um gráfico.
```
plt.figure(figsize=(10, 6))
[Plotagem de dados reais e previsões...]
```
## 🔘 Uso do código
Para prever o preço de imóveis, ajuste a variável mes e ano na seção de exemplo de uso da previsão e execute o código.
```
mes = 12
ano = 2024
data_futura = datetime(ano, mes, 1)
previsao = prever_imovel(data_futura)
python
df = pd.read_excel('dataframe.xlsx')
```
