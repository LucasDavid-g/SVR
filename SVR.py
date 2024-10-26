import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Carregar dados
df = pd.read_excel('dataframe.xlsx')

# Preprocessamento dos dados
df['data'] = pd.to_datetime(df['data'])
df['data_ord'] = df['data'].map(datetime.toordinal)  # Convertendo datas para formato ordinal

# Criar novas características polinomiais
df['data_ord_quadrado'] = df['data_ord'] ** 2
df['data_ord_cubo'] = df['data_ord'] ** 3  # Adicionando uma nova feature cúbica

# Definir as variáveis independentes e dependentes
X = df[['data_ord', 'data_ord_quadrado', 'data_ord_cubo']].values  # Usar variáveis originais e polinomiais
y1 = df['1d'].values
y2 = df['2d'].values
y3 = df['3d'].values
y4 = df['4d'].values

# Escalonamento dos dados
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

# Configurar GridSearchCV para SVR com uma gama maior de parâmetros
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # Testando uma gama mais ampla de C
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto']
}

# Função para encontrar o melhor modelo SVR usando GridSearchCV
def melhor_modelo_svr(X, y):
    busca_grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
    busca_grid.fit(X, y.ravel())
    print(f"Melhores parâmetros: {busca_grid.best_params_}")
    return busca_grid.best_estimator_

# Dividir dados em treino e teste (80% treino, 20% teste)
X_treino, X_teste, y1_treino, y1_teste = train_test_split(X_escalonado, y1_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y2_treino, y2_teste = train_test_split(X_escalonado, y2_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y3_treino, y3_teste = train_test_split(X_escalonado, y3_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y4_treino, y4_teste = train_test_split(X_escalonado, y4_escalonado, test_size=0.2, random_state=42)

# Treinamento dos modelos otimizados com os dados de treino
modelo1 = melhor_modelo_svr(X_treino, y1_treino)
modelo2 = melhor_modelo_svr(X_treino, y2_treino)
modelo3 = melhor_modelo_svr(X_treino, y3_treino)
modelo4 = melhor_modelo_svr(X_treino, y4_treino)

# Previsão nos dados de teste para cada conjunto de dados
y1_teste_pred = escalonador_y1.inverse_transform(modelo1.predict(X_teste).reshape(-1, 1))
y2_teste_pred = escalonador_y2.inverse_transform(modelo2.predict(X_teste).reshape(-1, 1))
y3_teste_pred = escalonador_y3.inverse_transform(modelo3.predict(X_teste).reshape(-1, 1))
y4_teste_pred = escalonador_y4.inverse_transform(modelo4.predict(X_teste).reshape(-1, 1))

# Calcular o MSE e MAE para o conjunto de teste
mse_y1_teste = mean_squared_error(escalonador_y1.inverse_transform(y1_teste), y1_teste_pred)
mse_y2_teste = mean_squared_error(escalonador_y2.inverse_transform(y2_teste), y2_teste_pred)
mse_y3_teste = mean_squared_error(escalonador_y3.inverse_transform(y3_teste), y3_teste_pred)
mse_y4_teste = mean_squared_error(escalonador_y4.inverse_transform(y4_teste), y4_teste_pred)

mae_y1_teste = mean_absolute_error(escalonador_y1.inverse_transform(y1_teste), y1_teste_pred)
mae_y2_teste = mean_absolute_error(escalonador_y2.inverse_transform(y2_teste), y2_teste_pred)
mae_y3_teste = mean_absolute_error(escalonador_y3.inverse_transform(y3_teste), y3_teste_pred)
mae_y4_teste = mean_absolute_error(escalonador_y4.inverse_transform(y4_teste), y4_teste_pred)

r2_y1_teste = r2_score(escalonador_y1.inverse_transform(y1_teste), y1_teste_pred)
r2_y2_teste = r2_score(escalonador_y2.inverse_transform(y2_teste), y2_teste_pred)
r2_y3_teste = r2_score(escalonador_y3.inverse_transform(y3_teste), y3_teste_pred)
r2_y4_teste = r2_score(escalonador_y4.inverse_transform(y4_teste), y4_teste_pred)

print("========Desempenho no Conjunto de Teste========")
print(f"Imóveis com 1 dormitório: MSE={mse_y1_teste:.2f}, MAE={mae_y1_teste:.2f}, R²={r2_y1_teste:.2f}")
print(f"Imóveis com 2 dormitórios: MSE={mse_y2_teste:.2f}, MAE={mae_y2_teste:.2f}, R²={r2_y2_teste:.2f}")
print(f"Imóveis com 3 dormitórios: MSE={mse_y3_teste:.2f}, MAE={mae_y3_teste:.2f}, R²={r2_y3_teste:.2f}")
print(f"Imóveis com 4 dormitórios: MSE={mse_y4_teste:.2f}, MAE={mae_y4_teste:.2f}, R²={r2_y4_teste:.2f}")

# Função para previsão de valores de imóveis em uma data futura
def prever_imovel(data_prevista):
    data_prevista_ord = datetime.toordinal(data_prevista)
    data_prevista_escalonada = escalonador_X.transform([[data_prevista_ord, data_prevista_ord**2, data_prevista_ord**3]])  # Adicionar a variável polinomial
    
    pred1 = escalonador_y1.inverse_transform(modelo1.predict(data_prevista_escalonada).reshape(-1, 1))
    pred2 = escalonador_y2.inverse_transform(modelo2.predict(data_prevista_escalonada).reshape(-1, 1))
    pred3 = escalonador_y3.inverse_transform(modelo3.predict(data_prevista_escalonada).reshape(-1, 1))
    pred4 = escalonador_y4.inverse_transform(modelo4.predict(data_prevista_escalonada).reshape(-1, 1))
    
    return pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0]

# Exemplo de uso da previsão para uma data específica
mes = 12  # mês de previsão (EM CASO DE MESES ANTERIORES A 10, NÃO UTILIZE 0 (ZERO) ANTES)
ano = 2024  # ano de 2025
data_futura = datetime(ano, mes, 1)

previsao = prever_imovel(data_futura)
print(f"========Previsão para {mes}/{ano}========")
print(f"Imóveis com 1 dormitório: R$ {previsao[0]:.2f}")
print(f"Imóveis com 2 dormitórios: R$ {previsao[1]:.2f}")
print(f"Imóveis com 3 dormitórios: R$ {previsao[2]:.2f}")
print(f"Imóveis com 4 dormitórios: R$ {previsao[3]:.2f}")

# Limitar os dados a 10 meses para trás
data_inicio = data_futura - timedelta(days=30*10)
dados_recentes = df[df['data'] >= data_inicio]

# Adicionar previsões para os meses até a data futura
previsoes = []
datas_previstas = []

# Obter o último mês de dados reais
ultimo_mes_real = df['data'].max()

# Começar as previsões a partir do próximo mês
proximo_mes = ultimo_mes_real + pd.DateOffset(months=1)

while proximo_mes <= data_futura:
    previsoes.append(prever_imovel(proximo_mes))
    datas_previstas.append(proximo_mes)
    proximo_mes += pd.DateOffset(months=1)

# Plotar resultados
plt.figure(figsize=(10, 6))

# Dados reais
plt.plot(dados_recentes['data'], dados_recentes['1d'], color='blue', marker='o', label='Imóveis com 1 dormitório', alpha=0.6)
plt.plot(dados_recentes['data'], dados_recentes['2d'], color='green', marker='o', label='Imóveis com 2 dormitórios', alpha=0.6)
plt.plot(dados_recentes['data'], dados_recentes['3d'], color='red', marker='o', label='Imóveis com 3 dormitórios', alpha=0.6)
plt.plot(dados_recentes['data'], dados_recentes['4d'], color='purple', marker='o', label='Imóveis com 4 dormitórios', alpha=0.6)

# Adicionando previsões
previsoes = np.array(previsoes)
plt.plot(datas_previstas, previsoes[:, 0], color='blue', marker='o', linestyle='--', label='Imóveis com 1 dormitório (Previsão)')
plt.plot(datas_previstas, previsoes[:, 1], color='green', marker='o', linestyle='--', label='Imóveis com 2 dormitórios (Previsão)')
plt.plot(datas_previstas, previsoes[:, 2], color='red', marker='o', linestyle='--', label='Imóveis com 3 dormitórios (Previsão)')
plt.plot(datas_previstas, previsoes[:, 3], color='purple', marker='o', linestyle='--', label='Imóveis com 4 dormitórios (Previsão)')

# Conectar o último ponto real ao primeiro ponto previsto no plot
plt.plot([dados_recentes['data'].iloc[-1], datas_previstas[0]], [dados_recentes['1d'].iloc[-1], previsoes[0][0]], color='blue', linestyle='--')
plt.plot([dados_recentes['data'].iloc[-1], datas_previstas[0]], [dados_recentes['2d'].iloc[-1], previsoes[0][1]], color='green', linestyle='--')
plt.plot([dados_recentes['data'].iloc[-1], datas_previstas[0]], [dados_recentes['3d'].iloc[-1], previsoes[0][2]], color='red', linestyle='--')
plt.plot([dados_recentes['data'].iloc[-1], datas_previstas[0]], [dados_recentes['4d'].iloc[-1], previsoes[0][3]], color='purple', linestyle='--')

plt.title('Previsão de Valores de Imóveis')
plt.xlabel('Data')
plt.ylabel('R$ / m²')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.legend()
plt.grid()
plt.show()