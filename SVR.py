import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# === ETAPA 1: Carregar os Dados ===

# Carregar o DataFrame principal
df = pd.read_excel('dataframe.xlsx')
df['data'] = pd.to_datetime(df['data'])

# Carregar o DataFrame de inflação
df_inflacao = pd.read_excel('inflacao.xlsx')
df_inflacao['percentual de inflacao'] = df_inflacao['percentual de inflacao'].str.replace(',', '.').astype(float)
df_inflacao['ano'] = df_inflacao['data referencia'].dt.year

# Mapear inflação acumulada para o DataFrame principal
df['ano'] = df['data'].dt.year
inflacao_map = df_inflacao.set_index('ano')['percentual de inflacao']
df['inflacao_acumulada'] = df['ano'].map(inflacao_map)

# Verificar se há valores "NaN" na inflação e preencher
if df['inflacao_acumulada'].isnull().any():
    df['inflacao_acumulada'] = df['inflacao_acumulada'].ffill()

# === ETAPA 2: Criar Características Polinomiais ===

df['data_ord'] = df['data'].map(datetime.toordinal)
df['data_ord_quadrado'] = df['data_ord'] ** 2
df['data_ord_cubo'] = df['data_ord'] ** 3

# Variáveis independentes (X) e dependentes (y)
X = df[['data_ord', 'data_ord_quadrado', 'data_ord_cubo', 'inflacao_acumulada']].values
y1 = df['1d'].values
y2 = df['2d'].values
y3 = df['3d'].values
y4 = df['4d'].values

# === ETAPA 3: Escalonamento ===

# Escalonar X
escalonador_X = StandardScaler()
X_escalonado = escalonador_X.fit_transform(X)

# Escalonar y
escalonador_y1 = StandardScaler()
escalonador_y2 = StandardScaler()
escalonador_y3 = StandardScaler()
escalonador_y4 = StandardScaler()

y1_escalonado = escalonador_y1.fit_transform(y1.reshape(-1, 1))
y2_escalonado = escalonador_y2.fit_transform(y2.reshape(-1, 1))
y3_escalonado = escalonador_y3.fit_transform(y3.reshape(-1, 1))
y4_escalonado = escalonador_y4.fit_transform(y4.reshape(-1, 1))

# === ETAPA 4: Dividir os Dados em Treino e Teste ===

X_treino, X_teste, y1_treino, y1_teste = train_test_split(X_escalonado, y1_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y2_treino, y2_teste = train_test_split(X_escalonado, y2_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y3_treino, y3_teste = train_test_split(X_escalonado, y3_escalonado, test_size=0.2, random_state=42)
X_treino, X_teste, y4_treino, y4_teste = train_test_split(X_escalonado, y4_escalonado, test_size=0.2, random_state=42)

# === ETAPA 5: Treinamento dos Modelos ===

# Parâmetros para otimização do SVR
'''param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.01, 0.1, 0.5],
    'gamma': [0.01, 0.1, 1, 10]
}

def melhor_modelo_svr(X, y):
    busca_grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
    busca_grid.fit(X, y.ravel())
    print(f"Melhores parâmetros: {busca_grid.best_params_}")
    return busca_grid.best_estimator_'''

def melhor_modelo_svr(X, y):
    modelo = SVR(kernel='rbf', C=1000, epsilon=0.01, gamma=1)
    modelo.fit(X, y.ravel())
    return modelo

# Treinar os modelos para cada variável dependente
modelo1 = melhor_modelo_svr(X_treino, y1_treino)
modelo2 = melhor_modelo_svr(X_treino, y2_treino)
modelo3 = melhor_modelo_svr(X_treino, y3_treino)
modelo4 = melhor_modelo_svr(X_treino, y4_treino)

'''=== ETAPA 6: Avaliar os Modelos ===
def avaliar_modelo(modelo, X_teste, y_teste, escalonador):
    y_pred = modelo.predict(X_teste)
    y_pred_original = escalonador.inverse_transform(y_pred.reshape(-1, 1))
    y_teste_original = escalonador.inverse_transform(y_teste)
    
    mse = mean_squared_error(y_teste_original, y_pred_original)
    mae = mean_absolute_error(y_teste_original, y_pred_original)
    r2 = r2_score(y_teste_original, y_pred_original)
    
    return mse, mae, r2

mse1, mae1, r21 = avaliar_modelo(modelo1, X_teste, y1_teste, escalonador_y1)
mse2, mae2, r22 = avaliar_modelo(modelo2, X_teste, y2_teste, escalonador_y2)
mse3, mae3, r23 = avaliar_modelo(modelo3, X_teste, y3_teste, escalonador_y3)
mse4, mae4, r24 = avaliar_modelo(modelo4, X_teste, y4_teste, escalonador_y4)

# === ETAPA 7: Exibindo resultados ===

# Exibir Resultados
print("Desempenho no Conjunto de Teste:")
print(f"Imóveis com 1 dormitório: MSE={mse1:.2f}, MAE={mae1:.2f}, R²={r21}")
print(f"Imóveis com 2 dormitórios: MSE={mse2:.2f}, MAE={mae2:.2f}, R²={r22}")
print(f"Imóveis com 3 dormitórios: MSE={mse3:.2f}, MAE={mae3:.2f}, R²={r23}")
print(f"Imóveis com 4 dormitórios: MSE={mse4:.2f}, MAE={mae4:.2f}, R²={r24}")'''

# Função para previsão de valores de imóveis em uma data futura
def prever_imovel(data_prevista):
    data_prevista_ord = datetime.toordinal(data_prevista)
    ano_previsto = data_prevista.year  # Obter o ano da data prevista
    inflacao_prevista = inflacao_map.get(ano_previsto, inflacao_map.iloc[-1])  # Pegar a inflação acumulada do ano ou usar o último valor disponível

    # Criar os dados para previsão incluindo todas as features
    data_prevista_escalonada = escalonador_X.transform(
        [[data_prevista_ord, data_prevista_ord**2, data_prevista_ord**3, inflacao_prevista]]
    )

    pred1 = escalonador_y1.inverse_transform(modelo1.predict(data_prevista_escalonada).reshape(-1, 1))
    pred2 = escalonador_y2.inverse_transform(modelo2.predict(data_prevista_escalonada).reshape(-1, 1))
    pred3 = escalonador_y3.inverse_transform(modelo3.predict(data_prevista_escalonada).reshape(-1, 1))
    pred4 = escalonador_y4.inverse_transform(modelo4.predict(data_prevista_escalonada).reshape(-1, 1))
    
    return pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0]

# Exemplo de uso da previsão para uma data específica
mes = 5  # mês de previsão (EM CASO DE MESES ANTERIORES A 10, NÃO UTILIZE 0 (ZERO) ANTES)
ano = 2025
data_futura = datetime(ano, mes, 1)

previsao = prever_imovel(data_futura)
print(f"========Previsão para {mes}/{ano}========")
print(f"Imóveis com 1 dormitório: R$ {previsao[0]:.2f}")
print(f"Imóveis com 2 dormitórios: R$ {previsao[1]:.2f}")
print(f"Imóveis com 3 dormitórios: R$ {previsao[2]:.2f}")
print(f"Imóveis com 4 dormitórios: R$ {previsao[3]:.2f}")

# === Plotagem dos resultados para melhor vizualização histórica ===

data_inicio = data_futura - timedelta(days=30*12) # Limita no plot os dados com diferença de 12 meses
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

def plotar_ausencia_dados(mensagem="Sem dados para além de 12 meses"):
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, mensagem, fontsize=18, color='red', alpha=0.7, ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Aviso de Ausência de Dados", fontsize=16)
    plt.axis('off')  # Remover eixos para uma aparência mais limpa
    plt.tight_layout()
    plt.show()
    plt.close()

if not dados_recentes.empty:
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
else:
    plotar_ausencia_dados("Sem dados para amostra além de 12 meses")