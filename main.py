#pip install pandas
#pip install numpy
#pip install openpyxl
#pip install matplotlib
#pip install seaborn
#pip install scikit-learn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importar a base de dados para o python
tabela = pd.read_csv("advertising.csv")

# Vizualizar a base e fazer ajustes na base de dados
print(tabela)

# Análise exploratória -> entender como a sua base de dados está se comportando

# CORRELAÇÃO
# Tv- Vendas
# Jornal- Vendas
# Radio- Vendas


# cria o gráfico
sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")

# exibe o gráfico
plt.show()

# Dados de Treino - 70% à 90%

# Dados de Teste - 10% à 30%

# O que voce está tentando prever é (y) o resto é (x) (x de teste e outro de treino) (y de teste e outro de treino)

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

# x_treino, x_teste, y_treino, y_teste (Nesta Ordem)
# test_size -> proporção teste-treino (30%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Criar a inteligência artificial e fazer as previsoes
modelo_regressaoLinear = LinearRegression()
modelo_arvoreDecisao = RandomForestRegressor()

modelo_regressaoLinear.fit(x_treino, y_treino)
modelo_arvoreDecisao.fit(x_treino, y_treino)

# Fazer a previsão para comparar com o valor real
previsao_regressaoLinear = modelo_regressaoLinear.predict(x_teste)
previsao_arvoreDecisao = modelo_arvoreDecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaoLinear))
print(r2_score(y_teste, previsao_arvoreDecisao))

# Vizualização gráfica das previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsoes Arvore de Decisao'] = previsao_arvoreDecisao
tabela_auxiliar['Previsoes Regressao Linear'] = previsao_regressaoLinear

plt.figure(figsize=(15, 6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

#Fazer uma nova previsao
novos = pd.read_csv("novos.csv")
print(novos)

#previsao vencedora foi a arvore de decisao
previsao = modelo_arvoreDecisao.predict(novos)
print(previsao)