import pandas as pd
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('busca.csv')
Y_df = df.comprou
X_df = df[['home', 'busca', 'logado']]

X_dummies_df = pd.get_dummies(X_df)
Y_dummies_df = Y_df

X = X_dummies_df.values
Y = Y_dummies_df.values

tamanho_treino = 0.9 * len(Y)
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:round(tamanho_treino)]
treino_marcacoes = Y[:round(tamanho_treino)]

teste_dados = X[-round(tamanho_teste):]
teste_marcacoes = Y[-round(tamanho_teste):]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)
resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(teste_dados)
taxa_acerto = 100.0 * total_acertos / total_elementos

print(f"Taxa de acerto: {taxa_acerto}%")
print(f"Total de elementos: {total_elementos}")