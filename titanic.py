from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv, get_dummies, DataFrame


def prever(dados_treino, dados_teste, colunas):
    sob_categorizados = get_dummies(dados_treino[colunas])
    sob_teste_categorizados = get_dummies(dados_teste[colunas])

    colunas_faltando = set(sob_categorizados.columns).difference(set(sob_teste_categorizados.columns))
    for coluna in colunas_faltando:
        sob_teste_categorizados[coluna] = False
    colunas_faltando = set(sob_teste_categorizados.columns).difference(set(sob_categorizados.columns))
    for coluna in colunas_faltando:
        sob_categorizados[coluna] = False

    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    modelo.fit(sob_categorizados, dados_treino['Survived'])
    dados_teste['Survived'] = modelo.predict(sob_teste_categorizados)

    return dados_teste


def main():
    dados_treino = read_csv('train edited.csv')
    dados_treino['Cabin'] = [x[0] if type(x) is str else None for x in dados_treino['Cabin']]
    dados_teste = read_csv('test edited.csv')
    dados_teste['Cabin'] = [x[0] if type(x) is str else None for x in dados_teste['Cabin']]

    previsoes = prever(dados_treino, dados_teste, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age'])

    res_submissao = DataFrame({'PassengerId': previsoes.PassengerId, 'Survived': previsoes.Survived})
    res_submissao.to_csv('saida/submission7.csv', index=False)


if __name__ == '__main__':
    main()