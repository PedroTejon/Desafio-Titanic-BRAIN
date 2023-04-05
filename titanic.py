from sklearn.ensemble import RandomForestClassifier
from numpy import concatenate
from pandas import read_csv, get_dummies, DataFrame, concat


def prever(dados_treino, dados_teste, colunas):
    sob_categorizados = get_dummies(dados_treino[colunas])
    sob_teste_categorizados = get_dummies(dados_teste[colunas])

    colunas_faltando = set(sob_categorizados.columns).difference(set(sob_teste_categorizados.columns))
    for coluna in colunas_faltando:
        sob_teste_categorizados[coluna] = False
    colunas_faltando = set(sob_teste_categorizados.columns).difference(set(sob_categorizados.columns))
    for coluna in colunas_faltando:
        sob_categorizados[coluna] = False

    modelo = RandomForestClassifier(n_estimators=200, random_state=1)
    modelo.fit(sob_categorizados, dados_treino['Survived'])
    dados_teste['Survived'] = modelo.predict(sob_teste_categorizados)

    return dados_teste


def main():
    dados_treino = read_csv('train.csv')
    dados_treino['Cabin'] = [x[0] if type(x) is str else None for x in dados_treino['Cabin']]
    dados_teste = read_csv('test.csv')
    dados_teste['Cabin'] = [x[0] if type(x) is str else None for x in dados_teste['Cabin']]
    dados_genero = read_csv('gender_submission.csv')

    result_c = prever(dados_treino.query('not Cabin.isnull()'), dados_teste.query('not Cabin.isnull()'), ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Fare'])
    result_s = prever(dados_treino.query('Cabin.isnull()'), dados_teste.query('Cabin.isnull()'), ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare'])

    previsoes = concat([result_c, result_s])
    previsoes.sort_values(by=['PassengerId'], inplace=True)

    res_submissao = DataFrame({'PassengerId': previsoes.PassengerId, 'Survived': previsoes.Survived})

    res_submissao.to_csv('saida/submission5.csv', index=False)


if __name__ == '__main__':
    main()