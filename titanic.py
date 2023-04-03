from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv, get_dummies, DataFrame
from os import listdir

dados_treino = read_csv('train.csv')
dados_teste = read_csv('test.csv')
dados_genero = read_csv('gender_submission.csv')

sobreviventes = dados_treino['Survived']

colunas = ['Pclass', 'Sex', 'SibSp', 'Parch']

sob_categorizados = get_dummies(dados_treino[colunas])
sob_teste_categorizados = get_dummies(dados_teste[colunas])

modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
modelo.fit(sob_categorizados, sobreviventes)

previsoes = modelo.predict(sob_teste_categorizados)

res_submissao = DataFrame({'PassengerId': dados_teste.PassengerId, 'Survived': previsoes})
res_visualizacao = DataFrame({'ID': dados_teste.PassengerId, 'Sobreviveu': ['Sim' if x else 'Não' for x in previsoes], 
                          'Classe': [['Primeira', 'Segunda', 'Terceira'][x - 1] for x in dados_teste.Pclass], 
                          'Sexo': ['Homem' if x else 'Mulher' for x in dados_teste.Sex], 
                          'Qntd. Parentes': dados_teste.SibSp,
                          'Qntd. Filhos/Pais': dados_teste.Parch})

res_submissao.to_csv('saida/submission.csv', index=False)
res_visualizacao.to_csv('saida/full_results.csv', index=False)