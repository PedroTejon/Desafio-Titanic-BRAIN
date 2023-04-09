from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv, get_dummies, DataFrame
from numpy import linspace


def main():
    dados_treino = read_csv('train edited.csv')

    resultados = dados_treino['Survived']
    dados_treino = dados_treino[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']]
    dados_treino_categorizados = get_dummies(dados_treino)

    n_estimators = [x for x in range(100, 2100, 100)]
    max_features = ['sqrt', 'log2', None]
    max_depth = [x for x in range(10, 120, 10)] + [None]
    min_samples_split = [2, 5, 7, 10, 12]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ['gini', 'entropy', 'log_loss']
    bootstrap = [True, False]

    parametros = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion}

    modelo = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=modelo, param_distributions=parametros, n_iter=100, cv=5, verbose=2, random_state=1, n_jobs=-1)
    rf_random.fit(dados_treino_categorizados, resultados)

    print(rf_random.best_params_)


if __name__ == '__main__':
    main()