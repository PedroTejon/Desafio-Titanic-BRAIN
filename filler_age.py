from csv import reader

def main():
    with open('train.csv', 'r') as f:
        dados_treino = [x for x in reader(f)][1:]
    with open('test.csv', 'r') as f:
        dados_teste = [x for x in reader(f)][1:]

    pronomes = ['Lady.', 'Jonkheer.', 'Dona.', 'Mrs.', 'Capt.', 'Dr.', 'Sir.', 'Mlle.',
                'Col.', 'Major.', 'Rev.', 'Don.', 'Mme.', 'Mr.', 'Miss.', 'Master.', 'Ms.']

    soma = {pronome: [0, 0] for pronome in pronomes}
    for linha in dados_teste:
        pronome = linha[2].split(', ')[1].split(' ')[0]
        if linha[4] != '' and pronome in soma:
            soma_ant = soma[pronome][0]
            qntd_ant = soma[pronome][1]

            soma[pronome] = [soma_ant + float(linha[4]), qntd_ant + 1]

    for linha in dados_treino:
        pronome = linha[3].split(', ')[1].split(' ')[0]
        if linha[5] != '' and pronome in soma:
            soma_ant = soma[pronome][0]
            qntd_ant = soma[pronome][1]

            soma[pronome] = [soma_ant + float(linha[5]), qntd_ant + 1]

    media = {pronome: soma[pronome][0] / soma[pronome][1] for pronome in soma}

    for linha in dados_treino:
        if linha[5] == '':
            linha[5] = f'{media[linha[3].split(", ")[1].split(" ")[0]]:.2f}'
    
    for linha in dados_teste:
        if linha[4] == '':
            linha[4] = f'{media[linha[2].split(", ")[1].split(" ")[0]]:.2f}'

    with open('train edited.csv', 'w') as f:
        f.write('PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n')
        [f.write(','.join(linha[:3]) + f',"{linha[3]}",' + ','.join(linha[4:]) + '\n') for linha in dados_treino]

    with open('test edited.csv', 'w') as f:
        f.write('PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n')
        [f.write(','.join(linha[:2]) + f',"{linha[2]}",' + ','.join(linha[3:]) + '\n') for linha in dados_teste]


if __name__ == '__main__':
    main()
