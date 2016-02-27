import csv
import numpy as np
from sklearn import preprocessing

def processData(filename):

    le = preprocessing.LabelEncoder()

    X = []
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        i = 0
        for row in data:
            X.append(np.array(row))
            i += 1;

    #X_matrix = np.matrix(X[: len(X) - 1])
    #print(X)
    X = X[0:len(X)-1 :]
    X = np.array(X).astype('str')
    lb = preprocessing.LabelEncoder()

    age = 'continuous'
    workclass = ' ?, Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
    fnlwgt = 'continuous'
    education = ' ?, Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'
    education_num = 'continuous'
    marital_status = ' ?, Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
    occupation = ' ?, Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
    relationship = ' ?, Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'
    race = ' ?, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
    sex = ' Female, Male'
    capital_gain = 'continuous'
    capital_loss = 'continuous'
    hour_per_week = 'continuous'
    native_country = ' ?, United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
    income = ' >50K, <=50K'
    names = [age,workclass, fnlwgt, education, education_num, marital_status,occupation,relationship ,race,sex,capital_gain,capital_loss,hour_per_week,native_country, income]

    i = 0;
    for name in names:
        if(name != 'continuous'):
            name = name.split(',')
            #print(name)
            lb.fit(name)
            col = lb.fit_transform(np.ravel(X[:, i]))
            X[:, i] = np.transpose(col)
            #print(col)
            #print(X_matrix[:, i])
        i += 1;

    new_Matrix = np.ones((X.shape[0], X.shape[1] + 1))
    print(new_Matrix.shape)
    new_Matrix[:, 1:new_Matrix.shape[1]] = X
    return new_Matrix



