import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import neighbors
from sklearn import svm


class Mon_Classifieur():
    def __init__(self, Xapp, Yapp, nb_classes):
        self.Xapp = Xapp
        self.Yapp = Yapp
        self.nb_classes = nb_classes
        self.nb_point = Xapp.shape[0]
        self.classifieurs = []

    def apprentissage(self):
        Y = np.ones((self.nb_classes, self.nb_point))
        for i in range(self.nb_classes - 1):
            self.classifieurs.append(svm.LinearSVC(C = 1, max_iter = 10000))

            Y[i,self.Yapp!=(i+1)] = -1

            self.classifieurs[i].fit(self.Xapp, Y[i])
    
    def prediction(self, Xtest):
        all_pred = np.zeros((Xtest.shape[0], self.nb_classes))

        for i in range(self.nb_classes - 1):
            all_pred[:,i] = self.classifieurs[i].decision_function(Xtest)
        

        Ypred = np.argmax(all_pred, axis=1)+1

        return Ypred

def evaluer(Ytest, Ypred):
    nb = Ytest.shape[0]
    err = np.count_nonzero(Ytest!=Ypred)

    return err/nb


data = np.loadtxt('defautsrails.dat')
nb_classes = 4

X = data[:,:-1]
Y = data[:,-1].astype(int)

classifieur = Mon_Classifieur(X, Y, nb_classes)

classifieur.apprentissage()

Ypred = classifieur.prediction(X)

err = evaluer(Y, Ypred)

print(str(int(err*100)) + "% d'erreur")