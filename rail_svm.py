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
        for _ in range(nb_classes):
            self.classifieurs.append(svm.LinearSVC(C = 1, max_iter = 10000))

    def apprentissage(self):
        err = 0
        all_err = np.zeros((self.nb_classes,1))

        #Pour chaque points de la base
        for one_out in range(self.nb_point):
            #Afficher une barre de chargement
            print("  Training  [", end='')
            for i in range(20):
                if i/20 < one_out/self.nb_point :
                    print("#", end='')
                else:
                    print(' ', end='')
            print("] "+str(one_out+1)+"     ", end='\r')

            #Base sans le one out
            Xone_out = np.delete(self.Xapp, one_out, axis=0)
            Yone_out = np.delete(self.Yapp, one_out, axis=0)

            #Permet de séparer la donnée out dans une array 2D
            Xtest = np.array([self.Xapp[one_out,:]])
            Ytest = self.Yapp[one_out]

            #Classification
            for j in range(self.nb_classes):
                #Tous les points sont la classe à rechercher
                Y = np.ones((self.nb_classes, self.nb_point-1))
                #Sauf ceux dont le Yapp != numéro du classifieur
                Y[j,Yone_out!=(j+1)] = -1

                #Lancer l'apprentissage
                self.classifieurs[j].fit(Xone_out, Y[j])

                #Calcul de l'erreur de chacun des classifieurs
                Ypred = self.classifieurs[j].predict(Xtest)
                if (Ypred[0] == 1 and Ytest != j+1) or (Ypred[0] == 0 and Ytest == j+1) :
                    all_err[j] += 1

            #Calcul de l'erreur multiclasse
            Ypred = self.prediction(Xtest)

            if Ypred[0] != Ytest :
                err += 1

        #Affichage des erreures
        err = err/self.nb_point
        all_err = np.divide(all_err, self.nb_point)
        print()
        print("L'erreur du LOO est en moyenne de "+str(int(err*100))+"%")
        for i in range(self.nb_classes) :
            print("Celle du classifieur " + str(i+1) + " est de "+str(int(all_err[i]*100))+"%")


    #Calcule la prédiction en utilisant tous les classifieurs
    def prediction(self, Xtest):
        all_pred = np.zeros((Xtest.shape[0], self.nb_classes))

        #On récupère les taux de certitude d'appartenance à la classe i de chaque classifieurs
        for i in range(self.nb_classes):
            all_pred[:,i] = self.classifieurs[i].decision_function(Xtest)
        
        #On met la prediction à l'indice du classifieur le plus sûr de lui
        Ypred = np.argmax(all_pred, axis=1)+1

        return Ypred

data = np.loadtxt('defautsrails.dat')
nb_classes = 4

X = data[:,:-1]
Y = data[:,-1].astype(int)

classifieur = Mon_Classifieur(X, Y, nb_classes)

classifieur.apprentissage()