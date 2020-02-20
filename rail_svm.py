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
        for i in range(self.nb_classes):
            #Créer un classifieur
            self.classifieurs.append(svm.LinearSVC(C = 1, max_iter = 10000))

            #Tous les points sont la classe à rechercher
            Y = np.ones((self.nb_classes, self.nb_point))
            #Sauf ceux dont le Yapp != numéro du classifieur
            Y[i,self.Yapp!=(i+1)] = -1

            #Lancer l'apprentissage
            self.classifieurs[i].fit(self.Xapp, Y[i])
    
    def apprentissage_loo(self):
        for j in range(self.nb_classes):
            #Créer un classifieur
            self.classifieurs.append(svm.LinearSVC(C = 1, max_iter = 10000))

        err = 0
        all_err = np.zeros((self.nb_classes,1))
        for one_out in range(self.nb_point):
            #Afficher une barre de chargement
            print("  Training  [", end='')
            for i in range(20):
                if i/20 < one_out/self.nb_point :
                    print("#", end='')
                else:
                    print(' ', end='')
            print("] "+str(one_out+1), end='\r')

            Xone_out = np.delete(self.Xapp, one_out, axis=0)
            Yone_out = np.delete(self.Yapp, one_out, axis=0)


            for j in range(self.nb_classes):
                #Tous les points sont la classe à rechercher
                Y = np.ones((self.nb_classes, self.nb_point-1))
                #Sauf ceux dont le Yapp != numéro du classifieur
                Y[j,Yone_out!=(j+1)] = -1

                #Lancer l'apprentissage
                self.classifieurs[j].fit(Xone_out, Y[j])

                Ypred = self.classifieurs[j].predict(self.Xapp)
                if (Ypred[one_out] == 1 and self.Yapp[one_out] != j+1) or (Ypred[one_out] == 0 and self.Yapp[one_out] == j+1) :
                    all_err[j] += 1

            
            Ypred = self.prediction(self.Xapp)


            if Ypred[one_out] != self.Yapp[one_out] :
                err += 1

        err = err/self.nb_point
        all_err = np.divide(all_err, self.nb_point)
        print()
        print("L'erreur du LOO est en moyenne de "+str(int(err*100))+"%")
        for i in range(self.nb_classes) :
            print("L'erreur du classifieur " + str(i+1) + " du LOO est en moyenne de "+str(int(all_err[i]*100))+"%")


    
    def prediction(self, Xtest):
        all_pred = np.zeros((Xtest.shape[0], self.nb_classes))

        #On récupère les taux de certitude d'appartenance à la classe i de chaque classifieurs
        for i in range(self.nb_classes):
            all_pred[:,i] = self.classifieurs[i].decision_function(Xtest)
        
        #On met la prediction à l'indice du classifieur le plus sûr de lui
        Ypred = np.argmax(all_pred, axis=1)+1

        return Ypred
    
    def every_pred(self, Xtest):
        Ypred = np.zeros((Xtest.shape[0], self.nb_classes))

        for i in range(self.nb_classes):
            Ypred[:,i] = self.classifieurs[i].predict(Xtest)

        return Ypred

def evaluer(Ytest, Ypred):
    nb = Ytest.shape[0]
    err = np.count_nonzero(Ytest!=Ypred)

    return err/nb

def evaluer_all(Ytest, Ypred):
    err = []

    for i in range(Ypred.shape[1]):
        err.append( 1 - (
                np.count_nonzero(Ypred[Ytest==(i+1),i])
                /
                np.count_nonzero(Ytest==(i+1))
            )
        )

    return err

data = np.loadtxt('defautsrails.dat')
nb_classes = 4

X = data[:,:-1]
Y = data[:,-1].astype(int)

classifieur = Mon_Classifieur(X, Y, nb_classes)

classifieur.apprentissage_loo()

# Ypred = classifieur.prediction(X)

# err = evaluer(Y, Ypred)

# print(str(int(err*100)) + "% d'erreur pour la classification")

# Ypred = classifieur.every_pred(X)
# err = evaluer_all(Y, Ypred)

# for i in range(nb_classes):
#     print(str(int(err[i]*100)) + "% d'erreur pour le classifieur " + str(i+1))