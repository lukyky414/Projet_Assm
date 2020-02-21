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

    def apprentissage(self):
        my_file = open("log.txt", "w+")
        err_gen = 0

        #Pour chaque points de la base
        for one_out in range(self.nb_point):

            #Base sans le one out
            Xone_out = np.delete(self.Xapp, one_out, axis=0)
            Yone_out = np.delete(self.Yapp, one_out, axis=0)

            #Permet de séparer la donnée out dans une array 2D
            Xtest_one = np.array([self.Xapp[one_out,:]])
            Ytest_one = self.Yapp[one_out]

            #Choix du meilleur C
            best_c = 0
            best_err = self.nb_point
            for c in [0.01, 0.1, 1, 10, 100]:
                self.classifieurs = []
                for _ in range(nb_classes):
                    self.classifieurs.append(svm.LinearSVC(C = c, max_iter = 10000))
                err = 0
                for two_out in range(self.nb_point-1):
                    #barre de chargement
                    print("Training pour "+str(one_out+1)+"/"+str(self.nb_point)+" [", end='')
                    for i in range(20):
                        if i/20 < two_out/(self.nb_point-1) :
                            print("#", end='')
                        else:
                            print(' ', end='')
                    print("] C="+str(c)+"      ", end='\r')
                    
                    Xtwo_out = np.delete(Xone_out, two_out, axis=0)
                    Ytwo_out = np.delete(Yone_out, two_out, axis=0)

                    Xtest_two = np.array([Xone_out[two_out,:]])
                    Ytest_two = Yone_out[two_out]

                    #Classification
                    for j in range(self.nb_classes):
                        #Tous les points sont la classe à rechercher
                        Y = np.ones((self.nb_classes, self.nb_point-2))
                        #Sauf ceux dont le Yapp != numéro du classifieur
                        Y[j,Ytwo_out!=(j+1)] = -1

                        #Lancer l'apprentissage
                        self.classifieurs[j].fit(Xtwo_out, Y[j])

                    #Calcul de l'erreur multiclasse
                    Ypred = self.prediction(Xtest_two)

                    if Ypred[0] != Ytest_two :
                        err += 1
                
                #Garder le meilleur c
                if err < best_err :
                    best_c = c
                    best_err = err

            my_file.write("i:"+str(one_out)+" - c:"+str(best_c)+" - e:"+str(best_err)+"/"+str(self.nb_point-2)+"\n")
            
            #On a choisi le meilleur C, maintenant il faut apprendre dessus avec le one out
            self.classifieurs = []
            for _ in range(nb_classes):
                self.classifieurs.append(svm.LinearSVC(C = best_c, max_iter = 10000))


            #Classification
            for j in range(self.nb_classes):
                #Tous les points sont la classe à rechercher
                Y = np.ones((self.nb_classes, self.nb_point-1))
                #Sauf ceux dont le Yapp != numéro du classifieur
                Y[j,Yone_out!=(j+1)] = -1

                #Lancer l'apprentissage
                self.classifieurs[j].fit(Xone_out, Y[j])

            #Calcul de l'erreur multiclasse
            Ypred = self.prediction(Xtest_one)

            if Ypred[0] != Ytest_one :
                err_gen += 1

        #Affichage des erreures
        err_gen = err_gen/self.nb_point
        print()
        print("L'erreur du LOO est en moyenne de "+str(int(err*100))+"%")
        my_file.write("L'erreur du LOO est en moyenne de "+str(err)+"")

        my_file.close()


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