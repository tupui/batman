#  ==========================================================================
#  Project: cfd - POD - Copyright (c) 2005 by CERFACS
#  Type   : RBF network class definitions
#  File   : RBFnet.py
#  Vers   : V1.0
#
#     RBFnet librairie pour l'utilisation de reseaux
#               de neurones a RBF
#     Notation et fondement theoriques issus de
#     "Introduction to Radial Basis Function Networks"
#                  par Mark J.L. Orr
#        www.anc.ed.ac.uk/~mjo/papers/intro.ps
#
from .TreeCut import Tree

from math import sqrt, exp
import numpy as np

class RBFnet:

    def __init__(self, *args, **kwargs):
        if args or kwargs: self.setNetwork(*args, **kwargs)


    def evaluate(self, point):
        return self.evalOut(point), 0


    def show(self):
        print('Radial Basis Function')
        print('  N Sample = %d' % self.Setsize)
        print('  N Input  = %d' % self.Ninput)
        print('  N Output = %d' % self.Noutput)
        print('  N Center = %d' % self.Ncenter)
        print('  Radius   = %s' % self.radius)
        print('  Regparam = %s' % self.regparam)

    # # METHODS RBF

    def evalOut(self, point):

        outacc = np.zeros(self.Noutput, dtype=np.float64)
        # adim
        pts = []
        for i in range(self.Ninput):
            pts.append((point[i] - self.minp[i]) / self.dd[i])
        #
        for i in range(self.Ncenter):
            outacc = outacc + self.weights[i] * self.RBFout(pts, i)
        outacc = outacc + self.Calc_moyenne(pts)
        return outacc

    def Calc_moyenne(self, Point):
        out = self.cf_moyenne[0]
        for i in range(1, self.Ninput + 1):
            out = out + self.cf_moyenne[i] * Point[i - 1]
        return out

    def RBFout(self, Point, neuroNum):
        out = 0.
        distC = 0.

        for i in range(self.Ninput):
            tmp = (Point[i] - self.center[neuroNum, i]) ** 2
            distC = distC + tmp / self.radii[neuroNum, i] ** 2
        return self.rfunction(sqrt(distC))

    def trainNet(self):
        # entrainement du reseau de neurone principal : calcul des poids
        # trainIn tableau de points d'entrainements
        # de taille Setsize*(Ninputs) pour les inputs et Setsize pour les outputs ds trainOut
        H = np.zeros((self.Setsize, self.Ncenter), dtype=np.float64)
        Hymoy = np.zeros((self.Setsize, self.Noutput), dtype=np.float64)

        # calcul des valeurs
        for i in range(self.Setsize):
            for j in range(self.Ncenter):
                H[i, j] = self.RBFout(self.trainIn[i], j)
            Hymoy[i] = self.Calc_moyenne(self.trainIn[i])

        HT = np.transpose(H)
        HTH = np.dot(HT, H)
        putOnDiag(HTH, self.regParam)
        # les valeurs moyennes sont retranchees
        Hy = np.dot(HT, self.trainOut - Hymoy)

        # self.weights = solve_linear_equations(HTH, Hy)
        # numpy.linalg.solve(a, b) ??
        self.weights = np.linalg.solve(HTH, Hy)

    def Calc_Coefs_Moyenne(self):
        # fonction permettant de calculer le plan moyen a partir de laquelle
        # partent les gaussiennes par regression lineaire multiple

        X = np.ones((self.Ninput + 1, self.Setsize), dtype=np.float64)
        for i in range(1, self.Ninput + 1):
            for j in range(self.Setsize):
                X[i, j] = self.trainIn[j, i - 1]

        XT = np.transpose(X)
        XXT = np.dot(X, XT)
        XY = np.dot(X, self.trainOut)
        # self.cf_moyenne = solve_linear_equations(XXT, XY)
        # numpy.linalg.solve(a, b) ??
        self.cf_moyenne = np.linalg.solve(XXT, XY)

    def setNetwork(self, trainIn, trainOut, regparam=0., radius=1.5, regtree=0,
                   function='default', Pmin=2, Radscale=1.):
        # initialise le reseau principal a partir d'un tableau de points
        # d'entrainements de taille Setsize*(Ninputs) pour les inputs et
        # Setsize*(Noutputs) pour les outputs ds trainOut
        # possibilite d'utiliser d'arbre de regression sur les donnees (regtree=1)
        # le reseau est ensuite entraine sur cet ensemble avec
        # le parametere Regparam

        # initialisation
        # trainIn est copie pour le pas affecter l'original dans le prog appelant
        self.trainIn = trainIn.copy()
        self.trainOut = trainOut.copy()
        self.radius = radius
        self.regparam = regparam
        self.Setsize = self.trainIn.shape[0]
        self.Ninput = self.trainIn.shape[1]
        self.Noutput = self.trainOut.shape[1]

        # adimensionnalisation des parametres
        self.minp = np.zeros((self.Ninput, ), dtype=np.float64)
        self.dd = np.ones((self.Ninput, ), dtype=np.float64)
        for i in range(self.Ninput):
            self.minp[i] = min(self.trainIn[::, i])
            maxp = max(self.trainIn[::, i])
            self.dd[i] = maxp - self.minp[i]
            self.trainIn[::, i] = (self.trainIn[::, i] - self.minp[i]) \
                / self.dd[i]

        if function == 'default':
            self.rfunction = default_function
        else:
            self.rfunction = function

        if self.trainIn.shape[0] != self.trainOut.shape[0]:
            raise 'Incoherent number of samples I/O'

        if regtree == 0:
            self.Ncenter = self.Setsize
            self.center = self.trainIn
            self.radii = np.zeros((self.Ncenter, self.Ninput), dtype=np.float64)
            for i in range(self.Ncenter):
                r = self.compute_radius(i) * radius
                for j in range(self.Ninput):
                    self.radii[i, j] = r
        else:

            # test
            if self.Noutput != 1:
                raise 'Output Dim must be 1 with regression Trees'
            # creation de l'arbre de regression

            tree1 = Tree()
            tree1.TreeDecomp(self.trainIn, self.trainOut, self.Setsize,
                             self.Ninput, Pmin)
            (self.center, self.radii) = tree1.setOutputs()
            self.Ncenter = self.center.shape[0]
            del tree1

        self.regParam = np.array([regparam] * self.Ncenter, dtype=np.float64)
        self.funcType = np.array([0.] * self.Ncenter, dtype=np.int32)

        self.Calc_Coefs_Moyenne()
        self.trainNet()

    def compute_radius(self, cel):
        # calcule le rayon pour la cellule i lorsque on utilise pas l'arbre de regression
        # ce rayon est calcule comme la demi distance a la plus proche cellule
        distmin = 1.E99
        plusproche = 0
        for i in range(self.Ncenter):
            if i != cel:
                dist = 0.
                for j in range(self.Ninput):
                    tmp = self.center[cel, j] - self.center[i, j]
                    tmp = tmp ** 2
                    dist = dist + tmp
                if dist < distmin:
                    plusproche = i
                    distmin = dist

        dist = 0.
        for j in range(self.Ninput):
            tmp = self.center[cel, j] - self.center[plusproche, j]
            tmp = tmp ** 2
            dist = dist + tmp

        dist = sqrt(dist)
        return dist


#   END CLASS DEFINITION

#   OTHER FUNCTIONS

#   put on diagonale


def putOnDiag(matrix, vect):
    for i in range(matrix.shape[0]):
        matrix[i, i] = matrix[i, i] + vect[i]


#   default radius function


def default_function(radius):
    try:
        out = exp(-radius ** 2)
    except:
        out = 0.
    return out


#   END

# test
if __name__ == '__main__':

    rbf1 = RBFnet()

    # sample = array (nsample,ninput)
    # out = array (nsample,noutput)
    sample = np.array([[0., 0., 0.], [0., 1., 1.], [0., 2., 2.], [0., 3., 4.],
                   [1., 0., 0.], [1., 1., 5.], [1., 2., 6.], [1., 3., 7.]])
    out = np.array([[0., 0], [0., 100], [0., 200], [0, 300], [10., 0], [10., 100],
                [10., 200], [10., 300]])
#    sample=np.array([  [0.,0.] , [0.,1.] , [0.,2.] , [1.,0.] , [1.,1.] , [1.,2.]  ])
#    out   =np.array([  [0.,0] , [0.,100], [0.,200], [10.,0], [10.,100], [10.,200] ])
#    point=[0.58,1.2]
    point = np.zeros([2, 3], Float)
    point[0, 0] = .7
    point[0, 1] = .2
    point[0, 2] = .5
    point[1, 0] = .6
    point[1, 1] = .1
    point[1, 2] = .5

#    point=[[0.7,0.2,.5],[0.6,0.1,.5]]


    def my_function(radius):
        # radius > 0.
        # 0.<out<1.
        try:
            out = max(0., 1 - .5 * radius)
        except:
            out = 0.
        return out


    # test sans arbre de regression
    rbf1.setNetwork(sample, out, function=my_function)
    print('RBF', point)
    for i in range(2):
        print(point[i, :])
        print('evaluation = ', rbf1.evalOut(point[i, :]))
    del rbf1

    # test avec arbre de regression
    noutput = out.shape[1]
    print('RBF regression Tree')
    for n in range(noutput):
        out2 = out[:, n:n + 1]
        rbf = RBFnet(name='rbf')
        rbf.setNetwork(sample, out2, function=my_function, regtree=1)
        for i in range(2):
            print('evaluation %d = %f' % (n, rbf.evalOut(point[i, :])))
