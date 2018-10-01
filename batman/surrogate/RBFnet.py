"""
RBFnet librairie pour l'utilisation de reseaux de neurones a RBF
Notation et fondement theoriques issus de
"Introduction to Radial Basis Function Networks" par Mark J.L. Orr
www.anc.ed.ac.uk/~mjo/papers/intro.ps
"""
import numpy as np
from .TreeCut import Tree
from ..functions.utils import multi_eval


class RBFnet:
    """RBF class."""

    def __init__(self, trainIn, trainOut, regparam=0., radius=1.5, regtree=0,
                 function='default', Pmin=2, Radscale=1.):
        """Initialization.

        initialise le reseau principal a partir d'un tableau de points
        d'entrainements de taille Setsize*(Ninputs) pour les inputs et
        Setsize*(Noutputs) pour les outputs ds trainOut
        possibilite d'utiliser d'arbre de regression sur les donnees (regtree=1)
        le reseau est ensuite entraine sur cet ensemble avec
        le parametere Regparam

        initialisation
        trainIn est copie pour le pas affecter l'original dans le prog
        appelant
        """
        self.trainIn = np.array(trainIn).copy()
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
            raise ValueError('Incoherent number of samples I/O')

        if regtree == 0:
            self.Ncenter = self.Setsize
            self.center = self.trainIn
            self.radii = np.zeros(
                (self.Ncenter, self.Ninput), dtype=np.float64)
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

        self.coefs_mean()
        self.trainNet()

    def __repr__(self):
        s = ("Radial Basis Function\n"
             "  N Sample: {}\n"
             "  N Input : {}\n"
             "  N Output: {}\n"
             "  N Center: {}\n"
             "  Radius  : {}\n"
             "  Regparam: {}"
             .format(self.Setsize, self.Ninput, self.Noutput,
                     self.Ncenter, self.radius, self.regparam))
        return s

    @multi_eval
    def evaluate(self, point):
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
        distC = 0.

        for i in range(self.Ninput):
            tmp = (Point[i] - self.center[neuroNum, i]) ** 2
            distC = distC + tmp / self.radii[neuroNum, i] ** 2
        return self.rfunction(np.sqrt(distC))

    def trainNet(self):
        """Train.

        entrainement du reseau de neurone principal : calcul des poids
        trainIn tableau de points d'entrainements
        de taille Setsize*(Ninputs) pour les inputs et Setsize pour les
        outputs ds trainOut
        """
        H = np.zeros((self.Setsize, self.Ncenter), dtype=np.float64)
        Hymoy = np.zeros((self.Setsize, self.Noutput), dtype=np.float64)

        # calcul des valeurs
        for i in range(self.Setsize):
            for j in range(self.Ncenter):
                H[i, j] = self.RBFout(self.trainIn[i], j)
            Hymoy[i] = self.Calc_moyenne(self.trainIn[i])

        HT = np.transpose(H)
        HTH = np.dot(HT, H)
        put_on_diag(HTH, self.regParam)
        # les valeurs moyennes sont retranchees
        Hy = np.dot(HT, self.trainOut - Hymoy)

        # self.weights = solve_linear_equations(HTH, Hy)
        # numpy.linalg.solve(a, b) ??
        self.weights = np.linalg.solve(HTH, Hy)

    def coefs_mean(self):
        """Mean coefficients.

        fonction permettant de calculer le plan moyen a partir de laquelle
        partent les gaussiennes par regression lineaire multiple
        """
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

    def compute_radius(self, cel):
        """Radius.

        calcule le rayon pour la cellule i lorsque on utilise pas l'arbre de regression
        ce rayon est calcule comme la demi distance a la plus proche cellule
        """
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

        dist = np.sqrt(dist)
        return dist

#   END CLASS DEFINITION
#   OTHER FUNCTIONS
#   put on diagonale


def put_on_diag(matrix, vect):
    for i in range(matrix.shape[0]):
        matrix[i, i] = matrix[i, i] + vect[i]


#   default radius function
def default_function(radius):
    try:
        out = np.exp(-radius ** 2)
    except:
        out = 0.
    return out


if __name__ == '__main__':
    # sample = array (nsample,ninput)
    # out = array (nsample,noutput)
    sample = np.array([[0., 0., 0.], [0., 1., 1.], [0., 2., 2.], [0., 3., 4.],
                       [1., 0., 0.], [1., 1., 5.], [1., 2., 6.], [1., 3., 7.]])
    out = np.array([[0., 0], [0., 100], [0., 200], [0, 300], [10., 0], [10., 100],
                    [10., 200], [10., 300]])
    point = [[0.7, 0.2, .5], [0.6, 0.1, .5]]

    def my_function(radius):
        # radius > 0.
        # 0.<out<1.
        try:
            out = max(0., 1 - .5 * radius)
        except:
            out = 0.
        return out

    # test sans arbre de regression
    rbf = RBFnet(sample, out, function=my_function)
    print('RBF without regression Tree: \n', point)
    print(rbf.evaluate(point))

    # test avec arbre de regression
    noutput = out.shape[1]
    print('RBF with regression Tree: \n')
    for n in range(noutput):
        out2 = out[:, n:n + 1]
        rbf = RBFnet(sample, out2, function=my_function, regtree=1)
        for i in range(2):
            print('evaluation %d = %f' % (n, rbf.evalOut(point[i, :])))
