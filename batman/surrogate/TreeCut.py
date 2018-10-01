import numpy as np


class cell:

    def __init__(self, name='cellTab'):
        self.name = name


class Tree:

    def __init__(self, name='Regression_Tree'):
        self.name = name

    # # METHODS

    def TreeDecomp(self, trainAbs, trainOrd, np, NbD, Pin=2):
        self.TrainSetInTree = trainAbs
        self.TrainSetOutTree = trainOrd
        self.NbPnts = np
        self.NbDir = NbD
        self.Pmin = Pin
        self.tabCellInit()
        NbCells = 1
        isFini = 1
        npas = 1
        card = np
        while isFini:
            isFini = 0
            NbCellTmp = 0
            Buftab = []
            for i in range(NbCells):
                res = self.Split(self.cellTab[i])
                if res:
                    isFini = res
                    Buftab.append(self.subCellL)
                    Buftab.append(self.subCellR)
                    del self.subCellL
                    del self.subCellR
                    NbCellTmp += 2
                else:
                    Buftab.append(self.cellTab[i])
                    NbCellTmp += 1

            NbCells = NbCellTmp
            self.cellTab = Buftab
            del Buftab
            card = 0
            for i in range(NbCells):
                card += self.cellTab[i].nbPoints
            npas += 1
            self.NbCells = NbCells

# print 'resu '
# print NbCells,card
# for i in range(NbCells):
# print 'cell ',i+1
# toto=self.cellTab[i]
# print 'npts ',toto.nbPoints
# for j in range(toto.nbPoints):
# print ' num ',toto.PointsOf[j]

        return

    def tabCellInit(self):

        self.cellTab = [cell()]

        self.cellTab[0].nbPoints = self.NbPnts

        self.cellTab[0].PointsOf = np.array(range(self.NbPnts), dtype=np.int16)
        self.cellTab[0].boundsInf = np.zeros((self.NbDir, ), dtype=np.float64)
        self.cellTab[0].boundsSup = np.zeros((self.NbDir, ), dtype=np.float64)

        for i in range(self.NbDir):
            self.cellTab[0].boundsInf[i] = np.min(self.TrainSetInTree[:, i])
            self.cellTab[0].boundsSup[i] = np.max(self.TrainSetInTree[:, i])

        self.ARtab = np.zeros((self.NbDir, self.NbDir), dtype=np.float64)
        for i in range(self.NbDir):
            for j in range(self.NbDir):
                self.ARtab[i, j] = (self.cellTab[0].boundsSup[i]
                                    - self.cellTab[0].boundsInf[i]) \
                    / (self.cellTab[0].boundsSup[j]
                       - self.cellTab[0].boundsInf[j])

        return

    def Split(self, cellIn):
        Nright = 0
        Nleft = 0
        ARcoef = 0.1
        BestScoreOfSplit = 1.e99
        ScoreOfSplit = 1.e99
        bestB = 0.
        bestDir = 0

        if cellIn.nbPoints >= 2 * self.Pmin:

            list_dir = range(self.NbDir)  # ??????????????
            list_dir = list(reversed(list_dir))
            for dir in list_dir:  # balayage des directions
                self.sort_inputs(cellIn, dir)
                for j in range(self.Pmin - 1, cellIn.nbPoints - 1 - (self.Pmin
                                                                     - 1)):
                    b = (self.TrainSetInTree[cellIn.PointsOf[j], dir]
                         + self.TrainSetInTree[cellIn.PointsOf[j + 1], dir]) \
                        * 0.5
                    c = self.TrainSetInTree[cellIn.PointsOf[j], dir] \
                        - self.TrainSetInTree[cellIn.PointsOf[j + 1], dir]
                    if abs(c) > 1.e-40:
                        ScoreOfSplit = self.SplitError(cellIn, b, dir)
                        ScoreOfSplit = ScoreOfSplit \
                            * self.ARfunc(self.SplitAR(cellIn, b, dir), ARcoef)
                    if ScoreOfSplit < BestScoreOfSplit and abs(c) > 1.e-40:
                        BestScoreOfSplit = ScoreOfSplit
                        bestB = b
                        bestDir = dir

            # creation des sousCellules  L et R
            # nb cell a droite
            for i in range(cellIn.nbPoints):
                if self.TrainSetInTree[cellIn.PointsOf[i], bestDir] <= bestB:
                    Nright += 1
            # points
            self.subCellL = cell()
            self.subCellR = cell()
            self.subCellR.nbPoints = Nright
            self.subCellL.nbPoints = cellIn.nbPoints - Nright
            self.subCellR.PointsOf = np.zeros((Nright, ), dtype=np.int16)
            self.subCellL.PointsOf = np.zeros(
                (cellIn.nbPoints - Nright, ), dtype=np.int16)

            Nright = 0
            Nleft = 0
            for i in range(cellIn.nbPoints):
                if self.TrainSetInTree[cellIn.PointsOf[i], bestDir] <= bestB:
                    self.subCellR.PointsOf[Nright] = cellIn.PointsOf[i]
                    Nright += 1
                else:
                    self.subCellL.PointsOf[Nleft] = cellIn.PointsOf[i]
                    Nleft += 1
            # bornes
            self.subCellR.boundsInf = cellIn.boundsInf.copy()
            self.subCellR.boundsSup = cellIn.boundsSup.copy()
            self.subCellL.boundsInf = cellIn.boundsInf.copy()
            self.subCellL.boundsSup = cellIn.boundsSup.copy()

            self.subCellR.boundsInf[bestDir] = cellIn.boundsInf[bestDir]
            self.subCellR.boundsSup[bestDir] = bestB
            self.subCellL.boundsSup[bestDir] = cellIn.boundsSup[bestDir]
            self.subCellL.boundsInf[bestDir] = bestB

            return 1
        else:
            return 0

    def sort_inputs(self, cellIn, dir):
        trifini = 1
        while trifini == 0:
            trifini = 1
            for j in range(1, cellIn.nbPoints):
                if self.TrainSetInTree[cellIn.PointsOf[j - 1], dir] \
                        > self.TrainSetInTree[cellIn.PointsOf[j], dir]:
                    tmp = cellIn.PointsOf[j - 1]
                    cellIn.PointsOf[j - 1] = cellIn.PointsOf[j]
                    cellIn.PointsOf[j] = tmp
                    trifini = 0
        return

    def SplitError(self, cellIn, b, dir):
        # evalue l'erreur commise par le split de cellIn avec les parametre
        # dir et b

        accLeft = 0.
        accRight = 0.
        RMS = 0.
        Nleft = 0
        Nright = 0

        # calcul de la valeur moyenne sur chaque sous cellule

        for i in range(cellIn.nbPoints):
            sum_out = self.TrainSetOutTree[cellIn.PointsOf[i], 0]
            if self.TrainSetInTree[cellIn.PointsOf[i], dir] <= b:
                accRight += sum_out
                Nright += 1
            else:
                accLeft += sum_out
                Nleft += 1

        accRight = accRight / Nright
        accLeft = accLeft / Nleft

        # calcul de la RMS

        for i in range(cellIn.nbPoints):
            sum_out = self.TrainSetOutTree[cellIn.PointsOf[i], 0]
            if self.TrainSetInTree[cellIn.PointsOf[i], dir] <= b:
                val = sum_out - accRight
                RMS += val ** 2
            else:
                val = sum_out - accLeft
                RMS += val ** 2

        RMS = np.sqrt(RMS) / cellIn.nbPoints
        return RMS

    def SplitAR(self, cellIn, b, dir):
        # permet d'estimer le rapport d'aspect minimal entre les deux cellules resultant
        # de la decoupe, l'aspect ratio est definit comme le min du rapport deux a deux des
        # dimensions des cellulles
        Armini = 1.e99
        for i in range(self.NbDir):
            for j in range(self.NbDir):
                if i != j:
                    # calcul des differents AR
                    if j == dir:
                        ArtmpL = cellIn.boundsSup[i] - cellIn.boundsInf[i]
                        ArtmpR = ArtmpL
                        try:
                            ArtmpL = ArtmpL / (cellIn.boundsSup[j] - b)
                        except:
                            ArtmpL = 1.e99
                        try:
                            ArtmpR = ArtmpR / (b - cellIn.boundsInf[j])
                        except:
                            ArtmpR = 1.e99
                    if i == dir:
                        ArtmpL = cellIn.boundsSup[i] - b
                        ArtmpR = b - cellIn.boundsInf[i]
                        try:
                            ArtmpL = ArtmpL / (cellIn.boundsSup[j]
                                               - cellIn.boundsInf[j])
                            ArtmpR = ArtmpR / (cellIn.boundsSup[j]
                                               - cellIn.boundsInf[j])
                        except:
                            ArtmpL = 1.e99
                            ArtmpR = 1.e99
                    if j != dir and i != dir:
                        ArtmpL = cellIn.boundsSup[i] - cellIn.boundsInf[i]
                        try:
                            ArtmpL = ArtmpL / (cellIn.boundsSup[j]
                                               - cellIn.boundsInf[j])
                        except:
                            ArtmpL = 1.e99
                        ArtmpR = ArtmpL
                    # correction du rapport d'aspect
                    ArtmpR = ArtmpR / self.ARtab[i, j]
                    ArtmpL = ArtmpL / self.ARtab[i, j]
                    # minimum
                    if ArtmpL < Armini:
                        Armini = ArtmpL
                    if ArtmpR < Armini:
                        Armini = ArtmpR
        return Armini

    def ARfunc(self, AR, seuil):
        # fonction permettant de gerer finement la penalisation des AR trop
        # failbles
        if AR < seuil:
            return 1.e99
        else:
            return 1

    def setOutputs(self):
        # fonction de post traitement des resultats pour la sortie
        # premets de recuperer les centres et rayons des cellules dans centers
        # et radii
        centers = np.zeros((self.NbCells, self.NbDir), dtype=np.float64)
        rayon = np.zeros((self.NbCells, self.NbDir), dtype=np.float64)
        for i in range(self.NbCells):
            cellIn = self.cellTab[i]
            for j in range(self.NbDir):
                centers[i, j] = (cellIn.boundsSup[j] + cellIn.boundsInf[j]) \
                    * 0.5
                rayon[i, j] = np.abs(cellIn.boundsSup[j] - cellIn.boundsInf[j]) \
                    * 0.5
        return (centers, rayon)
