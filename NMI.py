import math
import numpy as np
from sklearn import metrics

def myNMI(A,B):
    print(len(A), len(B))
    total = len(A)
    A_ids = set(A)
    print(A_ids)
    B_ids = set(B)
    print(B_ids)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            # idAOccur = np.where(A==idA)
            # idBOccur = np.where(B==idB)
            idAOccur = []
            idBOccur = []
            for index, item in enumerate(A):
                if item == idA:
                    idAOccur.append(index)
            for index, item in enumerate(B):
                if item == idB:
                    idBOccur.append(index)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            # print(idAOccur)
            px = 1.0*len(idAOccur)/total
            # print(idBOccur)
            py = 1.0*len(idBOccur)/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        ax = []
        for index, item in enumerate(A):
            if item == idA:
                ax.append(index)
        idAOccurCount = 1.0*len(ax)
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        bx = []
        for index, item in enumerate(B):
            if item == idB:
                bx.append(index)
        idBOccurCount = 1.0*len(bx)
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    print(MIhat)
    return MIhat


def sklearnNMI(A, B):
    print(metrics.normalized_mutual_info_score(A,B))


if __name__ == '__main__':
    A = np.array([3, 1, 3, 3, 3, 1, 3, 1, 3, 1])
    B = np.array([1, 1, 3, 1, 1, 1, 1, 1, 3, 1])
    # sklearnNMI(A, B)
    myNMI(A, B)