import numpy as np
import matplotlib.pyplot as plt

# 王少然210162401044
def L1(vecXi, vecXj):
    return np.sum(np.abs(vecXi - vecXj))


def kMedian(S, k, distMeas=L1):
    m = np.shape(S)[0]
    sampleTag = np.zeros(m)

    n = np.shape(S)[1]
    clusterCents = np.mat([[-1.93964824, 2.33260803], [7.79822795, 6.72621783], [10.64183154, 0.20088133]])

    sampleTagChanged = True
    SSE = 0.0
    while sampleTagChanged:
        sampleTagChanged = False
        SSE = 0.0

        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(k):
                d = distMeas(clusterCents[j, :], S[i, :])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex
            SSE += minD

        for i in range(k):
            ClustI = S[np.nonzero(sampleTag[:] == i)[0]]
            clusterCents[i, :] = np.median(ClustI, axis=0)

    return clusterCents, sampleTag, SSE


if __name__ == '__main__':
    samples = np.loadtxt("kmeansSamples.txt")
    clusterCents, sampleTag, SSE = kMedian(samples, 3)

    plt.scatter(clusterCents[:, 0].tolist(), clusterCents[:, 1].tolist(), c='r', marker='^')
    plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
    plt.title('K-median Clustering')
    plt.show()

    print("Cluster Centers:")
    print(clusterCents)
    print("Sum of Manhattan Distances:")
    print(SSE)
