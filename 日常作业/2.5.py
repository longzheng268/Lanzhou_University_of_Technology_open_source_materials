import numpy as np
import matplotlib.pyplot as plt


def L2(vecXi, vecXj):
    '''计算欧氏距离'''
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def kMeans(S, k, distMeas=L2):
    '''K均值聚类'''
    m = np.shape(S)[0]  # 样本总数
    sampleTag = np.zeros(m)

    # 随机产生k个初始簇中心
    n = np.shape(S)[1]  # 样本向量的特征数
    clusterCents = np.mat([[-1.93964824, 2.33260803], [7.79822795, 6.72621783], [10.64183154, 0.20088133]])

    sampleTagChanged = True
    SSE = 0.0
    while sampleTagChanged:
        sampleTagChanged = False
        SSE = 0.0

        # 计算每个样本点到各簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(len(clusterCents)):
                d = distMeas(clusterCents[j, :], S[i, :])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex
            SSE += minD ** 2

        # 重新计算簇中心
        for i in range(len(clusterCents)):
            ClustI = S[np.nonzero(sampleTag[:] == i)[0]]
            if len(ClustI) > 0:
                clusterCents[i, :] = np.mean(ClustI, axis=0)

    return clusterCents, sampleTag, SSE


def bisectingKMeans(S, k):
    '''二分K均值聚类'''
    clusters = [S]  # 初始只有一个簇
    clusterCents = []

    while len(clusters) < k:
        # 找到拥有最多样本的簇
        maxClusterIndex = np.argmax([len(cluster) for cluster in clusters])
        currentCluster = clusters[maxClusterIndex]

        # 对当前簇进行K-means聚类分成2个簇
        newCents, newTags, _ = kMeans(currentCluster, 2)

        # 更新簇列表，用新的两个簇替换旧簇
        clusters.pop(maxClusterIndex)  # 移除旧的簇
        clusters.append(currentCluster[newTags == 0])  # 添加第一个新簇
        clusters.append(currentCluster[newTags == 1])  # 添加第二个新簇

        # 记录新的簇中心
        clusterCents.append(newCents)

    # 最终簇中心
    finalCenters = []
    for cluster in clusters:
        center = np.mean(cluster, axis=0)
        finalCenters.append(center)

    return np.array(finalCenters), clusters


if __name__ == '__main__':
    samples = np.loadtxt("kmeansSamples.txt")
    clusterCents, clusters = bisectingKMeans(samples, 3)

    # 可视化结果
    plt.figure()
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i + 1}")

    plt.scatter(clusterCents[:, 0], clusterCents[:, 1], c='red', marker='^', s=100, label='Centroids')
    plt.title('Bisecting K-means Clustering')
    plt.legend()
    plt.show()

    print("Final Cluster Centers:")
    print(clusterCents)
