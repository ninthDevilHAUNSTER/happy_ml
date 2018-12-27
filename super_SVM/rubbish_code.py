
def test_in_plt_new():
    xcord = []
    ycord = []
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    fr = open('../data_loader/testSet.txt')
    for line in fr.readlines():
        lineSplit = line.strip().split('\t')
        xPt = float(lineSplit[0])
        yPt = float(lineSplit[1])
        label = int(lineSplit[2])
        xcord.append(xPt)
        ycord.append(yPt)
        # if (label == -1):
        #     xcord0.append(xPt)
        #     ycord0.append(yPt)
        # else:
        #     xcord1.append(xPt)
        #     ycord1.append(yPt)
    fr.close()
    dataMat, labelMat = loadDataSet('../data_loader/testSet.txt')
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40, ('lin', 1.0))
    ws = calcWs(alphas, dataMat, labelMat)
    ac = 0
    ac_list = []
    for i in range(dataMat.__len__()):
        ac_list.append(predict(ws, b, dataMat[i]))
        if predict(ws, b, dataMat[i]) == labelMat[i]:
            ac += 1
    print("准确率 {}".format(ac / dataMat.__len__()))

    X = np.array(dataMat, dtype='float')
    Y = np.array(labelMat, dtype='float')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(xcord, ycord)
    print(xx.shape)
    print(xx)
    plt.subplot(1, 1, 1)
    # Z = np.array(ac_list)
    Z = []
    for i in range(xx.__len__()):
        Z.append(predict(ws, b, [xcord[i], ycord[i]]))
    Z = np.array(Z)
    Z = Z.reshape((100, 100))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.legend()

    plt.show()

