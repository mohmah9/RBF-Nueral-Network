import numpy as np
import matplotlib.pyplot as plt


class RBF():
    def __init__(self, k,epochs ,lr=0.01):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def gaussian(self ,x, center, sig):
        return np.exp((-1 / (2 * sig ** 2)) * (x - center) ** 2)

    def sig_finder(self,clusters,X,k):
        '''
            computing spreads of clusters
            finding clusters with 1 or 0 points and compute their spreads as mean of others` spreads
        '''
        sigma = np.zeros(k)
        ideal_cluster = np.argmin(np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :])), axis=1)
        averg = []
        outliers = []
        for i in range(k):
            pointsForCluster = X[ideal_cluster == i]
            if len(pointsForCluster) < 2:
                outliers.append(i)
            else:
                averg.append(X[ideal_cluster == i])
                sigma[i] = np.std(X[ideal_cluster == i])
        averg = np.concatenate(averg)
        sigma[outliers] = np.mean(np.std(averg))
        return sigma

    def clustering(self ,X, k):
        ''' uniform setting of initial centers '''
        lx = len(X)
        diff = lx / k
        o = []
        for i in range(0, k):
            o.append(X[int(diff * i)])
        clusters = np.array(o)
        ''' random choice of initial centers '''
        # clusters = np.random.choice(np.squeeze(X), size=k)
        before = clusters.copy()
        flag = True
        while flag:
            '''find the cluster that's closest to each point'''
            ideal_cluster = np.argmin(np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :])), axis=1)
            '''updating each cluster by taking the mean of all of the points in it'''
            for i in range(k):
                Numbers = X[ideal_cluster == i]
                if len(Numbers) > 0:
                    clusters[i] = np.mean(Numbers, axis=0)
            flag = np.average(clusters - before) > 0.000001
            before = clusters.copy()
        sigma = self.sig_finder(clusters,X,k)
        return clusters, sigma

    def fit(self, X, y):
        self.centers, self.stds = self.clustering(X, self.k)
        for epoch in range(self.epochs):
            if epoch % 100 ==0:
                print("epoch "+str(epoch))
            for i in range(len(X)):
                # forward pass
                sig=np.array(self.gaussian(X[i], self.centers, self.stds))
                Y = np.dot(sig,self.w) + self.b
                # backward pass
                error = -(y[i] - Y)
                self.w = self.w - self.lr * sig * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            sig = np.array(self.gaussian(X[i], self.centers, self.stds))
            Y = sig.dot(self.w) + self.b
            y_pred.append(Y)
        return np.array(y_pred)

train_x = np.linspace(-10, 10, 500)
train_y = train_x**2
test_x = np.linspace(-3,3,50)
rbfnet = RBF(k=224, epochs=5000)
rbfnet.fit(train_x, train_y)
y_pred = rbfnet.predict(test_x)

plt.plot(train_x[100:400], train_y[100:400], label='true')
plt.plot(test_x, y_pred, label='RBF-Net')
plt.legend()
plt.show()