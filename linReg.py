import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import RegscorePy as rsp
import statsmodels.api as sm
import process as ps

cwd = os.getcwd()
trainData = pd.read_csv(os.path.join(cwd, 'train.csv'))
corr = trainData.corr()

# TRAIN THE MODEL
# y = np.array(trainData['SalePrice'])
# X = np.array(trainData[['OverallQual', 'LotArea', 'GrLivArea']])
y = trainData['SalePrice']
X = trainData[['OverallQual', 'LotArea', 'GrLivArea']]
# X = sm.add_constant(X)

linReg = LinearRegression(fit_intercept=True).fit(X, y)
# olsModel = sm.OLS(y, X).fit()
# print(olsModel.rsquared, olsModel.rsquared_adj)

rSq = linReg.score(X, y)
n = len(y)
p = len(X.columns)

adjRsq = ps.get_adjRsq(X, y, rSq)

print('rSq:', rSq, 'adjRsq:', adjRsq)
exit()

yHat = linReg.predict(X)
res = {'pred': yHat,
       'actual': y,
       'err': (yHat - y) ** 2}

f, ax = plt.subplots(figsize=(10, 10))
ax.plot(X[:, 2], yHat)
plt.show()
exit()

resDF = pd.DataFrame(res)

print(resDF)

b0 = linReg.intercept_
coeff = linReg.coef_
print('intercept', b0)
print('slope:', coeff)


def main():


    lm = b0 + np.sum(coeff * X, axis=1)
    print(lm)
    exit()
    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(X, lm)
    ax.scatter(X, y)


    plt.show()
    print(lm)
    exit()
    cols = trainData.columns
    print(cols)

    ax = sns.heatmap(corr)
    plt.tight_layout()
    plt.show()
    exit()

    # TEST THE MODEL

if __name__ == '__main__':
    main()
