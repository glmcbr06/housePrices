import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
import numpy as np
import process as ps
from math import sqrt
import scipy.stats as sc
import statsmodels.api as sm
from sklearn import preprocessing
import scipy.stats as stats
from math import e


cwd = os.getcwd()
trainData = pd.read_csv(os.path.join(cwd, 'train.csv'))
testData = pd.read_csv(os.path.join(cwd, 'test.csv'))

corr = trainData.corr()
trainData['SalePrice'] = np.log(trainData['SalePrice'])

# CORRELATION PLOT
# fname = os.path.join(cwd, 'heatmap.png')
# f, ax1 = plt.subplots(figsize=(12, 12))
# sns.heatmap(corr, linewidths=.01, annot=True, fmt='.2f', annot_kws={'size': 6})
# plt.tight_layout()
# f.savefig(fname)

# OBJECTIVE: Select a linear regression model that accurately predicts sale price for the dataset.

# TRAIN THE MODEL
# y = trainData['SalePrice']
# standard = preprocessing.StandardScaler
X = np.log(trainData[['OverallQual', 'LotArea', 'GrLivArea', 'YearBuilt']])
testX = np.log(testData[['OverallQual', 'LotArea', 'GrLivArea', 'YearBuilt']])
# X = trainData[['OverallQual', 'LotArea', 'GrLivArea', 'OverallCond']]

# X = trainData.loc[:, trainData.columns != 'SalePrice']

linReg = LinearRegression(fit_intercept=True).fit(X, trainData['SalePrice'])

rSq = linReg.score(X, trainData['SalePrice'])
n = len(trainData)
p = len(X.columns)
adjRsq = ps.get_adjRsq(X, trainData['SalePrice'], rSq)

trainData['yPred'] = linReg.predict(X)
testData['yPred'] = e ** linReg.predict(testX)
print(np.mean(trainData['yPred']), np.std(trainData['yPred']))
print(np.mean(testData['yPred']), np.std(testData['yPred']))
print(min(testData['yPred']), max(testData['yPred']))
fname = os.path.join(cwd, 'testDataPredictions.csv')
df = pd.DataFrame({'Id': testData['Id'],
                   'SalePrice': testData['yPred']}).set_index('Id')
df.to_csv(fname)

fname = os.path.join(cwd, 'testPredDistributions.png')
f, ax3 = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax3[0].hist(testData['yPred'], ec='w')
testNorm = testData['yPred'].apply(lambda x: ps.normalize(x, np.mean(testData['yPred']), np.std(testData['yPred'])))
ax3[1].hist(testNorm, ec='w')
f.savefig(fname)


trainData['residual'] = trainData['yPred'] - trainData['SalePrice']

fname = os.path.join(cwd, 'residualVsFit.png')
statsDf = pd.DataFrame({'F stat': f_regression(X, trainData['SalePrice'])[0],
                        'P value': f_regression(X, trainData['SalePrice'])[1]})


trainData['norm'] = trainData['SalePrice'].apply(lambda x: ps.normalize(x, trainData['SalePrice'].mean(), trainData['SalePrice'].std()))
trainData['pValue'] = trainData['norm'].apply(lambda x: stats.norm.cdf(x))

f, ax2 = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
ax2[0, 0].scatter(trainData['yPred'], trainData['residual'])
ax2[0, 0].hlines(y=0, linestyles='dashed', color='black', xmax=max(trainData['yPred']), xmin=min(trainData['yPred']))
ax2[0, 0].set_xlabel('Fitted Y')
ax2[0, 0].set_ylabel('Residual')
ax2[0, 0].set_title('Residual Vs. Fitted Y')
# ax2[1].scatter(trainData['residual'], randNorm)
# stats.probplot(y, dist='norm', plot=ax2[1])

stats.probplot(trainData['pValue'], dist='norm', plot=ax2[0, 1])

sm.qqplot(trainData['norm'], ax=ax2[1, 0], line='45')
ax2[1, 1].hist(trainData['SalePrice'], ec='w')
ax2[2, 1].hist(trainData['norm'], ec='w')

ax2[2, 0].hist(trainData['residual'], ec='w')
print(min(trainData['norm']), max(trainData['norm']))
print(trainData['residual'])

# ax2[1].scatter(trainData['SalePrice'], trainData['yPred'])
plt.tight_layout()
f.savefig(fname)


print('rSq: {}, adjRsq: {}'.format(rSq, adjRsq))
print('RMSE: {}'.format(sqrt(mean_squared_error(y_true=trainData['SalePrice'], y_pred=trainData['yPred']))))

res = {'pred': trainData['yPred'],
       'actual': trainData['SalePrice'],
       'err': (trainData['yPred'] - trainData['SalePrice']) ** 2}

resDF = pd.DataFrame(res)
b0 = linReg.intercept_
coeff = linReg.coef_
print('intercept: {}, slope: {}'.format(b0, coeff))

exit()
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
