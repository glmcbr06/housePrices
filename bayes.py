import os
from linReg import cwd
import process as ps
from linReg import trainData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Objective:
# Classify sales price in a number of bins given a few different predictors from the data set with a bayes model

f, ax = plt.subplots()
nBins = 9
bWidth = 72025
ax.hist(trainData['SalePrice'], ec='w', bins=nBins)

mn = min(trainData['SalePrice'])
mx = max(trainData['SalePrice'])

# cls = ps.classify_sale_price(bins=bins, x=755000)
bins = ps.create_bins(mn, width=bWidth, quantity=nBins)
cls = trainData['SalePrice'].apply(lambda x: ps.classify_sale_price(bins, x))
trainData['priceClass'] = cls

trainData = trainData[['TotRmsAbvGrd', 'FullBath', 'TotalBsmtSF', 'GarageArea', 'GrLivArea', 'OverallQual', 'priceClass']]
train = np.array(trainData)
cols = trainData.columns[0:-1]

stats = {'index': [],
         'mean': [],
         'stDev': []}

XsArray = train[:, :-1]
YsArray = train[:, -1:]


for i in range(XsArray.shape[1]):
    mu = np.mean(XsArray[:, i])
    stDev = np.std(XsArray[:, i])
    if stDev == 0:
        stDev = 0.001
    stats['index'].append(i)
    stats['mean'].append(mu)
    stats['stDev'].append(stDev)

statsDf = pd.DataFrame(stats)
statsDf.set_index('index').to_csv(os.path.join(cwd, 'trainingStatsData.csv'))

# print(trainData.corr().sort_values(by='SalePrice'))
# Get the probability of being in a price range
classList = [float(i) for i in range(0, nBins + 1)]
classProb = {'class': classList,
             'values': bins,
             'totalObs': np.zeros(len(classList)),
             'prob': np.zeros(len(classList))}
totalObs = len(trainData)
for idx in range(len(classList)):
    tmp = trainData[trainData['priceClass'] == classProb['class'][idx]]
    n = len(tmp)
    prob = n / totalObs
    classProb['prob'][idx] = prob
    classProb['totalObs'][idx] = n

# Get bin stats
binStats = {}
for c in classList:
    tmp = train[np.where(train[:, -1] == c)]
    xs = tmp[:, :-1]
    for i in range(xs.shape[1]):
        m = np.mean(xs[:, i])
        sd = np.std(xs[:, i])
        xVar = cols[i]
        binStats[c] = (xVar, m, sd)

# Make predictions and get results
cm = np.zeros()
success = 0
failure = 0


exit()


# 1st: Using categorical variables, make a classifier for 14 sales price ranges.
print('classes:', classProb['class'])
print('total Prob:', sum(classProb['prob']))


print(pd.DataFrame(classProb))


def main():


    pass

if __name__ == '__main__':
    main()