# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:40:44 2018

@author: Francis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

parser = lambda dates: pd.datetime.strptime(dates,'%b-%Y')
data = pd.read_csv('Australian_tourists_total_arrivals.csv',parse_dates=[0],index_col = 'Time' ,date_parser = parser)

#%%
def year_cut(data,time):
    '''
    data: df, (n,1), index type must be DatetimeIndex, must be 1 
    time: str,'MMM-YYYY'. Must comply with date form in data.
    '''    
    return data[time:]


#plt.figure(figsize=(20, 10))
#plt.plot(data['Time'],data['Total arrivals'])
#plt.title('Total movement to australia')
#plt.xlabel('Month')

def tt_split(data,seperator=0.7):
    '''
    Input:
        data: df, (n,1)
        seperator: float, (0,1). string, 'MMM-YYYY'.
    Output:
        trainin, test: df
    '''
    try:
        n = int(data.shape[0]*seperator)
        return data[:n+1],data[n+1:]
    except:
        return data[:seperator],data[seperator:]
#%%
data_ten_y = year_cut(data,'Jan-2004')
train,test = tt_split(data_ten_y)

plt.figure()
plt.plot(train.index,train)
plt.show()

#%% Decomposition
from statsmodels.tsa import seasonal
multiplicative = seasonal.seasonal_decompose(train, model='multiplicative') 

def plot_decomposed(orignal,decompose_result):
    fig, ax = plt.subplots(4, sharex=True)
    fig.set_size_inches(15, 10)
    
    orignal.plot(ax=ax[0], color='b', linestyle='-')
    ax[0].set_title('Original')
    
    pd.Series(data=decompose_result.trend.values.ravel(), index=decompose_result.trend.index).plot(ax=ax[1], color='r', linestyle='-')
    ax[1].set_title('Trend line')#MA-2
    
    pd.Series(data=decompose_result.seasonal.values.ravel(), index=orignal.index).plot(ax=ax[2], color='g', linestyle='-')
    ax[2].set_title('Seasonal component')
    
    pd.Series(data=decompose_result.resid.values.ravel(), index=orignal.index[:]).plot(ax=ax[3], color='k', linestyle='-')
    ax[3].set_title('Residual plot')

plot_decomposed(train,multiplicative)
#%%
class decomposition_additive(object):
    '''
    original: df, n*1  ONLY MONTHLY DATA
    '''
    def __init__(self,original,order = 2, cycle = None):
        self.original = original
        self.original_1d = original.values.ravel()
        self.index = original.index
        self.cycle = cycle
        self.order = order
        self.col_name = original.columns[0]
#    def show_pic(self):
#        try:
#            plt
#        except:
#            import matplotlib.pyplot as plt
#            
#        plt.figure(figsize=(15, 10))
#        plt.plot(self.index,self.original_1d)
#        plt.ylabel(self.original.columns[0])
#        plt.show()
        
    def seasonal_component(self):
        '''
        Note: only 12 monthly data is return. Does not cover the whole spectrum.
        '''
        try:
            np
        except:
            import numpy as np
        try:
            pd
        except:
            import pandas as pd
        
        coef = np.polyfit(np.arange(len(self.original)),self.original_1d,self.order)
        poly_mdl = np.poly1d(coef)
        trend = pd.Series(data = poly_mdl(np.arange(len(self.original))),index = self.index)

        self.detrended = self.original.iloc[:,0] - trend
        self.seasonal = self.detrended.groupby(by = self.detrended.index.month).mean()

        for i in self.seasonal.index:
            self.seasonal_full = self.original.subtract(-self.seasonal.loc[i],
                                                     level = (self.original.index.month == i))
        self.seasonal_full = pd.Series(data = (self.seasonal_full.iloc[:,0].values - test.iloc[:,0].values),index = self.index)

        return self.seasonal
    
    def trend_component(self):
        self.seasonal_component()
        for i in self.seasonal.index:
            self.deseasonal = self.original.subtract(self.seasonal.loc[i],
                                                     level = (self.original.index.month == i))
    
#        plt.figure(figsize=(15, 10))
#        plt.plot(self.index,self.deseasonal)
#        plt.ylabel(self.deseasonal.columns[0])
#        plt.show()
        
        coef = np.polyfit(np.arange(len(self.deseasonal)),self.original_1d,self.order)
        poly_mdl = np.poly1d(coef)
        self.trend = pd.Series(data = poly_mdl(np.arange(len(self.deseasonal))),index = self.index)
        
        return self.trend
    
#    def cycle(self): # add for cycle component
    
#    def resid(self):
#        self.trend_component()
#        self.residual = self.deseasonal - self.trend
        
    
    
decomp = decomposition_additive(train)
trend = decomp.trend_component()
seasonal = decomp.seasonal
detrended = decomp.detrended
seasonal_full = decomp.seasonal_full


plt.figure(figsize=(15,10))
plt.plot(trend.index,trend)
plt.plot(trend.index,seasonal_full)
plt.show()

#%%
class foo(object):
    def __init__(self,num):
        self.num = num
        self.num2 = num*10
        
    def a(self):
        self.result = self.num2
    
    def b(self):
        self.a()
        print(self.result+100)

mdl = foo(2)
print(mdl.a(3))
        
seasonal_test = train.groupby(by = train.index.month).mean()
test = pd.DataFrame(data = np.ones([len(train)]),index = train.index,columns = ['Total arrivals'])
test2 = pd.DataFrame(data = np.ones([len(train),12]),index = train.index)
for i in seasonal_test.index[:6]:
#    test = test.loc[test.index.month == i,:] +seasonal_test.iloc[i,:]
    j=0
    test2.iloc[:,j].subtract(seasonal_test['Total arrivals'].loc[i],
                                  level = (test.index.month == i))
    seasonal_full = test.subtract(seasonal_test['Total arrivals'].loc[i],
                                  level = (test.index.month == i))
    j+=1

for i in seasonal_test.index:
    deseasonal = train.subtract(-seasonal_test['Total arrivals'].loc[i],
                                                     level = (train.index.month == i)) 
deseasonal = deseasonal.values- train.values
plt.plot(np.arange(len(deseasonal)),deseasonal)
#%% S-ARIMA
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX()
