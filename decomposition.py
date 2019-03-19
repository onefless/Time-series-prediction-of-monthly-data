# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:00:31 2019

@author: Francis
"""

#%%

class tt_split(object):
    
    def __init__(self):
        pass
    
    def year_cut(self,data,cut_time):
        
        '''
        Arguments:
            data: df, (n,1), index type must be DatetimeIndex, must be 1 
            cut_time: str,'MMM-YYYY'. Must comply with date form in data.
        Output:
            data: df, (k,1). All the data up to time
        '''    
        return data[time:]
    
    def split(data,seperator=0.7):
        '''
        Arguments:
            data: df, (n,1)
            seperator: float, (0,1). string, 'MMM-YYYY'.
        Output:
            in-sample: df, (k,1). All the data up to time
            out-of-sample: df, (n-k,1).
        '''
        try:
            n = int(data.shape[0]*seperator)
            return data[:n+1],data[n+1:]
        except:
            return data[:seperator],data[seperator:]
    
#t = tt_split
#train,test = t.split(data)

#%%
class decomposition_additive(object):
    '''
    Apply additive decomposition model to evenly spaced monthly time series problem. It allows forecast and vishualisation.
    '''
    
    def __init__(self,order = 2, cycle = None):
        '''
        Arguments:
            order: int, order for polymonial regression
            Cycle: TBA
            metrics: str, 'rmse' TBA
        '''
        self.cycle = cycle
        self.order = order
        
    def seasonal_component(self):
        '''
        Output:
            seasonal_full: Series, (12,). Seasonal component.
        Note: self.seasonal returns only 12 monthly data. self.seasonal_full is the complete seasonal component.
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

        self.seasonal_full = self.original.copy()
        for i in self.seasonal.index:
            self.seasonal_full.loc[self.seasonal_full.index.month == i,self.col_name] =self.seasonal.loc[i]
        
        return self.seasonal_full
    
    def trend_component(self):
        '''
        Output:
            trend: Series, (n,). Trend component. 
        '''
        self.seasonal_component()
        for i in self.seasonal.index:
            self.deseasonal = self.original.subtract(self.seasonal.loc[i],
                                                     level = (self.original.index.month == i))
    
        coef = np.polyfit(np.arange(len(self.deseasonal)),self.original_1d,self.order)
        self.poly_mdl = np.poly1d(coef) #polynominal model
        self.trend = pd.Series(data = self.poly_mdl(np.arange(len(self.deseasonal))),index = self.index)
        
        return self.trend
    
    def fit(self,original):
        '''
        original: df, (n*1). ONLY MONTHLY DATA. Must not have missing months. Index must be Timestamp
        '''
        
        self.original = original
        self.original_1d = original.values.ravel()
        self.index = original.index
        self.col_name = original.columns[0]
        self.trend_component()
        self.residual = pd.Series(data = (self.original.values.ravel() - self.seasonal_full.values.ravel() - self.trend.values.ravel()),index = self.index)
        rmse_training = np.sqrt((((self.trend.values.ravel()+self.seasonal_full.values.ravel()) - self.original_1d) ** 2).mean())
        print('The training RMSE is:{}'.format(rmse_training))
        
    def test(self,test_data, metrics = 'rmse'):
        '''
        Argument:
            test_Data: df, (n*1). Test date must be continueous with training.
        Output:
            prediction_test: Series, (n,)
        '''
        
        self.test_data = test_data
        self.test_index = test_data.index
        self.metrics = metrics.lower()

        #seasonal test
        if self.test_data.columns[0] != self.col_name:
            self.test_data.rename(index=str,columns={self.test_data.columns[0]:self.col_name})
        self.seasonal_full_test = self.test_data.copy()
        for i in self.seasonal.index:
            self.seasonal_full_test.loc[self.seasonal_full_test.index.month == i, self.col_name] = self.seasonal.loc[i]
        #deseasonal test
        self.deseasonal_test = pd.Series(data = (self.test_data.values - self.seasonal_full_test.values).ravel(),index = self.test_index)

        #trend test
        poly_values = self.poly_mdl(np.arange(len(self.test_data)+len(self.original)))
        self.trend_test = pd.Series(data = poly_values[-self.test_data.shape[0]:],index = self.test_index)
        self.prediction_test = self.trend_test+self.seasonal_full_test.iloc[:,0]
        self.residual_test = pd.Series(data = (self.test_data.values.ravel() - self.seasonal_full_test.values.ravel() - self.trend_test.values.ravel()),index = self.test_index)
        rmse_test = np.sqrt(((self.prediction_test.values.ravel() - self.test_data.values.ravel()) ** 2).mean())
        print('The test RMSE is:{}'.format(rmse_test))
        return self.prediction_test
    
    def predict(self,months = 36):
        '''
        Arguments:
            month: int, how many months to be predicted after original.
        Output:
            prediction: Series, (n,)
        '''
        from dateutil.relativedelta import relativedelta
        new_index = pd.date_range(self.index[-1]+relativedelta(months = 1),self.index[-1]+relativedelta(months = months),freq='MS')
        self.seasonal_full_pred = pd.Series(data = np.zeros(months),index = new_index)
        for i in self.seasonal.index:
            self.seasonal_full_pred.loc[self.seasonal_full_pred.index.month == i] = self.seasonal.loc[i]
        poly_values = self.poly_mdl(np.arange(len(new_index)+len(self.original)))
        self.trend_pred = pd.Series(data = poly_values[-months:],index = new_index)
        self.prediction = self.trend_pred + self.seasonal_full_pred
        self.prediction.rename(self.col_name,inplace = True)
        return self.prediction
        
    def plot_components(self,data = 'training'):
        '''
        ***Credit: modified from the code by Marcel Scharth (Github: https://github.com/mscharth). 
        Plot Plot original data, seasonal component, trend component and residual, also predicted values for 'test'.
        Arguments:
            data: 'training' or 'test', str. Which data to be plotted. 
        '''
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        if data == 'training':
            fig, ax = plt.subplots(4, sharex=True)
            self.original.plot(ax=ax[0],color='b', linestyle='-')
            ax[0].set_title('Original')
            
            self.trend.plot(ax=ax[1], color='r', linestyle='-')
            ax[1].set_title('Trend line (Order = {}.)'.format(self.order))
            
            self.seasonal_full.plot(ax=ax[2], color='g', linestyle='-')
            ax[2].set_title('Seasonal component')
            
            self.residual.plot(ax=ax[3], color='k', linestyle='-')
            ax[3].set_title('Residual plot')
        if data == 'test':
            fig, ax = plt.subplots(4, sharex=True)
            self.test_data.plot(ax=ax[0],color='b', linestyle='-')
            self.prediction_test.plot(ax=ax[0],color='r', linestyle='-')
            ax[0].set_title('Test data')
            
            self.trend_test.plot(ax=ax[1], color='r', linestyle='-')
            ax[1].set_title('Trend line (Order = {}.)'.format(self.order))
            
            self.seasonal_full_test.plot(ax=ax[2], color='g', linestyle='-')
            ax[2].set_title('Seasonal component')
            
            self.residual_test.plot(ax=ax[3], color='k', linestyle='-')
            ax[3].set_title('Residual plot')
            
    def plot_test(self,data = 'test'):
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        if data == 'test':
            self.test_data.plot(color='b',linestyle='-')
            self.prediction_test.plot(color='r',linestyle='-')
        if data == 'training':
            self.original.plot(color='b',linestyle='-')
            training_pred = pd.Series(data = (self.trend.values.ravel() + self.seasonal_full.values.ravel()),index = self.index)
            training_pred.plot(color='r',linestyle='-')
            
#%%

class decomposition_multiplicative(object):
    
    def __init__(self,order = 2, cycle = None):
        '''
        Arguments:
            order: int, order for polymonial regression
            Cycle: TBA
            metrics: str, 'rmse' TBA
        '''
        self.cycle = cycle
        self.order = order
    
    def seasonal_component(self):
        '''
        Output:
            seasonal_full: Series, (12,). Seasonal component.
        Note: self.seasonal returns only 12 monthly data. self.seasonal_full is the complete seasonal component.
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

        detrended = self.original.iloc[:,0]/trend
        self.seasonal = detrended.groupby(by = detrended.index.month).mean()

        self.seasonal_full = self.original.copy()
        for i in self.seasonal.index:
            self.seasonal_full.loc[self.seasonal_full.index.month == i,self.col_name] =self.seasonal.loc[i]
        
        return self.seasonal_full
    
    def trend_component(self):
        '''
        Output:
            trend: Series, (n,). Trend component. 
        '''
        self.seasonal_component()
        for i in self.seasonal.index:
            self.deseasonal = self.original.div(self.seasonal.loc[i],
                                                     level = (self.original.index.month == i))
    
        coef = np.polyfit(np.arange(len(self.deseasonal)),self.original_1d,self.order)
        self.poly_mdl = np.poly1d(coef) #polynominal model
        self.trend = pd.Series(data = self.poly_mdl(np.arange(len(self.deseasonal))),index = self.index)
        
        return self.trend
    
    def fit(self,original):
        '''
        original: df, (n*1). ONLY MONTHLY DATA. Must not have missing months. Index must be Timestamp
        '''
        self.original = original
        self.original_1d = original.values.ravel()
        self.index = original.index
        self.col_name = original.columns[0]
        self.trend_component()
        self.residual = pd.Series(data = (self.original.values.ravel() / self.seasonal_full.values.ravel() / self.trend.values.ravel()),index = self.index)
        rmse_training = np.sqrt((((self.trend.values.ravel()*self.seasonal_full.values.ravel()) - self.original_1d) ** 2).mean())
        print('The training RMSE is:{}'.format(rmse_training))
    
    def test(self,test_data, metrics = 'rmse'):
        '''
        Argument:
            test_Data: df, (n*1). Test date must be continueous with training.
        Output:
            prediction_test: Series, (n,)
        '''
        
        self.test_data = test_data
        self.test_index = test_data.index
        self.metrics = metrics.lower()

        #seasonal test
        if self.test_data.columns[0] != self.col_name:
            self.test_data.rename(index=str,columns={self.test_data.columns[0]:self.col_name})
        self.seasonal_full_test = self.test_data.copy()
        for i in self.seasonal.index:
            self.seasonal_full_test.loc[self.seasonal_full_test.index.month == i, self.col_name] = self.seasonal.loc[i]
        #deseasonal test
        self.deseasonal_test = pd.Series(data = (self.test_data.values.ravel() / self.seasonal_full_test.values.ravel()),index = self.test_index)

        #trend test
        poly_values = self.poly_mdl(np.arange(len(self.test_data)+len(self.original)))
        self.trend_test = pd.Series(data = poly_values[-self.test_data.shape[0]:],index = self.test_index)
        self.prediction_test = self.trend_test.values.ravel() * self.seasonal_full_test.values.ravel()
        self.prediction_test = pd.Series(data = self.prediction_test, index = self.test_index)
        self.residual_test = pd.Series(data = (self.test_data.values.ravel() / self.seasonal_full_test.values.ravel() - self.trend_test.values.ravel()),index = self.test_index)
        rmse_test = np.sqrt(((self.prediction_test.values.ravel() - self.test_data.values.ravel()) ** 2).mean())
        print('The test RMSE is:{}'.format(rmse_test))
        return self.prediction_test
    
    def predict(self,months = 36):
        '''
        Arguments:
            month: int, how many months to be predicted after original.
        Output:
            prediction: Series, (n,)
        '''
        from dateutil.relativedelta import relativedelta
        new_index = pd.date_range(self.index[-1]+relativedelta(months = 1),self.index[-1]+relativedelta(months = months),freq='MS')
        self.seasonal_full_pred = pd.Series(data = np.zeros(months),index = new_index)
        for i in self.seasonal.index:
            self.seasonal_full_pred.loc[self.seasonal_full_pred.index.month == i] = self.seasonal.loc[i]
        poly_values = self.poly_mdl(np.arange(len(new_index)+len(self.original)))
        self.trend_pred = pd.Series(data = poly_values[-months:],index = new_index)
        self.prediction = self.trend_pred * self.seasonal_full_pred
        self.prediction.rename(self.col_name,inplace = True)
        return self.prediction
    
    def plot_components(self,data = 'training'):
        '''
        ***Credit: modified from the code by Marcel Scharth (Github: https://github.com/mscharth). 
        Plot Plot original data, seasonal component, trend component and residual, also predicted values for 'test'.
        Arguments:
            data: 'training' or 'test', str. Which data to be plotted. 
        '''
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        if data == 'training':
            fig, ax = plt.subplots(4, sharex=True)
            self.original.plot(ax=ax[0],color='b', linestyle='-')
            ax[0].set_title('Original')
            
            self.trend.plot(ax=ax[1], color='r', linestyle='-')
            ax[1].set_title('Trend line (Order = {}.)'.format(self.order))
            
            self.seasonal_full.plot(ax=ax[2], color='g', linestyle='-')
            ax[2].set_title('Seasonal component')
            
            self.residual.plot(ax=ax[3], color='k', linestyle='-')
            ax[3].set_title('Residual plot')
        if data == 'test':
            fig, ax = plt.subplots(4, sharex=True)
            self.test_data.plot(ax=ax[0],color='b', linestyle='-')
            self.prediction_test.plot(ax=ax[0],color='r', linestyle='-')
            ax[0].set_title('Test data')
            
            self.trend_test.plot(ax=ax[1], color='r', linestyle='-')
            ax[1].set_title('Trend line (Order = {}.)'.format(self.order))
            
            self.seasonal_full_test.plot(ax=ax[2], color='g', linestyle='-')
            ax[2].set_title('Seasonal component')
            
            self.residual_test.plot(ax=ax[3], color='k', linestyle='-')
            ax[3].set_title('Residual plot')
    
    def plot_test(self,data = 'test'):
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        if data == 'test':
            self.test_data.plot(color='b',linestyle='-')
            self.prediction_test.plot(color='r',linestyle='-')
        if data == 'training':
            self.original.plot(color='b',linestyle='-')
            training_pred = pd.Series(data = (self.trend.values.ravel() + self.seasonal_full.values.ravel()),index = self.index)
            training_pred.plot(color='r',linestyle='-')
