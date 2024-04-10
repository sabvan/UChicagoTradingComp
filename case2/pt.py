import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

print(pd.__version__) 

data = pd.read_csv("testdata7.csv", index_col = 0)

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

# TO GRADERS: If the code does not run or 
# takes too long, use the commented out
# Allocator

class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.running_weights_paths = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
    
    class Model:
        def __init__(self):
            self.data = None
            self.model = None
            
        def __build_model(self, input_shape, outputs):
            '''
            Builds and returns the Deep Neural Network that will compute the allocation ratios
            that optimize the Sharpe Ratio of the portfolio
            
            inputs: input_shape - tuple of the input shape, outputs - the number of assets
            returns: a Deep Neural Network model
            '''
            model = Sequential([
                LSTM(64, input_shape=input_shape),
                Flatten(),
                Dense(outputs, activation='softmax')
            ])
            
            

            def sharpe_loss(_, y_pred):
                # make all time-series start at 1
                data = tf.divide(self.data, self.data[0])  
                
                # value of the portfolio after allocations applied
                portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
                portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula
                sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
                
                # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
                #   we can negate Sharpe (the min of a negated function is its max)
                return -sharpe
            
            model.compile(loss=sharpe_loss, optimizer='adam')
            return model
        
        def get_allocations(self, data):
            data = pd.DataFrame(data, columns=['0', '1', '2', '3', '4', '5'])
            '''
            Computes and returns the allocation ratios that optimize the Sharpe over the given data
            input: data - DataFrame of historical closing prices of various assets
            return: the allocations ratios for each of the given assets
            '''
            
            # data with returns
            data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:] ], axis=1)

            data = data.iloc[1:]
            self.data = tf.cast(tf.constant(data), float)
            
            if self.model is None:
                self.model = self.__build_model(data_w_ret.shape, len(data.columns))

            fit_predict_data = data_w_ret[np.newaxis,:]        
            self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=10, shuffle=False)
            return self.model.predict(fit_predict_data)[0]
        
    def calculateWeights(self, pricePath, running_weights_paths, historydays):
        mod = self.Model()
        return mod.get_allocations(pricePath)
    
    
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        ### edit: used np.append to make this work
        self.running_price_paths = np.append(self.running_price_paths, asset_prices)
    
        ### TODO Implement your code here
        #print(self.running_price_paths)
        
        historydays = 100
        
        paths1 = self.running_price_paths[-historydays * 6:].T
        pricePaths = []
        for i in range(historydays):
            sub = paths1[i*6: i*6+6]
            pricePaths.append(sub)
        pricesPaths = pd.DataFrame(pricePaths, columns=['0', '1', '2', '3', '4', '5'])
        
            
        weights = self.calculateWeights(pricePaths, self.running_weights_paths, historydays)
        self.running_weights_paths.append(weights)
        
        return weights





'''
class Allocator():
    def __init__(self, train_data):
        
        Anything data you want to store between days must be stored in a class field
        
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.running_weights_paths = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
        
    def MaximizeSharpeRatioOptmzn(self, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):        
        # define maximization of Sharpe Ratio using principle of duality
        def  f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
            funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T) )
            funcNumer = np.matmul(np.array(MeanReturns),x.T)-RiskFreeRate
            func = -(funcNumer / funcDenomr)
            return func

        #define equality constraint representing fully invested portfolio
        def constraintEq(x):
            A=np.ones(x.shape)
            b=1
            constraintVal = np.matmul(A,x.T)-b 
            return constraintVal
        
        
        #define bounds and other parameters
        xinit=np.repeat(0.33, PortfolioSize)
        cons = ({'type': 'eq', 'fun':constraintEq})
        lb = -1
        ub = 1
        bnds = tuple([(lb,ub) for x in xinit])
        
        #invoke minimize solver
        opt = optimize.minimize (f, x0 = xinit, args = (MeanReturns, CovarReturns,\
                                RiskFreeRate, PortfolioSize), method = 'SLSQP',  \
                                bounds = bnds, constraints = cons, tol = 10**-3)
        
        return opt
        
        
    def StockReturnsComputing(self, StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):        # j: Assets
            for i in range(Rows-1):     # i: Daily Prices
                StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100

        return StockReturn
        
        
    def calculateWeights(self, pricePath, running_weights_paths, historydays):
        arReturns = self.StockReturnsComputing(pricePath, historydays, 6)
    
        portfolioSize = 6
        risk = 4
        meanReturns = np.mean(arReturns, axis = 0)
        covReturns = np.cov(arReturns, rowvar=False)

        #compute daily risk free rate in percentage
        r0 = (np.power((1 + risk),  (1.0 / 251.0)) - 1.0) * 100 

        #initialization
        xOptimal =[]

        #compute maximal Sharpe Ratio and optimal weights
        result = self.MaximizeSharpeRatioOptmzn(meanReturns, covReturns, r0, portfolioSize)

        res = result.x
        res[res > 1] = 1
        res[res < -1] = -1
        
        return res
        
    def allocate_portfolio(self, asset_prices):
        
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        
        ### edit: used np.append to make this work
        self.running_price_paths = np.append(self.running_price_paths, asset_prices)
    
        ### TODO Implement your code here
        #print(self.running_price_paths)
        
        historydays = 40
        
        paths1 = self.running_price_paths[-historydays * 6:].T
        pricePaths = []
        for i in range(historydays):
            sub = paths1[i*6: i*6+6]
            pricePaths.append(sub)
        pricePaths = np.array(pricePaths)
        
        weights = self.calculateWeights(pricePaths, self.running_weights_paths, historydays)
        self.running_weights_paths.append(weights)
        
        return weights


'''

def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()

print(sharpe)



# creds https://github.com/shilewenuw/deep-learning-portfolio-optimization/blob/main/Model.py
# https://github.com/PaiViji/PythonFinance-PortfolioOptimization/blob/master/Lesson6_SharpeRatioOptimization/Lesson6_MainContent.ipynb
