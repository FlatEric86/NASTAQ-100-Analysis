import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import random as rand
import math

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

plt.style.use('seaborn')



time_scaler = 0.0001    


### nasdaq100
df_nas = pd.read_csv('./nasdaq_100.csv', usecols=['Date', 'Close'])

df_nas['Date'] = pd.to_datetime(df_nas['Date'])
df_nas = df_nas.set_index('Date')

#df_nas['Close'] = df_nas['Close']/df_nas['Close'].max()

# we add here a further column that represent the time ordinate
# we do not use interger instead we transform the integers to decimal values
# we do so because the timeindex appears as exponent that may induce numerical problems 
# due to too large numbers during gradient decent optimization
df_nas['T'] = [i*time_scaler for i in range(len(df_nas))]





################################# Exponetial Regression Model ###################################

# we define here the foreward model in form of the standard exponential function as particular solution
# the differential equation df(t)/dt = lambda*f(t)
# of type f(t) = f(t=0)*exp(lambda*t) with two coefficients A_0 that represents the funtion value at t=0
# as well as lambda as the grow factor
def forward_model(t, A_0, l):
    return A_0*np.exp(l*t)


#### model_1 -> precorona

# we fit here the model on the observation data represent by the NASDAQ Stock 
coef, cov = scipy.optimize.curve_fit(forward_model, df_nas['T'].loc[:'2020-01-01'].values, df_nas['Close'].loc[:'2020-01-01'].values)


A_0_m_1, lam_m_1 = coef

# add model result to DataFrame
df_nas['model_1'] = [forward_model(t, A_0_m_1, lam_m_1) for t in df_nas['T'].values]


#### model_2 -> recent data

# we fit here the model on the observation data represent by the NASDAQ Stock 
coef, cov = scipy.optimize.curve_fit(forward_model, df_nas['T'].values, df_nas['Close'].values)


A_0_m_2, lam_m_2 = coef

# add model result to DataFrame
df_nas['model_2'] = [forward_model(t, A_0_m_2, lam_m_2) for t in df_nas['T'].values]



################################# Hyperbolic Regression Model ###################################


def hyp_forward_model(x, A, B):  
    return A /((B - x)**2) 


# we fit here the model on the observation data represent by the NASDAQ Stock 
coef, cov = scipy.optimize.curve_fit(hyp_forward_model, df_nas['T'].values, df_nas['Close'].values)


A, B = coef


# add model result to DataFrame
df_nas['model_3'] = [hyp_forward_model(t, A, B) for t in df_nas['T'].values]




##################################### Forecast ####################################################

date = pd.date_range(start=df_nas.index.values[0], end='1/01/2025')

future = pd.DataFrame({ 'Date': date}) 
future['Date'] = pd.to_datetime(future['Date'])
future = future.set_index('Date')
future['T'] = [i*0.000069 for i in range(len(future))]


future['m_1'] = [forward_model(x, A_0_m_1, lam_m_1) for x in future['T'].values]
future['m_2'] = [forward_model(x, A_0_m_2, lam_m_2) for x in future['T'].values]
future['m_3'] = [hyp_forward_model(x, A, B) for x in future['T'].values]






########################################### Plot Section #########################################


fig, ax = plt.subplots(figsize=(16,9))

# plot model forecast

k_size = 365

# # plot nasdaq100 and exponentian
df_nas['Close'].plot(
    color='royalblue', 
    label='NASDAQ100', 
    ax=ax,
    alpha=0.6    
    )
    
df_nas['Close'].rolling(k_size, center=True).mean().plot(
        color='red', 
        label=f'rolling mean @ kernel size = {k_size} timesteps', 
        ax=ax,
        alpha=0.6
        )
        
df_nas['model_1'].plot(
    color='salmon', 
    label='$f(t) = A_0 e ^{\lambda t}$ \nwith $A_0 = $' + str(round(A_0_m_1, 2)) + '; $\lambda = $' + str(round(lam_m_1, 2)),
    ax=ax,
    alpha=0.6   
    )
df_nas['model_2'].plot(
    color='gold', 
    
    label='$f(t) = A_0 e ^{\lambda t}$ \nwith $A_0 = $' + str(round(A_0_m_2, 2)) + '; $\lambda = $' + str(round(lam_m_2, 2)), 
    ax=ax,
    alpha=0.6
    )
    
df_nas['model_3'].plot(
    color='limegreen', 
    label='$f(t) = \dfrac{C_1}{(C_2 - t)^2}$ \nwith $C_1 = $' + str(round(A, 2)) + '; $C_2 = $' + str(round(B, 2)), 
    alpha=0.6,
    ax=ax
    )





ax.plot(
    future['m_1'].loc[df_nas.index.values[-1]::], 
    linewidth=.8, 
    linestyle=':', 
    color='grey'
    )
    
ax.plot(
    future['m_2'].loc[df_nas.index.values[-1]::], 
    linewidth=.8, 
    linestyle=':', 
    color='grey'
    )

ax.plot(
    future['m_3'].loc[df_nas.index.values[-1]::], 
    linewidth=.8, 
    linestyle=':', 
    color='grey'
    )

#plot a horizontal line represents the time start after .com crash
ax.plot([df_nas.loc['2003-01-01':'2003-01-02'].index, df_nas.loc['2003-01-01':'2003-01-02'].index], [df_nas['Close'].min(), df_nas['Close'].max()], color='k', linewidth=.8)
ax.text(df_nas.loc['2003-01-01':'2003-01-02'].index, df_nas['Close'].max(), 'after dot com')


ax.set_title('NASTAQ-100 Analysis regarding its growth behavior')

ax.set_ylabel('Value')

ax.legend()
plt.show()





