import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as web
import yfinance as yf

# interval = 'm'
# tickers = ['AAPL']
# risk_free = 0.0402
# end = dt.datetime.now()
# start = end - dt.timedelta(weeks=52*N)
# start = dt.datetime(2017,4,29)
# end = dt.datetime(2022,8,6)

### Import price data
# Get adjusted close price
# Use pandas_datareader
def get_data(tickers, start, end, interval, OHLC='Adj Close', market=True):
    if market == True:
        tickers.insert(0,'^GSPC')
    
    # compounding frequency per annum
    frequency = {'d':252, 'w':52, 'm': 12}
    # monthly frequency
    frequency = frequency[interval]

    df = pd.DataFrame()
    for t in tickers:
        df[t] = web.get_data_yahoo(t, start, end, interval=interval)[OHLC]
    df.dropna(inplace=True)
    
    return df

# Use yfinance 
def get_data_yf(tickers, start, end, interval, OHLC='Adj Close', market=True):
    if market == True:
        tickers.insert(0,'^GSPC')
    
    # compounding frequency per annum
    frequency = {'1d':252, '1wk':52, '1mo': 12}
    # monthly frequency
    frequency = frequency[interval]

    df = pd.DataFrame()
    for t in tickers:
        df[t] = yf.download(t, start, end, interval=interval)[OHLC]
    df.dropna(inplace=True)
    
    return df

# Get total price with OHLC using pandas_datareader
def get_OHLC(tickers, start, end, interval,OHLC='Adj Close'):
    tickers.insert(0,'^GSPC')
    # compounding frequency per annum
    frequency = {'d':252, 'w':52, 'm': 12}
    # monthly frequency
    frequency = frequency[interval]

    open_df = get_data(tickers, start, end, interval,OHLC='Open')
    high_df = get_data(tickers, start, end, interval,OHLC='High')
    low_df = get_data(tickers, start, end, interval,OHLC='Low')
    adjc_df = get_data(tickers, start, end, interval)
    
    OHLC = pd.concat([open_df,
               high_df,
               low_df, 
               adjc_df], join="inner").sort_values(by='Date')
    return OHLC

# Get total price with OHLC using yfinance
def get_OHLC_yf(tickers, start, end, interval,OHLC='Adj Close', market=True):
    if market == True:
        tickers.insert(0,'^GSPC')

    # compounding frequency per annum
    frequency = {'1d':252, '1wk':52, '1mo': 12}
    # monthly frequency
    frequency = frequency[interval]

    open_df = get_data(tickers, start, end, interval,OHLC='Open')
    high_df = get_data(tickers, start, end, interval,OHLC='High')
    low_df = get_data(tickers, start, end, interval,OHLC='Low')
    adjc_df = get_data(tickers, start, end, interval)
    
    OHLC = pd.concat([open_df,
               high_df,
               low_df, 
               adjc_df], join="inner").sort_values(by='Date')
    return OHLC

### Two returns calculation methods
# Calculate Simple Returns of the stocks
# (Missing first observation)
def simp_ret(stocks_price):
    simp_ret = (stocks_price/stocks_price.shift(1) - 1)[1:]
    
    return simp_ret

# Calculate Log Returns of the stocks
def log_ret(stocks_price):
    log_return = np.log(stocks_price/stocks_price.shift(1))[1:]
    
    return log_return

### Annualizing returns above
# Annualized simple return
def annual_simpret(simp_return, frequency):
    # Gross returns
    grossret = simp_return + 1 
    # Periodic geomean return
    geomret = np.prod(grossret)**(1/len(grossret))
    # Periodic geomean return compounded to frequency per annum
    annual_simpret = geomret**frequency - 1
    
    return annual_simpret

# Annualized log return
def annual_logret(log_return, frequency):
    # Arithmetic mean of log returns compounded to annualize
    annual_logret = log_return.mean()*frequency
    
    return annual_logret


### Beta of stocks
# risk-free rate of annualized risk-free rate
# Difference from including risk_free is negigible, using risk-free=0 is fine
def get_beta(stock_returns, risk_free=0):
    
    # De-annualize risk-free rate
    periodic_rf = (1+risk_free)**(1/frequency)-1
    
    tickers = stock_returns.columns
    Beta_df = pd.DataFrame(columns = ['ticker','Beta'])
    stock_returns = stock_returns - periodic_rf

    for stock in range(len(tickers)):
        stock_mkt_cov = stock_returns.cov().loc['^GSPC',tickers[stock]]
        mkt_var = stock_returns['^GSPC'].var()
        Beta = stock_mkt_cov/mkt_var
        Beta_df = Beta_df.append({'ticker':tickers[stock],'Beta':Beta },ignore_index=True)
        
    return Beta_df

# Mkt_ret_simp = annual_simpret(simpret['^GSPC'], frequency) 
# actual_simpret = annual_simpret(simpret, frequency)

# risk-free, Mkt, actual stock returns must be annualized returns
def CAPM(beta_df, risk_free, Mkt_ret, actual_ret): # expected returns in annualized terms
    stocks = list(beta_df['ticker'])[1:] # tickers except for S&P500 index
    ER_df = pd.DataFrame(columns = ['ticker','CAPM ER'])
    
    for stock in stocks:
        beta = float(beta_df[beta_df['ticker']==stock]['Beta'])
        expected_return = risk_free + beta*(Mkt_ret - risk_free)
        actual_return = actual_ret[stock]
        return_gap = actual_return - expected_return
        ER_df = ER_df.append({'ticker':stock,
                              'Beta': beta,
                              'CAPM ER':expected_return, 
                              'Actual Return': actual_return,
                              'Return Gap': return_gap},
                             ignore_index=True)
        
    
    # if CAPM Expected Returns < actual returns, undervalued
    ER_df['Undervalued'] = ER_df['CAPM ER'] < ER_df['Actual Return']
    ER_df = ER_df.sort_values(by=['Return Gap'], ascending=False)
    return ER_df

def rolling_beta(stock_df, ticker, beta_window, ma_window):
    ticker = ticker
    stock_P = get_data(['^GSPC',ticker], start, end, interval)[:-1]
    
    stock_returns = simp_ret(stock_P[['^GSPC',ticker]]).rolling(beta_window)
    stock_mkt_cov = stock_returns.cov().iloc[1::2, :].drop(columns=[ticker])#.loc['^GSPC','AAPL']
    stock_mkt_cov.index = stock_mkt_cov.index.droplevel(level=1)
    stock_mkt_cov = stock_mkt_cov.rename(columns={'^GSPC':f'{ticker} Beta'})

    mkt_var = stock_returns.cov().iloc[::2, :].drop(columns=[ticker])
    mkt_var.index =mkt_var.index.droplevel(level=1)
    Beta = stock_mkt_cov.iloc[:,0]/mkt_var.iloc[:,0]

    Beta_ma = Beta.rolling(ma_window).mean()
    return Beta, Beta_ma


# Get Historical Beta, one datapoint per each stock
def get_beta_yf(stock_returns):
    tickers = stock_returns.columns
    Beta_df = pd.DataFrame(columns = ['ticker','Beta'])

    for stock in range(len(tickers)):
        stock_mkt_cov = stock_returns.cov().loc['^GSPC',tickers[stock]]
        mkt_var = stock_returns['^GSPC'].var()
        Beta = stock_mkt_cov/mkt_var
        Beta_df = Beta_df.append({'ticker':tickers[stock],'Beta':Beta },ignore_index=True)
        
    return Beta_df

# Get Time series of rolling Betas per each stock
def rolling_beta_yf(stock_df, ticker, beta_window, ma_window): # takes in single ticker at a time

    stock_returns = simp_ret(stock_df[['^GSPC',ticker]]).rolling(beta_window)
    stock_mkt_cov = stock_returns.cov().iloc[1::2].drop(columns=[ticker]).dropna()
    stock_mkt_cov.index = stock_mkt_cov.index.droplevel(level=1)

    mkt_var = stock_returns.cov().iloc[::2].drop(columns=[ticker]).dropna()
    mkt_var.index =mkt_var.index.droplevel(level=1)
    Beta = stock_mkt_cov/mkt_var
    Beta = Beta.rename(columns={'^GSPC':ticker})
    
    Beta_ma = Beta.rolling(ma_window).mean().dropna()
    
    return Beta, Beta_ma