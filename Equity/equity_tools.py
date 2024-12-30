import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

index_component = {
        'sp500': ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0),
        'nasdaq100': ('https://en.wikipedia.org/wiki/Nasdaq-100#Components', 4),
        'dowjones': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components', 2)
    }

def plot_missing_pct(df, ax=None):
    missing_pct = 100 * df.isnull().sum(axis=0) / df.shape[0]
    missing_pct.sort_values().plot(kind='barh', ax=ax)

    return missing_pct

def get_index_constituents(index='sp500'):
    ''' index = ['sp500', 'nasdaq100', 'dowjones'] '''

    # download S&P500 constituents data
    index_component_url = index_component[index][0]
    index_constituents = pd.read_html(index_component_url, header=0)[index_component[index][1]]
    # clean symbol names
    index_constituents.Symbol = [tkr.replace('.','-') if '.' in tkr else tkr for tkr in index_constituents.Symbol]
    # clean column names
    index_constituents.columns = index_constituents.columns.str.lower().str.replace(' ','_')

    return index_constituents

def get_price_returns(index='sp500', interval='1d'):
    index_constituents = get_index_constituents(index)
    ticker_list = index_constituents.symbol.to_list() + ['^GSPC', '^NDX', 'DJIA'] # add market
    component_prices = yf.download(ticker_list, interval=interval, period='max')['Adj Close']
    component_returns = component_prices.pct_change()

    output_dict = {
        'prices': component_prices,
        'returns': component_returns
    }
    
    return output_dict

def plot_corr_mat(returns, ax=None):
    corr_mat = returns.dropna().corr()
    
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat, cmap='coolwarm', mask=mask, annot=False, cbar=False, ax=ax);
    
    return corr_mat
    
def static_reg(returns, y_asset, X_factors):
    returns = returns[[y_asset] + X_factors].dropna()
    X = returns[X_factors]
    y = returns[y_asset]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    return model

def rolling_reg(returns, y_asset, X_factors, window):
    returns = returns[[y_asset] + X_factors].dropna()
    X = returns[X_factors]
    y = returns[y_asset]
    model = RollingOLS(y, sm.add_constant(X), window=window, min_nobs=window).fit()
    
    return model

def vectorized_beta(returns, market_definition='^GSPC'):
    market = returns[market_definition]
    assets = returns
    # Calculate betas for all assets
    market_demeaned = market - market.mean()
    assets_demeaned = assets - assets.mean()
    betas = assets_demeaned.mul(market_demeaned, axis=0).sum(axis=0) / np.sum(market_demeaned ** 2)
    betas.name = market.name
    betas.index.name = 'symbol'

    return betas

def vectorized_corr(returns, market_definition='^GSPC'):
    # Calculate corrs for all assets
    corrs = returns.corr().loc[:, market_definition]
    corrs.index.name = 'symbol'
    
    return corrs

def filter_outliers(ser, lower_percentile=0.01, upper_percentile=0.99):
    # filter outliers
    lower_threshold,upper_threshold = ser.quantile([lower_percentile, upper_percentile])
    filtered_ser = ser[ser.between(lower_threshold, upper_threshold)]
    return filtered_ser

def standardize(ser):
    return (ser - ser.mean()) / ser.std()
    
def vectorized_rolling_calc(returns, market_definition='^GSPC', window_size=30, beta=True):
    rolling_list = []
    for w in returns.rolling(window=window_size, min_periods=window_size):
        if w.shape[0] < window_size:
            # make rolling period less than minobs nan
            nan_ser = pd.Series(index=w.columns)
            if beta:
                rolling_list.append(nan_ser)
            else:
                rolling_list.append(nan_ser)
        else:
            # calculate rolling betas
            if beta:
                betas_ser = vectorized_beta(w, market_definition=market_definition)
                rolling_list.append(betas_ser)
            else:
                corrs_ser = vectorized_corr(w, market_definition=market_definition)
                rolling_list.append(corrs_ser)
        
    rolling_df = pd.concat(rolling_list, axis=1).set_axis(returns.index, axis=1).T
    
    return rolling_df

# calculation check
def get_beta_trends(returns_df, market, y_returns, window, plot=False, ax=None):
    ols_model = static_reg(returns_df, y_asset=y_returns, X_factors=[market])
    rols_model = rolling_reg(returns_df, y_asset=y_returns, X_factors=[market], window=window)
    static_corr = returns_df[y_returns].corr(returns_df[market])
    rolling_corr = returns_df[y_returns].rolling(window=window).corr(returns_df[market])
    
    results = {
        'ols_model': ols_model,
        'rols_model': rols_model,
        'static_corr': static_corr,
        'rolling_corr': rolling_corr
    }

    if plot:
        # standardize returns
        standardize(returns_df[y_returns]).plot(ax=ax[0], label=y_returns, c='tab:blue')
        standardize(returns_df[market]).plot(ax=ax[0], label='market', c='tab:orange')
        # Draw horizontal line at y=0
        ax[0].axhline(0, color='black', linestyle='--', linewidth=1)
        
        rols_model.params[market].plot(label=f'{market} {window}d rols_model beta: {rols_model.params[market][-1]:.2f}', ax=ax[1], c='tab:orange')
        rolling_corr.plot(label=f'{market} {window}d rolling_corr: {rolling_corr[-1]:.2f}', ax=ax[1], c='tab:blue')
        ax[1].axhline(ols_model.params[market], label=f'{market} ols_model beta: {ols_model.params[market]:.2f}', ls='--', c='tab:orange')
        ax[1].axhline(static_corr, label=f'{market} static_corr: {static_corr:.2f}', ls='--', c='tab:blue')
        ax[1].axhline(0, c='black')

        for i in range(len(ax)):
            ax[i].axhline(0, color='black')
            ax[i].grid()
            ax[i].legend()
        plt.tight_layout();
    
    return results

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