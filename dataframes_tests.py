# %% [markdown]
# ## Testing DataFrames

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%matplotlib inline
import yfinance as yf
import seaborn as sn
from datetime import datetime
from IPython.display import display, HTML


# %% [markdown]
# ## Helper Functions:
# ### Get Formated Datetime from Timestamp

# %%
def get_datetime_str_from_timestamp(tstimestamp,datatime_szformat='%Y-%m-%d'):
    """ 
    Convers a python Timestamp valiable into a string formatted per the provided
    szformat string
    format can be found at https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    """
    
    return datetime.strftime(tstimestamp, datatime_szformat)

# %% [markdown]
# ### Get Yahoo Finance Stock Price History

# %%
def get_yf_price_history(symbol, period=None,interval='1d',start_date=None,end_date=None):
    """ 
    Calls Yahoo Finance Ticker() function to retrieve stock historical prices 
    and returns a DataFrame
    """
    return yf.Ticker(symbol).history(interval=interval,period=period, start=start_date, end=end_date)

# %% [markdown]
# ### Get Period Performace DataFrame 

# %%
def get_performance_df(df_list,period_start,period_end):
    """ 
    Takes a list of DataFrames of pre-calculated % Change price data
    and creates a Performance DataFrame over the period provided 
    """
    #print("^GSPC",df_list['^GSPC'])
    perf_df = pd.DataFrame(columns=['Stock','% Price Change']) 
    i = 0
    for s in df_list.keys():
        tmp_df = pd.DataFrame(df_list[s])
        #print(s,df_list[s])
        perf_df.loc[i] = [s,tmp_df.at[tmp_df.index[-1],'Close']]
        #print(s,tmp_df.at[tmp_df.index[-1],'Close'])
        #print(perf_df.loc[i])
        i+=1

    perf_df.set_index('Stock',drop=True,verify_integrity=True)
    perf_df.index.rename('Stock',inplace=False)
    print(perf_df)
    #print(perf_df.index.name)
    return perf_df

# %% [markdown]
# ### Download and Compare Changes in Stock Prices

# %%
def plot_price_change_comp(stock_list,start_date,end_date,interval):
    """
    Plot multiple stocks price %changes on one chart and returns
    a list of Dataframes for each stock symbob in stock_list 
    """
    
    main_df = {}
    for s in stock_list:
        #main_df[s] = yf.Ticker(s).history(interval=interval, start=start_date, end=end_date)['Close'].pct_change(periods=1).dropna().cumsum() * 100
        main_df[s] = get_yf_price_history(symbol=s,interval=interval,start_date=start_date,end_date=end_date)['Close'].pct_change(periods=1).dropna().cumsum() * 100

    ax = ''
    for k in main_df.keys():
        if ax == '':
            ax = main_df[k].plot(figsize=(16,8),linewidth=3)    
        else:
            main_df[k].plot(ax=ax,linewidth=3)

    ax.set_title('Relative Perfomance: % Change of Stock Prices')
    ax.title.set_size(20)

    ax.set_xlabel("Time")
    ax.set_ylabel("% Price Change")

    ax.grid(True)
    ax.legend(main_df.keys())

    plt.show()
    return main_df


# %%

stock_list = ['^DJI','^GSPC','^IXIC']
start_date = '2022-01-03'
end_date = datetime.now()
interval = '1d'

main_df = plot_price_change_comp(stock_list,start_date,end_date,interval)
perf_df = get_performance_df(df_list=main_df,period_start=start_date,period_end=end_date)

display(perf_df)



"""
# %%
def check_sma(price_history: object, SMA_lookback_in_dayes: int) -> float:
    
    df_price_history = pd.DataFrame(price_history)
    # Calculate the stock's SMA
    sma = df_price_history['Close'].rolling(window=SMA_lookback_in_dayes).mean().dropna()
    #print(sma)
    # Calculate the change in the SMA over the specified period
    sma_change = sma.iloc[-1] - sma.iloc[-1*SMA_lookback_in_dayes]
    
    return sma.iloc[-1], sma_change

symb = "AAPL"

testdf = yf.Ticker(symb).history(interval=interval, start=start_date, end=end_date)['Close']

sma_periods= [20,50,200]
date_format = "%A %m-%d-%Y %I:%M%p"

#last_date_str = datetime.strftime(testdf.index[-1], date_format)
last_date_str = get_datetime_str_from_timestamp(testdf.index[-1],date_format)

print(symb,"Price EOD on",last_date_str," = $",np.round(testdf.iloc[-1],2))
for i in sma_periods:
    sma, sma_change = check_sma(testdf, i)
    print("-- SMA(",i,") = ",np.round(sma,2),". The % Change in last ",i,"days is ",sma_change)

print("Done!")

# %%
import matplotlib.dates as mpl_dates
import mplfinance as mpf

symbol='^GSPC'
interval = '1wk'

start_date = '2020-01-01'
end_date ='2022-12-31'

# Extracting Data for plotting
data = yf.Ticker(symbol).history(interval=interval,start=start_date, end=end_date)
data.index.name = 'Date'
kwargs={'warn_too_much_data':10000}
mpf.plot(data,figsize=(18,10),type='candle',volume=True,**kwargs,style='sas',title=symbol+" Stock")

"""

print("The End!")

