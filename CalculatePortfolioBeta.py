import yfinance as yf
import numpy as np
import scipy as sci
import matplotlib as plt
import statistics as stats
import pandas as pd
import json
from pandas_datareader import data as pdr

def GetStockPrices(StockSymb, StartDate,EndDate):
    yf.pdr_override() # <== that's all it takes :-)

    # download dataframe
    data = pdr.get_data_yahoo(StockSymb, )
    return data


df = GetStockPrices('^GSPC',"2017-01-01", "2017-04-30")

print(df)