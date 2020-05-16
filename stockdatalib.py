
"""
Auth : Al Sabawi
Date :March, 2020
"""
import os
import json
import pandas as pd
import numpy as np
import datetime as datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import scipy.stats as stats
import math
import requests
import urllib
from urllib import request as url_request
from urllib.request import urlopen
from collections import defaultdict

stock = ''
directory = './data/'  # '../getstockdata/data/'
stocks_list_csv = ''
fund_filename = ''
xcolumn_name = 'date'
column_map = {}
sp_df = ''


def print_over(txt):
    print('\r', end='')
    print(txt, end='')

def set_stock(s):
    global stock
    global directory
    global stocks_list_csv
    global fund_filename
    global sp_df

    stock = s.lower()

    if (directory == ''):
        raise NameError("Error: 'data directory' is NOT set")
        print('ERROR')
    else:
        set_data_directory(directory)
        sp_df = pd.read_csv(stocks_list_csv)
        if stock.upper() not in sp_df['Symbol'].values:
            raise NameError("Error: '" + stock + "' is NOT AVAILABLE")
        fund_filename = directory + stock + '_fund.csv'


def get_stock():
    return stock

def get_stock_info(symbol):
    global sp_df

    symbol = symbol.lower()
    # Get stock company information
    stock_info = sp_df[sp_df.Symbol == symbol.upper()]
    company_name = stock_info['Name'].iloc[0]
    company_sector = stock_info['Sector'].iloc[0]
    return {'stock' : [stock], 'name': [company_name], 'sector':[company_sector]}

def set_stocks_list_filename(filename):
    global stocks_list_csv
    stocks_list_csv = directory + filename

def set_data_directory(datadir):
    global directory
    global stocks_list_csv
#    if stocks_list_csv == '':
#        print("Error: stocks list file name is not set. Use set_stocks_list_filename(filename) to set it.")
#        return
    
    directory = datadir
    #stocks_list_csv = directory + stocks_list_csv
    #print(stocks_list_csv)


def parser(x):
    return datetime.datetime.strptime(x, "%Y-%m-%d")

def days_count(from_date, to_date=datetime.datetime.today().date()):
    if isinstance(from_date, datetime.datetime) is False:
        from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d").date()
    if isinstance(to_date, datetime.datetime) is False:
        to_date = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()

    delta = abs((to_date - from_date).days)
    return delta


def time_series_trendline(datetime_axis, values):
    x = range(len(datetime_axis))
    y = values
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)

    print('Slope = ', fit[0], ", ", "Intercept = ", fit[1])
    print('Line Equation :  ', fit_fn)

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111).grid()
    plt.plot(values.index, y, label="Data")
    plt.plot(values.index, fit_fn(x), 'b-', label="Trend Line", color='red')
    _ = plt.xticks(rotation=45)
    plt.show()


def GetStocksList():
    global stocks_list_csv
    sp_df = pd.DataFrame()
    #print(stocks_list_csv)
    try:
        sp_df = pd.read_csv(stocks_list_csv)
        sp_df.sort_values('Symbol', ascending=True, inplace=True)

    except:
        print("Cannot open file", stocks_list_csv)

    # returns count_row, count_col, df
    return sp_df.shape[0], sp_df.shape[1], sp_df


def GetYahooStockData(symbol):
    symbol = symbol.lower()
    # create filename and read it
    fname = directory + symbol + '.json'
    with open(fname, 'r') as rfile:
        read_content = json.load(rfile)

    return read_content


def GetStockDataFrame(symbol):
    symbol = symbol.lower()
    data = GetYahooStockData(symbol)
    df = pd.DataFrame()
    if(data['chart']['result'] == None):
        return df
    if "timestamp" not in data["chart"]["result"][0]:
        return df
    
    df['Timestamps'] = pd.to_datetime(data["chart"]["result"][0]["timestamp"], unit='s')
    df["Open"] = data["chart"]["result"][0]["indicators"]["quote"][0]["open"]
    df["High"] = data["chart"]["result"][0]["indicators"]["quote"][0]["high"]
    df["Low"] = data["chart"]["result"][0]["indicators"]["quote"][0]["low"]
    df["AdjClose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
    df['Volume'] = data["chart"]["result"][0]["indicators"]["quote"][0]["volume"]
    df['Volume'] = df['Volume']

    df.set_index('Timestamps', inplace=True)
    return df


def GetFund_Dict(symbol):
    symbol = symbol.lower()
    set_stock(symbol)
    global column_map
    sources = ['income-statement', 'cash-flow-statement', 'balance-sheet-statement',
               'enterprise-value', 'company-key-metrics', 'financial-ratios']
    filename_list = {}
    fund_df_list = {}

    for src in sources:
        filename_list[src] = directory + stock.lower() + '_' + src + '.csv'
        fund_df_list[src] = pd.DataFrame()

    try:
        for src in sources:
            fund_df_list[src] = pd.read_csv(filename_list[src], parse_dates=[0], date_parser=parser)
            if fund_df_list[src].columns[0] == 'Date':
                fund_df_list[src].rename(columns={'Date': 'date'}, inplace=True)

            index = xcolumn_name
            if (fund_df_list[src][xcolumn_name].is_unique):
                fund_df_list[src].set_index(index, inplace=True)
            else:
                print("Cannot create and index - Rows are not unique")

            fund_df_list[src] = fund_df_list[src].sort_values(xcolumn_name, ascending=True)

            for col in fund_df_list[src].columns:
                column_map[col] = src
    except Exception as e:
        print('Exception : ', e)

    # returns count_row, count_col, df
    return fund_df_list.keys(), column_map, fund_df_list


def FindColumn(name):
    global column_map
    col_names = [key for key, value in column_map.items() if name.lower() in key.lower()]
    dataframes = [value for key, value in column_map.items() if key in col_names]

    ref_table = pd.DataFrame(list(zip(col_names, dataframes)), columns=['column_name', 'DataFrame'])

    stmt = 'These column names ' + str(col_names) + ' are in these dataframes ' + str(dataframes)

    return ref_table, stmt


def MakeFund_Subset(main_df_org, relevent_columns):
    main_df = main_df_org[relevent_columns].copy()

    return main_df


def CleanUp_DataFrame(main_df):
    # Remove any column with 'None'  value
    main_df = main_df[~main_df.eq('None').any(1)]
    main_df = main_df.dropna()

    for col in main_df.columns:
        if col != xcolumn_name:
            main_df[col] = main_df[col].astype(float)

    main_df = main_df.dropna()
    return main_df


def DatesRange(df, start, end=datetime.datetime.today()):
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def skip_every_n(df, n):
    df2 = df.iloc[::n, :]
    return df2

def PlotBasicCharts(symbol, price_df = {}):
    symbol = symbol.lower()
    set_stock(symbol)

    if len(price_df)==0:
        price_df = GetStockDataFrame(get_stock())

    stock_info = get_stock_info(symbol)
    # Get stock company information
    company_name = stock_info['name'][0]
    company_sector = stock_info['sector'][0]

    stock_title = stock.upper() + ' ' + '(' + company_name + ')' + ' [' + company_sector + ' Sector]'
    #price_df = GetStockDataFrame(stock)

    # Load the stock fundamental data
    keys, column_map, main_df_org = GetFund_Dict(stock)

    # Create a subset of columns in a DataFrame from the stock fundamentals dataset
    cols = ['Revenue', 'Earnings before Tax', 'Net Income', 'Free Cash Flow margin', 'Gross Profit',
            'Net Profit Margin']  # [ 'Shares', 'Price', 'Revenue', 'Earnings']
    main_df1 = MakeFund_Subset(main_df_org['income-statement'], cols)

    # Create another subset of columns from another fundamentals dataset
    cols = ['Number of Shares', 'Stock Price']
    main_df2 = MakeFund_Subset(main_df_org['enterprise-value'], cols)

    # Create a 3rd subject of columns from yet another fundamentals dataset
    cols = ['Capital Expenditure', 'Free Cash Flow']
    main_df3 = MakeFund_Subset(main_df_org['cash-flow-statement'], cols)

    # Create a 4th subject of columns from yet another fundamentals dataset
    cols = ['Free Cash Flow per Share']
    main_df4 = MakeFund_Subset(main_df_org['company-key-metrics'], cols)

    # Merge the 3 dataframes into a single dataframe
    main_df = pd.merge(main_df1, main_df2, on='date')
    main_df = pd.merge(main_df, main_df3, on='date')
    main_df = pd.merge(main_df, main_df4, on='date')

    # Calculate the moving averages
    MADays = [40, 100, 200]
    #MA = [{}]
    ma_names = []
    for ma in MADays:
        mean_name = str(ma) + ' days MA'
        ma_names.append(mean_name)
        tmp_df = pd.DataFrame(
            {'Timestamps': price_df.index.tolist(), mean_name: price_df['AdjClose'].rolling(window=ma).mean()})
        price_df = pd.merge(price_df, tmp_df, left_index=True, right_index=True)

    fig = plt.figure(figsize=(15, 6))

    linear_trends_df = pd.DataFrame()
    last_date = price_df.index[-1]
    # ***************************************************
    # Find the full trend line
    x = range(len(price_df.index))
    fit = np.polyfit(x, price_df['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    linear_trends_df['5 Years Trend Line Function'] = [str(fit_fn).strip()]

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df['5 Yr Trend'] = trend
    # ***************************************************

    twoyears_ago = last_date - datetime.timedelta(days=2 * 365)
    price_df_2y = price_df[price_df.index > twoyears_ago].copy(deep=True)
    # price_df = price_df_2y
    # Find the 2 years trend line
    x = range(len(price_df_2y.index))
    fit = np.polyfit(x, price_df_2y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    linear_trends_df['2 Years Trend Line Function'] = [str(fit_fn).strip()]

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_2y['2 Yr Trend'] = trend
    # ***************************************************

    oneyears_ago = last_date - datetime.timedelta(days=1 * 365)
    price_df_1y = price_df[price_df.index > oneyears_ago].copy(deep=True)
    # Find the 1 year trend line
    x = range(len(price_df_1y.index))
    fit = np.polyfit(x, price_df_1y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    linear_trends_df['1 Year Trend Line Function'] = [str(fit_fn).strip()]

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_1y['1 Yr Trend'] = trend
    # ***************************************************

    sixmonth_ago = last_date - datetime.timedelta(days=0.5 * 365)
    price_df_6m = price_df[price_df.index > sixmonth_ago].copy(deep=True)
    # Find the 6 months trend line
    x = range(len(price_df_6m.index))
    fit = np.polyfit(x, price_df_6m['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    linear_trends_df['6 Months Trend Line Function'] = [str(fit_fn).strip()]

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_6m['6 Month Trend'] = trend
    # ***************************************************

    ts = pd.to_datetime(str(price_df.index[0]))
    frm = ts.strftime('%Y-%m-%d')
    ts = pd.to_datetime(str(price_df.index[-1]))
    to = ts.strftime('%Y-%m-%d')

    stock_title = stock_title + '\nFrom ' + frm \
                  + ' To ' + to

    ## Create Plot
    fig, axs = plt.subplots(4, figsize=(15, 25), sharex=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1, 1, 1]})
    #fig.suptitle(stock_title, fontsize=30)

    y_list = ['Low', 'AdjClose', 'High']
    for ma in ma_names:
        y_list.append(ma)
    y_list.append('5 Yr Trend')

    price_df.plot(ax=axs[0], y=y_list,
                  title=stock_title, lw=2)
    price_df_2y.plot(ax=axs[0], y=['2 Yr Trend'], lw=2)
    price_df_1y.plot(ax=axs[0], y=['1 Yr Trend'], lw=2)
    price_df_6m.plot(ax=axs[0], y=['6 Month Trend'], grid=True, lw=2)

    axs[0].set_ylabel('Stock Price')

    dic = [{}]
    i = 0
    main_df['EPS'] = MakeFund_Subset(main_df_org['income-statement'], ['EPS'])
    main_df['PE ratio'] = MakeFund_Subset(main_df_org['company-key-metrics'], ['PE ratio'])

    from_date = main_df['EPS'].index[i]

    to_date = main_df['EPS'].index[i + 1]
    for d in price_df.index:
        if d > to_date:
            i = i + 1
            if i < len(main_df['EPS']) - 1:
                from_date = main_df['EPS'].index[i]
                to_date = main_df['EPS'].index[i + 1]
                continue
            else:
                break

        elif from_date <= d <= to_date:
            jj = main_df[main_df.index == from_date]['EPS'][0]
            pp = main_df[main_df.index == from_date]['PE ratio'][0]
            ff = main_df[main_df.index == from_date]['Free Cash Flow per Share'][0]
            rr = main_df[main_df.index == from_date]['Revenue'][0]

            ee = main_df[main_df.index == from_date]['Earnings before Tax'][0]
            nn = main_df[main_df.index == from_date]['Net Income'][0]
            cc = main_df[main_df.index == from_date]['Free Cash Flow'][0]
            gg = main_df[main_df.index == from_date]['Gross Profit'][0]

            kk = price_df[price_df.index == d]['AdjClose'].values[0]

            dic.append({'Timestamps': d, 'Calc. P/E': (kk / jj) / 4.0, 'EPS': jj,
                        'P/E': pp, 'FCF/Share': ff, 'Revenue': rr,
                        'Earnings before Tax': ee, 'Net Income': nn,
                        'Free Cash Flow': cc, 'Gross Profit': gg})

        else:
            continue

    plots_df = pd.DataFrame(dic).dropna()
    plots_df.set_index('Timestamps',inplace=True)

    plots_df.plot(ax=axs[1], y=['EPS'], grid=True, lw=4)
    plots_df.plot(ax=axs[1], y=['FCF/Share'], grid=True, lw=4)
    axs[1].set_ylabel('Earning && FCF/Share')
    plots_df.plot(ax=axs[2], y=['P/E'], grid=True, lw=4)
    plots_df.plot(ax=axs[2], y=['Calc. P/E'], grid=True, lw=4)
    axs[2].set_ylabel('Price/Earning (P/E)')

    plots_df.plot(ax=axs[3], y=['Revenue'], lw=4)
    plots_df.plot(ax=axs[3], y=['Earnings before Tax'], lw=4)
    plots_df.plot(ax=axs[3], y=['Net Income'], lw=4)
    plots_df.plot(ax=axs[3], y=['Free Cash Flow'], lw=4)
    plots_df.plot(ax=axs[3], y=['Gross Profit'], grid=True, lw=4)

    axs[3].set_ylabel('Growth')

    for ax in axs:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    plt.show()

    return linear_trends_df


def PlotTrends(symbol, price_df = {}):
    trend_fn_dict = {}
    symbol = symbol.lower()
    set_stock(symbol)
    if len(price_df)==0:
        price_df = GetStockDataFrame(get_stock())

    # Get stock company information
    sp_df = pd.read_csv(stocks_list_csv)
    stock_info = sp_df[sp_df.Symbol == get_stock().upper()]
    company_name = stock_info['Name'].iloc[0]
    company_sector = stock_info['Sector'].iloc[0]

    price_data_df = price_df.copy(deep=True)
    price_data_df.dropna()
    dif_period = 5

    # Load stock End of Day prices file

    price_change = 'Price $ Change from ' + str(dif_period) + ' days ago'
    price_data_df[price_change] = price_data_df['AdjClose'].diff(dif_period).dropna().astype(float)

    price_change_percent = 'Price % Change from ' + str(dif_period) + ' days ago'
    price_data_df[price_change_percent] = price_data_df['AdjClose'].pct_change(dif_period).dropna().astype(float) * 100

    volume_change_percent = 'Volume % Change from ' + str(dif_period) + ' days ago'
    price_data_df[volume_change_percent] = price_data_df['Volume'].pct_change(dif_period).dropna().astype(float) * 100

    last_date = price_data_df.index[-1]
    # ***************************************************
    # Find the full trend line
    x = range(len(price_data_df.index))
    fit = np.polyfit(x, price_data_df['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    trend_fn_dict.update({"5y Trend" : fit_fn})
    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_data_df['5 Yr Trend'] = trend
    # ***************************************************

    twoyears_ago = last_date - datetime.timedelta(days=2 * 365)
    price_df_2y = price_data_df[price_data_df.index > twoyears_ago].copy(deep=True)
    # Find the 2 years trend line
    x = range(len(price_df_2y.index))
    fit = np.polyfit(x, price_df_2y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    trend_fn_dict.update({"2y Trend" : fit_fn})
    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_2y['2 Yr Trend'] = trend
    # ***************************************************

    oneyears_ago = last_date - datetime.timedelta(days=1 * 365)
    price_df_1y = price_data_df[price_data_df.index > oneyears_ago].copy(deep=True)
    # Find the 1 year trend line
    x = range(len(price_df_1y.index))
    fit = np.polyfit(x, price_df_1y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    trend_fn_dict.update({"1y Trend" : fit_fn})
    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_1y['1 Yr Trend'] = trend
    # ***************************************************

    sixmonth_ago = last_date - datetime.timedelta(days=0.5 * 365)
    price_df_6m = price_data_df[price_data_df.index > sixmonth_ago].copy(deep=True)
    # Find the 6 months trend line
    x = range(len(price_df_6m.index))
    fit = np.polyfit(x, price_df_6m['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    trend_fn_dict.update({"6m Trend" : fit_fn})
    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_6m['6 Month Trend'] = trend
    # ***************************************************

    stock_title = stock.upper() + ' ' + '(' + company_name + ')' + ' [' + company_sector + ' Sector]'
    ts = pd.to_datetime(str(price_data_df.index[0]))
    frm = ts.strftime('%Y-%m-%d')
    ts = pd.to_datetime(str(price_data_df.index[-1]))
    to = ts.strftime('%Y-%m-%d')

    stock_title = stock_title + '\nFrom ' + frm + ' To ' + to

    price_data_df['2 Yr Trend'] = price_df_2y['2 Yr Trend'].copy(deep=True)
    price_data_df['1 Yr Trend'] = price_df_1y['1 Yr Trend'].copy(deep=True)
    price_data_df['6 Month Trend'] = price_df_6m['6 Month Trend'].copy(deep=True)

    # , squeeze=True, gridspec_kw = {'height_ratios':[1,2,5,2,3]}
    fig, axs = plt.subplots(8, figsize=(15, 25), sharex=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [3, 3, 3, 3, 2, 2, 2, 2]})
    #fig.suptitle(stock_title, fontsize=30)

    price_data_df.plot(ax=axs[0], y=['Low', 'AdjClose', 'High', '5 Yr Trend'], grid=True, title=stock_title,)
    price_data_df.plot(ax=axs[1], y=['Low', 'AdjClose', 'High', '2 Yr Trend'], grid=True)
    price_data_df.plot(ax=axs[2], y=['Low', 'AdjClose', 'High', '1 Yr Trend'], grid=True)
    price_data_df.plot(ax=axs[3], y=['Low', 'AdjClose', 'High', '6 Month Trend'], grid=True)

    price_data_df.plot(ax=axs[4], y=[price_change], grid=True, color='green')
    axs[4].set_ylabel('Price Delta in ' + str(dif_period) + 'd')

    price_data_df.plot(ax=axs[5], y=[price_change_percent], grid=True, color='black')
    axs[5].set_ylabel('Price % Delta in ' + str(dif_period) + 'd')

    price_data_df.plot(ax=axs[6], y=['Volume'], grid=True, color='green')
    axs[6].set_ylabel('Vol. Delta in ' + str(dif_period) + 'd')

    price_data_df.plot(ax=axs[7], y=[volume_change_percent], grid=True, color='black')
    axs[7].set_ylabel('Vol. % Delta in ' + str(dif_period) + 'd')

    # Find Max and Min for price and volume changes and mark price max and min on chart
    max_price_change = price_data_df[np.round(price_data_df[price_change_percent], 2)
                                     == np.round(price_data_df[price_change_percent].max(), 2)]
    min_price_change = price_data_df[np.round(price_data_df[price_change_percent], 2)
                                     == np.round(price_data_df[price_change_percent].min(), 2)]

    max_vol_change = price_data_df[price_data_df[volume_change_percent]
                                   == price_data_df[volume_change_percent].max()]
    min_vol_change = price_data_df[price_data_df[volume_change_percent]
                                   == price_data_df[volume_change_percent].min()]

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    ts = pd.to_datetime(str(max_price_change.index.values[0]))
    mxp = ts.strftime('%Y-%m-%d')

    ts = pd.to_datetime(str(min_price_change.index.values[0]))
    mnp = ts.strftime('%Y-%m-%d')

    ts = pd.to_datetime(str(max_vol_change.index.values[0]))
    mxv = ts.strftime('%Y-%m-%d')

    # Annotate charts with max and min
    axs[5].annotate(str(np.round(max_price_change[price_change_percent].values[0], 2)) + '%',
                    xy=(max_price_change.index.values[0],
                        max_price_change[price_change_percent].values[0]),
                    xytext=(max_price_change.index.values[0],
                            max_price_change[price_change_percent].values[0] * 0.5),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

    ts = pd.to_datetime(str(min_vol_change.index.values[0]))
    mnv = ts.strftime('%Y-%m-%d')
    axs[5].annotate(str(np.round(min_price_change[price_change_percent].values[0], 2)) + '%',
                    xy=(min_price_change.index.values[0],
                        min_price_change[price_change_percent].values[0]),
                    xytext=(min_price_change.index.values[0],
                            min_price_change[price_change_percent].values[0] * 0.5),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    )

    maxtable_df = pd.DataFrame({'Change Type': ['Max % Rise', 'Max % Drop', 'Max % Rise', 'Max % Drop'],
                                'Date': [mxp, mnp, mxv, mnv],
                                'Price': [str(np.round(max_price_change[price_change_percent].values[0], 2)) + '%',
                                          str(np.round(min_price_change[price_change_percent].values[0], 2)) + '%',
                                          '--', '--'],
                                'Volume': ['--', '--',
                                           str(np.round(max_vol_change[volume_change_percent].values[0], 2)) + '%',
                                           str(np.round(min_vol_change[volume_change_percent].values[0], 2)) + '%']})
    maxtable_df = maxtable_df.set_index('Change Type')

    plt.show()

    return price_data_df, maxtable_df, trend_fn_dict

def quote(symbol):
    set_stock(symbol)

    price_df = GetStockDataFrame(symbol)
    if(price_df.empty):
        return price_df
    
    date = price_df.index[-1]
    price = np.round(price_df['AdjClose'].iloc[-1],2)
    vol = price_df['Volume'].iloc[-1]

    dict = {'symbol' :[symbol], 'date' : [date], 'close' : [price], 'volume': [vol]}

    res = pd.DataFrame(dict)
    return res

def key_stat(symbol):
    set_stock(symbol)

    # Get stock company information
    stock_info = get_stock_info(symbol)
    # Get stock company information
    company_name = stock_info['name'][0]
    company_sector = stock_info['sector'][0]

    # Load the stock fundamental data
    keys, column_map, main_df_org = GetFund_Dict(stock)

    date = main_df_org['income-statement'].index[-1]
    revenue = main_df_org['income-statement']['Revenue'].iloc[-1]
    earning_before_tax = main_df_org['income-statement']['Earnings before Tax'].iloc[-1]
    net_income = main_df_org['income-statement']['Net Income'].iloc[-1]
    free_cash_flow_margin = main_df_org['income-statement']['Free Cash Flow margin'].iloc[-1]
    gross_profit =  main_df_org['income-statement']['Gross Profit'].iloc[-1]

    number_of_shares = main_df_org['enterprise-value']['Number of Shares'].iloc[-1]

    free_cash_flow_per_share = main_df_org['company-key-metrics']['Free Cash Flow per Share'].iloc[-1]

    pe_ratio = main_df_org['company-key-metrics']['PE ratio'].iloc[-1]
    eps = main_df_org['income-statement']['EPS'].iloc[-1]

    dict = {'symbol': [symbol], 'Date': [date], 'No. of Shares' : [number_of_shares],'Revenue': [revenue], 'Earning before Tax': [earning_before_tax],
            'PE Ratio' : [pe_ratio] ,'Earning per Share':[eps] ,'Net Income' : [net_income], 'Free Cash Flow Margin' : [free_cash_flow_margin],
            'Gross Profit' : [gross_profit], 'Free Cash Flow per Share' : [free_cash_flow_per_share]}

    res = pd.DataFrame(dict)
    return res


def GetPriceChangesPercent(price_df, periods):
    if (len(periods) < 1):
        raise NameError("Error : A list of atl least 1 period must be provided")
        return

    price_changes = pd.DataFrame(columns=periods + ['Timestamps'])
    updownlist = []

    for i in periods:
        price_changes[i] = (price_df['AdjClose'].pct_change(i) * 100).dropna().copy(deep=True)

        updownlist = list()
        for n in price_changes[i]:
            if n > 0:
                updownlist.append(1)
            elif n < 0:
                updownlist.append(-1)
            else:
                updownlist.append(0)

        price_changes['Directions ' + str(i)] = updownlist

    return price_changes


def PlotPriceChangesPercent(price_df, periods):
    if (len(periods) < 1):
        raise NameError("Error : A list of atl least 1 period must be provided")
        return

    price_changes = GetPriceChangesPercent(price_df, periods)

    # Start plotting
    fig, ax = plt.subplots(len(periods) + 1, 1, figsize=(15, 20), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle("Price Changes Chart", fontsize=30)
    for axx in ax:
        axx.xaxis.grid(which='minor')
        axx.yaxis.grid(which='minor')
        axx.grid(True)

        _ = plt.xticks(rotation=45)
        axx.minorticks_on()

        axx.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
        axx.xaxis.grid(which='major', linestyle='-', linewidth='1.0', color='black')
        axx.yaxis.grid(which='major', linestyle='-', linewidth='1.0', color='black')

    price_df['AdjClose'].plot(ax=ax[0])
    ax[0].set_ylabel('Stock Price')

    k = 1
    for i in periods:
        price_changes[i].plot(ax=ax[k], color='green', lw=2.0)
        ax[k].set_ylabel(str(i) + ' Day Change')
        k = k + 1

    plt.show()

def PlotPriceChangesKDE(price_df, periods):

    if (len(periods) < 1):
        raise NameError("Error : A list of atl least 1 period must be provided")
        return

    price_changes = GetPriceChangesPercent(price_df, periods)

    # Start plotting
    rows = len(periods)
    cols = 1
    fig, ax = plt.subplots(rows, cols, figsize=(10, 15), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    ax[0].set_ylabel('Probability Density')

    fig.suptitle("Price Changes KDE & Normal Dist.", fontsize=30)

    k = 0
    for i in periods:
        price_changes_inperiod = price_changes[i].dropna()
        test_df = pd.DataFrame({str(i) + ' Day KDE Distribution': price_changes_inperiod.tolist()})
        if rows == 1 and cols == 1:
            ax.axvline(x=price_changes_inperiod.mean(), linewidth=2, color='r')
            test_df.plot(ax=ax, kind='kde', color='black', lw=2, grid=True)
        else:
            ax[k].axvline(x=price_changes_inperiod.mean(), linewidth=2, color='r')
            test_df.plot(ax=ax[k], kind='kde', color='black', lw=2, grid=True)

        k = k + 1

    ######################################################################
    stats_list = []
    kn = 0

    for i in periods:
        price_changes_inperiod = price_changes[i].dropna()
        d = price_changes_inperiod.describe()
        dictt = {'index': i,
                 'Record Count': d['count'],
                 'Mean of % Price Change': price_changes_inperiod.mean(),
                 'Std. Dev of % Price Change': price_changes_inperiod.std(),
                 'Var of % Price Change': price_changes_inperiod.var(),
                 'Max. % Price Rise': d['max'],
                 'Max % Price Drop': d['min']}
        stats_list.append(dictt)

        mean = price_changes_inperiod.mean()
        std = price_changes_inperiod.std()
        xn = np.linspace(d['min'], d['max'], 100)
        yn = stats.norm.pdf(xn, mean, std)

        if rows == 1 and cols == 1:
            ax.plot(xn, yn, color='green', label=str(i) + ' Normal Distribution')
            ax.legend()
        else:
            ax[kn].plot(xn, yn, color='green', label=str(i) + ' Normal Distribution')
            ax[kn].legend()

        kn = kn + 1

    stats_out = pd.DataFrame(stats_list)
    stats_out.set_index('index', inplace=True)
    plt.show()
    return price_changes, stats_out

def GetBuySellEnvelope(s,price_df, period):
    price_changes = GetPriceChangesPercent(price_df, [period])
    price_changes['data'] = price_changes[period].dropna().copy(deep=True)
     
    # Set the period
    # number of points to be checked before and after
    n = period 

    # Find local peaks
    price_changes['min'] = price_changes.iloc[argrelextrema(price_changes.data.values,
                                                            np.less_equal, order=n)[0]]['data']
    price_changes['max'] = price_changes.iloc[argrelextrema(price_changes.data.values,
                                                            np.greater_equal, order=n)[0]]['data'] 
                                                            
    low_df = price_changes['min'].dropna()
    hi_df = price_changes['max'].dropna()
    
    #print(low_df.shape[0])
    
    action_df = ''
    if(low_df.shape[0] == 0 or hi_df.shape[0]==0):
        return False, price_df['AdjClose'], low_df, hi_df, action_df
    
    date_last_buy = low_df.index[-1]
    date_last_sell = hi_df.index[-1]
        
    price_last_buy = round(price_df['AdjClose'][price_df.index == low_df.index[-1]][0],2)
    price_last_sell = round(price_df['AdjClose'][price_df.index == hi_df.index[-1]][0],2)

    ave_min_delta = round(low_df.mean(),2)
    ave_max_delta = round(hi_df.mean(),2)    
    
    range_percent = ave_max_delta - ave_min_delta
    
    q = quote(s)
    margin = 0.10 * (range_percent/100) # 10% of the range
    
    buy_str = sell_str = buy_recom = sell_recom = ''
    buy_recommendation =  sell_recommendation = 'no' 
    buy_at = sell_at = ''
    if date_last_buy > date_last_sell:
        range_buy  = round( price_last_buy,2)
        
        range_buy_plus_margin = round(range_buy+ (range_buy * margin),2)
        
        upper_limit = round( price_last_buy + (price_last_buy * (ave_max_delta/100) ) , 2)
        range_sell  = round( price_last_sell , 2)
        
        buy_str = str(round( price_last_buy,2))+' - '+ str( range_sell ) 
        buy_at = round( price_last_buy,2)
        sell_at = range_sell
        buy_recom = "Buy@"+str(range_buy)+' - '+'Sell@'+str(range_sell)
        
        if q.close[0] <= range_buy_plus_margin and range_buy_plus_margin < range_sell :
            buy_recommendation = 'buy'
    else:
        range_sell  = round( price_last_sell,2)
        
        range_sell_minus_margin = round(range_sell - (range_sell * margin), 2)
        
        lower_limit = round( price_last_sell - (price_last_sell * (ave_min_delta/100) ), 2)
        
        range_buy = round( price_last_buy ,2)
        
        sell_str =  str( range_sell ) +' - '+ str(round( price_last_buy,2))
        buy_at = round( price_last_buy,2)
        sell_at = range_sell
        sell_recom = "Sell@"+str(range_sell)+' - '+'Buy@'+str(range_buy)
        if q.close[0] >= range_sell_minus_margin and range_sell_minus_margin > range_buy :
            sell_recommendation = 'sell'        

    range_buy  = round( price_last_buy+ave_min_delta,2)
    range_sell = round( price_last_buy+ave_max_delta,2)
    
    dict = {'Signal':['Buy','Sell'],'Recommendation':[buy_recommendation,sell_recommendation], 'Statement':[buy_recom,sell_recom],
    'Last Date':[date_last_buy,date_last_sell],
    'Average % Change':[ave_min_delta,ave_max_delta],'Signal Price':[price_last_buy,price_last_sell],'Range':[buy_str,sell_str],
    'BuyAt':buy_at,'SellAt':sell_at}
    
    action_df = pd.DataFrame(dict)
    
    return True, price_df['AdjClose'], low_df, hi_df, action_df  
    

def PlotBuySellEnvelope(price_df, period):
    price_changes = GetPriceChangesPercent(price_df, [period])

    price_changes['data'] = price_changes[period].dropna().copy(deep=True)

    # Set the period
    n = period  # number of points to be checked before and after

    # Find local peaks
    price_changes['min'] = price_changes.iloc[argrelextrema(price_changes.data.values,
                                                            np.less_equal, order=n)[0]]['data']
    price_changes['max'] = price_changes.iloc[argrelextrema(price_changes.data.values,
                                                            np.greater_equal, order=n)[0]]['data']

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle("Buy-Sell Envelope Chart", fontsize=30)

    price_df['AdjClose'].plot(ax=axs[0]).grid(True)
    axs[0].set_ylabel('Stock Price')

    # Plot the lower and upper envelop
    price_minima = pd.DataFrame(price_changes['min'].dropna()).copy(deep=True)
    plt.plot(price_minima, c='green', label='Buy Line', lw=2.0)

    price_maxima = pd.DataFrame(price_changes['max'].dropna()).copy(deep=True)
    plt.plot(price_maxima, c='red', label='Sell Line')
    ###############################

    plt.scatter(price_changes.index, price_changes['min'], color='green')
    plt.scatter(price_changes.index, price_changes['max'], color='red')
    plt.plot(price_changes.index, price_changes['data'], c='gray')
    plt.grid(which='major', axis='both')
    axs[1].set_ylabel(str(n) + ' Day Direction Chart')
    axs[1].legend()

    for ax in axs:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    #ave_min_delta = price_changes['min'].mean()
    #ave_max_delta = price_changes['max'].mean()
    
    #print("Buy range from ", price_df['AdjClose'].iloc[-1]+ave_min_delta, "to", price_df['AdjClose'].iloc[-1]+ave_max_delta)
    #price_last_buy = price_df[ price_df['AdjClose'].index == price_changes['min'].index].value
    #date_last_buy = price_changes['min'].index
    #print("Last buy signal on ",date_last_buy,"Close price was $",price_last_buy)
    plt.show()
    
    return price_df['AdjClose'],price_changes['min'],price_changes['max']
    
    



def UpdateStockData(symbol):
    global directory

    if (directory == ''):
        raise NameError("Error: 'data directory' is NOT set")
        return

    # make 2 parts of the yahoo URL so we can insert the symbol in between them
    data_range = '5y'  # 2y to 5y
    urlA = 'https://query1.finance.yahoo.com/v7/finance/chart/'
    urlB = '?range=' + data_range + '&interval=1d&indicators=quote&includeTimestamps=true'

    s = symbol

    # Create the yahoo finance URL
    url = urlA + s + urlB
    print("Downloading EOD data for "+s+" from "+url)

    # Request the data
    try:
        print("Requesting data .. ")
        stockdata = requests.get(url)
        #data = url_request.urlopen(url).read().decode('utf-8')
        print("Write to JSON file ")
        data = stockdata.json()
    except  urllib.error.URLError as e:
        print('ERROR ' + s + ' ' + e.reason, 'error')
        return

    # write data into a json filw
    try:
        print("Open and write to file "+ directory + s + '.json')
        with open(directory + s + '.json', 'w') as f:
            json.dump(data, f, indent=4)
    except:
        print("ERROR: Opening/Writing file ", directory + s + '.json', 'error')


    ### Down load fundamentals
    Source = 'FMP'
    print("Downloading " + s + " stock' fundementals FROM " + Source.upper())
    url = 'UN-INITIALIZED'

    if Source.upper() == 'STOCKPOP':
        replace_symb = {'AAL': 'AMR', 'ANDV': 'TSO', 'ANTM': 'WLP', 'AON': 'AOC', 'ARNC': 'AA', 'ATGE': 'DV',
                        'BEAM': 'FO'}
        exclude_symbols = ['GOOGL']
        if s in replace_symb.keys():
            print(s + ' is being replaced with ' + replace_symb[s])
            s = replace_symb[s]

        if s in exclude_symbols:
            print("Skipping " + s)
            return
        url = 'http://www.stockpup.com/data/' + s + '_quarterly_financial_data.csv'
    elif Source.upper() == 'FINANCIALMODELINGPREP' or 'FMP':
        replace_symb = {}
        exclude_symbols = ['SCG']
        if s in replace_symb.keys():
            print(s + ' is being replaced with ' + replace_symb[s])
            s = replace_symb[s]

        if s in exclude_symbols:
            print("Skipping " + s)
            return

        url_list = {}
        url_list[
            'income-statement'] = 'https://financialmodelingprep.com/api/v3/financials/income-statement/' + s + '?datatype=csv&period=quarter'
        url_list[
            'cash-flow-statement'] = 'https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/' + s + '?datatype=csv&period=quarter'
        url_list[
            'balance-sheet-statement'] = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/' + s + '?datatype=csv&period=quarter'
        url_list[
            'enterprise-value'] = 'https://financialmodelingprep.com/api/v3/enterprise-value/' + s + '?period=quarter'
        url_list[
            'company-key-metrics'] = 'https://financialmodelingprep.com/api/v3/company-key-metrics/' + s + '?datatype=csv&period=quarter'
        url_list[
            'financial-ratios'] = 'https://financialmodelingprep.com/api/v3/financial-ratios/' + s + '?datatype=csv&period=quarter'

    else:
        # print('*** Error: Invalid data source URL', url, ' ***')
        print('ERROR ' + s + ': ' + '*** Error: Invalid data source URL', url, ' ***')
        return

    try:
        if Source.upper() == 'STOCKPOP':
            print(url)
            s = s.lower()
            csvfilename = './data/' + s + '_fund.csv'
            fund_data = url_request.urlopen(url).read().decode('utf-8')
            with open(csvfilename, 'w') as csvfile:
                csvfile.write(fund_data)

        elif Source.upper() == 'FINANCIALMODELINGPREP' or 'FMP':
            s = s.lower()
            for key in url_list:
                url = url_list[key]
                print(url)
                response = urlopen(url)
                # print(response.headers['content-type'])
                if response.headers['content-type'].lower() == 'text/csv;charset=utf-8' or response.headers[
                    'Content-Type'].lower() == 'text/csv;charset=utf-8':
                    raw_data = response.read().decode("utf-8")

                    csvfilename = directory + 'Temp_' + s + '_' + key + '.csv'
                    with open(csvfilename, 'w') as csvfile:
                        csvfile.write(raw_data)

                    csvfilename_out = directory + s + '_' + key + '.csv'
                    pd.read_csv(csvfilename, header=None).T.to_csv(csvfilename_out, header=False, index=False)
                    os.remove(csvfilename)
                elif response.headers['content-type'].lower() == 'application/json;charset=utf-8' or \
                        response.headers['Content-Type'].lower() == 'application/json;charset=utf-8':
                    raw_data = requests.get(url)
                    data = raw_data.json()

                    if len(data) == 0:
                        # print(s + ' data.dict Empty ERROR -- Skipped')
                        print(s + ' data.dict Empty ERROR -- Skipped')
                        return
                    elif 'Error' in data.keys():
                        # print(s + ' data.dict ', data['Error'], ' ERROR -- Skipped')
                        print(s + ' data.dict ' + str(data['Error']) + ' ERROR -- Skipped')
                        return

                    l = list(data.keys())
                    keyl = l[1]

                    d = defaultdict(list)
                    if not isinstance(data, dict) or keyl not in data.keys():
                        print('data.dict ERROR -- File "' + key + '" Skipped')
                        return
                    if not isinstance(data[keyl], dict):
                        if len(data[keyl]) > 0 and not isinstance(data[keyl][0], dict):
                            print(keyl, ' : data[key] ERROR -- File "' + key + '" Skipped')
                            return
                        else:
                            # Drill deep into the JSON structure to find data
                            tmp = defaultdict(list)
                            for i in range(len(data[keyl])):
                                for k in data[keyl][i]:
                                    if not (isinstance(data[keyl][i][k], dict)):
                                        tmp[k].append(data[keyl][i][k])
                                    else:
                                        for k2 in data[keyl][i][k]:
                                            if isinstance(data[keyl][i][k][k2], dict):
                                                for k3 in data[keyl][i][k][k2]:
                                                    tmp[str(k) + '_' + str(k2) + '-' + str(k3)].append(
                                                        data[keyl][i][k][j][k2][k3])
                                            else:
                                                tmp[str(k) + '_' + str(k2)].append(data[keyl][i][k][k2])

                        d = tmp

                    df = pd.DataFrame(d)
                    # df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

                    if df.empty:
                        print('df.empty ERROR - File "' + key + '" Skipped')
                        return

                    df.set_index('date', inplace=True)

                    csvfilename_out = directory + s + '_' + key + '.csv'
                    df.to_csv(csvfilename_out)

    except  urllib.error.URLError as e:
        print('ERROR ' + s + ' ' + e.reason, 'error')
        return
    except urllib.error.HTTPError as e:
        print('ERROR ' + s + ' ' + e.reason, 'error')
        return

def init_stocks_data(data_dir,stocks_list_file):
    if data_dir == '':
        data_dir = './data/'
    set_data_directory(data_dir)
    if stocks_list_file == '' :
        print ("Error: stocks_list_file is not set")
        return
    set_stocks_list_filename(stocks_list_file)
    stock_count, stock_fields, sp_df = GetStocksList()
    return stock_count, stock_fields, sp_df 

def get_sp_constituents(filename):
    stocklist = []
    if not filename:
        print("No filename passed.")
        #sp_cons_csv = directory + 'sp_const.csv'
        sp_cons_csv = directory + 'screener_results.csv'
        print("Using filename '"+sp_cons_csv,"'")
    else:
        sp_cons_csv = filename
        
    sp_df = pd.read_csv(sp_cons_csv)
    sp_df.sort_values('Symbol', ascending=True, inplace=True)
    sp_df.drop_duplicates(subset ="Symbol",  keep = "first", inplace = True) 
    stocklist = sp_df['Symbol'].str.lower()
    
    return stocklist

def SaveBuySellList2File(recomm_filename,recomm_df):
    recomm_df.to_csv(recomm_filename,index=False)
    print('Table writen to file "'+recomm_filename+'"')
        
def GenerateBuySellList(period,directory,argv,years=1,months=0,weeks=0,days=0):
    tcount = bcount = scount = 0
    
    print('Using stocks list in : ',directory+argv)
    stocklist = get_sp_constituents(directory+argv)
    stock_count, stock_fields, sp_df = init_stocks_data(directory,argv)
    #print(stocklist)
    
    lookback_date = str(datetime.date.today()- relativedelta(years=years,months=months,weeks=weeks,days=days))
    print("Start scanning from :"+lookback_date)
    
    rec_columns=['Symbol','Name','Sector','Recommendation','Close','BuyAt','SellAt']
    record = defaultdict(list) 
    recomm_list = []
    for s in stocklist:
        price_df = GetStockDataFrame(s)
        set_stock(s)
        stock_info = get_stock_info(s)
        if(price_df.empty):
            continue
            

        price_df = DatesRange(price_df, lookback_date) # limit the data since a specific past date or a range

        rc, price_df2, low_df, hi_df, action_df = GetBuySellEnvelope(s,price_df, period)
        if(not rc):
            #print('Data not available for stock '+s.upper())
            continue
        tcount = tcount +1
        if(action_df['Recommendation'][0] == 'buy'):
            bcount = bcount +1
            #print(str(bcount)+'-'+'BUY '+':'+s.upper()+','+stock_info['name'][0]+','+ stock_info['sector'][0],
            #',Close: $'+str(quote(s).close[0]),', Statement : ',action_df['Statement'][0])
            
            record = {'Symbol':s.upper(),'Name': stock_info['name'][0], 'Sector':stock_info['sector'][0],
                'Recommendation':'BUY','Close':quote(s).close[0],'BuyAt':action_df['BuyAt'][0],
                'SellAt':action_df['SellAt'][0],'Upside $': round( action_df['SellAt'][0] - quote(s).close[0],2),
                'Upside %': str( round( 100 * ((action_df['SellAt'][0] - quote(s).close[0] )/quote(s).close[0] )  ,2) )+'%',
                'Ave. Hold (days)': 0}

            recomm_list.append(record)
            
        if(action_df['Recommendation'][1] == 'sell'):
            scount = scount +1
            ## ToDo: Add records of Sell Recommendations here
        
    recomm_df = pd.DataFrame(recomm_list)
    
    return tcount, bcount, scount, recomm_df

