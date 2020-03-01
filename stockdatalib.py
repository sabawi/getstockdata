import os
import json
import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt

stock = ''
directory = '' #'../getstockdata/data/'
sp_cons_csv = ''
fund_filename = '' 
xcolumn_name = 'date'
column_map = {}

def set_stock(s):
    global stock
    global directory
    global sp_cons_csv
    global fund_filename

    if(directory==''):
        raise NameError("Error: 'data directory' is NOT set")
        print('ERROR')
    else:
        stock = s
        sp_cons_csv = directory+'sp_const.csv'
        sp_df = pd.read_csv(sp_cons_csv)
        if stock.upper() not in sp_df['Symbol'].values:
            raise NameError("Error: '"+stock+"' is NOT AVAILABLE")
        fund_filename = directory+stock+'_fund.csv'

def get_stock():
    return stock

def set_data_directory(datadir):
    global directory
    directory = datadir


def parser(x):
    return datetime.datetime.strptime(x,"%Y-%m-%d")

def days_count(from_date, to_date=datetime.datetime.today().date()):
    if isinstance(from_date, datetime.datetime) is False:
        from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d").date()
    if isinstance(to_date, datetime.datetime) is False:
        to_date = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
       
    delta = abs((to_date - from_date).days)
    return delta

def time_series_trendline(datetime_axis,values):
    x = range(len(datetime_axis))
    y = values
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)

    print('Slope = ', fit[0], ", ", "Intercept = ", fit[1])
    print('Line Equation :  ', fit_fn)

    fig = plt.figure(figsize=(15,6))
    ax  = fig.add_subplot(111).grid()
    plt.plot(values.index, y, label="Data")
    plt.plot(values.index,fit_fn(x), 'b-',label="Trend Line",color='red')
    _=plt.xticks(rotation=45)
    plt.show()


def GetSP500_List():
    sp_df = pd.DataFrame()
    try:
        sp_df = pd.read_csv(sp_cons_csv)
        sp_df.sort_values('Symbol', ascending=True, inplace=True)
    except:
        print("Cannot open file", sp_cons_csv)

    # returns count_row, count_col, df
    return  sp_df.shape[0], sp_df.shape[1], sp_df


def GetYahooStockData(symbol):
    # create filename and read it
    fname = directory + symbol + '.json'
    with open(fname, 'r') as rfile:
        read_content = json.load(rfile)

    return read_content


def GetStockDataFrame(symbol):
    data = GetYahooStockData(symbol)
    df = pd.DataFrame()

    df['Timestamps'] = pd.to_datetime(data["chart"]["result"][0]["timestamp"], unit='s')
    df["Open"] = data["chart"]["result"][0]["indicators"]["quote"][0]["open"]
    df["High"] = data["chart"]["result"][0]["indicators"]["quote"][0]["high"]
    df["Low"] = data["chart"]["result"][0]["indicators"]["quote"][0]["low"]
    df["AdjClose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
    df['Volume'] = data["chart"]["result"][0]["indicators"]["quote"][0]["volume"]
    df['Volume'] = df['Volume'].astype(int)

    return df


def GetFund_Dict(stock):
    global column_map
    sources = ['income-statement', 'cash-flow-statement', 'balance-sheet-statement',
               'enterprise-value', 'company-key-metrics', 'financial-ratios']
    filename_list = {}
    fund_df_list = {}

    for src in sources:
        filename_list[src] = directory + stock.lower() + '_' + src + '.csv'
        fund_df_list[src] = pd.DataFrame()

    # fund_df = pd.DataFrame(columns = sources)

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


def where_is_column(name):
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

def df_start_after_datetime(df, start, end=datetime.datetime.today()):
    return df[ (df.index >= start) & (df.index <= end)].copy()

def skip_every_n(df, n):
    df2 = df.iloc[::n, :]
    return df2

