import os
import json
import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt

stock = ''
directory = ''  # '../getstockdata/data/'
sp_cons_csv = ''
fund_filename = ''
xcolumn_name = 'date'
column_map = {}


def set_stock(s):
    global stock
    global directory
    global sp_cons_csv
    global fund_filename

    if (directory == ''):
        raise NameError("Error: 'data directory' is NOT set")
        print('ERROR')
    else:
        stock = s
        sp_cons_csv = directory + 'sp_const.csv'
        sp_df = pd.read_csv(sp_cons_csv)
        if stock.upper() not in sp_df['Symbol'].values:
            raise NameError("Error: '" + stock + "' is NOT AVAILABLE")
        fund_filename = directory + stock + '_fund.csv'


def get_stock():
    return stock


def set_data_directory(datadir):
    global directory
    directory = datadir


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


def GetSP500_List():
    sp_df = pd.DataFrame()
    try:
        sp_df = pd.read_csv(sp_cons_csv)
        sp_df.sort_values('Symbol', ascending=True, inplace=True)
    except:
        print("Cannot open file", sp_cons_csv)

    # returns count_row, count_col, df
    return sp_df.shape[0], sp_df.shape[1], sp_df


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
    return df[(df.index >= start) & (df.index <= end)].copy()


def skip_every_n(df, n):
    df2 = df.iloc[::n, :]
    return df2


def plot_basic_charts():
    if (stock == ""):
        print('Error: Stock Symbol is NOT set.')
        return

    # Get stock company information
    sp_df = pd.read_csv(sp_cons_csv)
    stock_info = sp_df[sp_df.Symbol == stock.upper()]
    company_name = stock_info['Name'].iloc[0]
    company_sector = stock_info['Sector'].iloc[0]

    stock_title = stock.upper() + ' ' + '(' + company_name + ')' + ' [' + company_sector + ' Sector]'
    price_df = GetStockDataFrame(stock)

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
    # dates = pd.DataFrame({'date': main_df_org['cash-flow-statement'].index})
    # main_df['Timestamps​'] = dates
    # main_df = main_df.set_index('date')
    # print(dates)
    # print(price_df['Timestamps'][0])
    # main_df = sd.df_start_after_datetime(main_df, price_df['Timestamps'][0])

    # Calculate the moving averages
    MADays = [40, 100, 200]
    MA = [{}]
    ma_names = []
    for ma in MADays:
        mean_name = str(ma) + ' days MA'
        ma_names.append(mean_name)
        tmp_df = pd.DataFrame(
            {'Timestamps': price_df['Timestamps'].to_list(), mean_name: price_df['AdjClose'].rolling(window=ma).mean()})
        price_df = pd.merge(price_df, tmp_df, on='Timestamps')

    fig = plt.figure(figsize=(15, 6))

    # ***************************************************
    # Find the full trend line
    x = range(len(price_df['Timestamps']))
    fit = np.polyfit(x, price_df['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    print('5 Years Trend Line Function :' + str(fit_fn))

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df['5 Yr Trend'] = trend
    # ***************************************************

    twoyears_ago = datetime.datetime.today() - datetime.timedelta(days=2 * 365)
    price_df_2y = price_df[price_df['Timestamps'] > twoyears_ago].copy(deep=True)
    # price_df = price_df_2y
    # Find the 2 years trend line
    x = range(len(price_df_2y['Timestamps']))
    fit = np.polyfit(x, price_df_2y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    print('2 Years Trend Line Function :' + str(fit_fn))

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_2y['2 Yr Trend'] = trend
    # ***************************************************

    oneyears_ago = datetime.datetime.today() - datetime.timedelta(days=1 * 365)
    price_df_1y = price_df[price_df['Timestamps'] > oneyears_ago].copy(deep=True)
    # Find the 1 year trend line
    x = range(len(price_df_1y['Timestamps']))
    fit = np.polyfit(x, price_df_1y['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    print('1 Year Trend Line Function :' + str(fit_fn))

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_1y['1 Yr Trend'] = trend
    # ***************************************************

    sixmonth_ago = datetime.datetime.today() - datetime.timedelta(days=0.5 * 365)
    price_df_6m = price_df[price_df['Timestamps'] > sixmonth_ago].copy(deep=True)
    # Find the 6 months trend line
    x = range(len(price_df_6m['Timestamps']))
    fit = np.polyfit(x, price_df_6m['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    print('6 Months Trend Line Function :' + str(fit_fn))

    trend = []
    for k in x:
        t = fit_fn(k)
        trend.append(t)
    price_df_6m['6 Month Trend'] = trend
    # ***************************************************

    ts = pd.to_datetime(str(price_df['Timestamps'].iloc[0]))
    frm = ts.strftime('%Y-%m-%d')
    ts = pd.to_datetime(str(price_df['Timestamps'].iloc[-1]))
    to = ts.strftime('%Y-%m-%d')

    stock_title = stock_title + '\nFrom ' + frm \
                  + ' To ' + to

    ## Create Plot
    fig, axs = plt.subplots(4, figsize=(15, 25), sharex=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1, 1, 1]})
    fig.suptitle(stock_title, fontsize=30)

    y_list = ['Low', 'AdjClose', 'High']
    for ma in ma_names:
        y_list.append(ma)
    y_list.append('5 Yr Trend')

    price_df.plot(ax=axs[0], x='Timestamps', y=y_list,
                  title=stock_title)
    price_df_2y.plot(ax=axs[0], x='Timestamps', y=['2 Yr Trend'])
    price_df_1y.plot(ax=axs[0], x='Timestamps', y=['1 Yr Trend'])
    price_df_6m.plot(ax=axs[0], x='Timestamps', y=['6 Month Trend'], grid=True)

    axs[0].set_ylabel('Stock Price')

    # earn_df =

    dic = [{}]
    p_over_e_df = [{}]  # pd.DataFrame(columns=['Timestamps','Price/Earning'])

    i = 0
    main_df['EPS'] = MakeFund_Subset(main_df_org['income-statement'], ['EPS'])
    main_df['PE ratio'] = MakeFund_Subset(main_df_org['company-key-metrics'], ['PE ratio'])

    from_date = main_df['EPS'].index[i]

    to_date = main_df['EPS'].index[i + 1]
    for d in price_df['Timestamps']:
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

            kk = price_df[price_df['Timestamps'] == d]['AdjClose'].values[0]

            dic.append({'Timestamps': d, 'Calc. P/E': (kk / jj) / 4.0, 'EPS': jj,
                        'P/E': pp, 'FCF/Share': ff, 'Revenue': rr,
                        'Earnings before Tax': ee, 'Net Income': nn,
                        'Free Cash Flow': cc, 'Gross Profit': gg})

        else:
            continue

    plots_df = pd.DataFrame(dic).set_index('Timestamps').dropna()

    plots_df.plot(ax=axs[1], y=['EPS'], grid=True)
    plots_df.plot(ax=axs[1], y=['FCF/Share'], grid=True)
    axs[1].set_ylabel('Earning && FCF/Share')
    plots_df.plot(ax=axs[2], y=['P/E'], grid=True)
    plots_df.plot(ax=axs[2], y=['Calc. P/E'], grid=True)
    axs[2].set_ylabel('Price/Earning (P/E)')

    plots_df.plot(ax=axs[3], y=['Revenue'])
    plots_df.plot(ax=axs[3], y=['Earnings before Tax'])
    plots_df.plot(ax=axs[3], y=['Net Income'])
    plots_df.plot(ax=axs[3], y=['Free Cash Flow'])
    plots_df.plot(ax=axs[3], y=['Gross Profit'], grid=True)

    axs[3].set_ylabel('Growth')

    for ax in axs:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

def vital_data():
