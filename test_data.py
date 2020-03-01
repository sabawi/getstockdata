
import stockdatalib as sd
from IPython.display import display
from IPython.display import HTML
import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt

def plot_stock_charts():
    stock_title = sd.stock.upper() + ' ' + '(' + company_name + ')' + ' [' + company_sector + ' Sector]'
    price_df = sd.GetStockDataFrame(sd.stock)
    mean_1 = price_df['AdjClose'].rolling(window=100).mean()
    mean_1_name = '100d Mov.Ave.'
    mean_1_df = pd.DataFrame({'Timestamps': price_df['Timestamps'].to_list(), mean_1_name: mean_1})
    price_df = pd.merge(price_df, mean_1_df, on='Timestamps')

    fig = plt.figure(figsize=(15, 6))

    # ***************************************************
    # Find the full trend line
    x = range(len(price_df['Timestamps']))
    fit = np.polyfit(x, price_df['AdjClose'].astype(float), 1)
    fit_fn = np.poly1d(fit)
    print(fit_fn)

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
    print(fit_fn)

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
    print(fit_fn)

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
    print(fit_fn)

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

    fig, axs = plt.subplots(3, figsize=(15, 20), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(stock_title, fontsize=30)

    price_df.plot(ax=axs[0], x='Timestamps', y=['Low', 'AdjClose', mean_1_name, 'High', '5 Yr Trend'],
                  title=stock_title)
    price_df_2y.plot(ax=axs[0], x='Timestamps', y=['2 Yr Trend'])
    price_df_1y.plot(ax=axs[0], x='Timestamps', y=['1 Yr Trend'])
    price_df_6m.plot(ax=axs[0], x='Timestamps', y=['6 Month Trend'], grid=True)

    # earn_df =

    dic = [{}]
    p_over_e_df = [{}]  # pd.DataFrame(columns=['Timestamps','Price/Earning'])

    i = 0
    main_df['EPS'] = sd.MakeFund_Subset(main_df_org['income-statement'], ['EPS'])
    main_df['PE ratio'] = sd.MakeFund_Subset(main_df_org['company-key-metrics'], ['PE ratio'])

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

            kk = price_df[price_df['Timestamps'] == d]['AdjClose'].values[0]

            dic.append({'Timestamps': d, 'Calc. P/E': (kk / jj) / 4.0, 'EPS': jj, 'P/E': pp})

        else:
            continue

    p_over_e_df = pd.DataFrame(dic).set_index('Timestamps').dropna()

    p_over_e_df.plot(ax=axs[1], y=['EPS'], grid=True)
    p_over_e_df.plot(ax=axs[2], y=['P/E'], grid=True)
    p_over_e_df.plot(ax=axs[2], y=['Calc. P/E'], grid=True)


# Set the data directory and the stock name
sd.set_data_directory('./data/')
sd.set_stock('aapl')

# Load S&P Stock list
stock_count, stock_fields, sp_df = sd.GetSP500_List()

# Get stock company information
stock_info = sp_df[sp_df.Symbol == sd.stock.upper()]
company_name = stock_info['Name'].iloc[0]
company_sector = stock_info['Sector'].iloc[0]


# Load stock End of Day prices file
price_df = sd.GetStockDataFrame(sd.stock)
# Index the data frame by 'Timestamps' column
price_df.set_index('Timestamps',inplace=True)

# Load the stock fundamental data
keys, column_map, main_df_org = sd.GetFund_Dict(sd.stock)

# Search for a key word in the data column names
f,s = sd.where_is_column('Revenue')
display(f.to_html())

# Limit the stock prices data in the dates range
price_df = sd.df_start_after_datetime(price_df, '2015-01-01','2016-06-01')

# Plot the prices chart
#price_df['AdjClose'].plot(figsize=(15,8)).grid()

# Create a subset of columns in a DataFrame from the stock fundamentals dataset
cols = ['Revenue','Earnings before Tax','Net Income','Free Cash Flow margin','Gross Profit','Net Profit Margin'] # [ 'Shares', 'Price', 'Revenue', 'Earnings']
main_df1 = sd.MakeFund_Subset(main_df_org['income-statement'],cols)
main_df1.plot()
# Create another subset of columns from another fundamentals dataset
cols = ['Number of Shares','Stock Price']
main_df2 = sd.MakeFund_Subset(main_df_org['enterprise-value'],cols)

# Create a 3rd subject of columns from yet another fundamentals dataset
cols = ['Capital Expenditure','Free Cash Flow']
main_df3 = sd.MakeFund_Subset(main_df_org['cash-flow-statement'],cols)

# Merge the 3 dataframes into a single dataframe
main_df = pd.merge(main_df1,main_df2,on='date')
main_df = pd.merge(main_df,main_df3,on='date')

#main_df.plot()
main_df['Net Income'].plot().grid()
main_df.plot(y=['Revenue','Earnings before Tax','Net Income','Free Cash Flow','Gross Profit'],figsize=(15,8)).grid()


plot_stock_charts()
