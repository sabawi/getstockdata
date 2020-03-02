#!/usr/bin/env python
# coding: utf-8
import stockdatalib as sd

# # Must set the data subdirectory
# Set the data directory and the stock name
sd.set_data_directory('./data/')


# # Load Data and plot a stock
# Load S&P Stock list
stock_count, stock_fields, sp_df = sd.GetSP500_List()

# Set stock symbol
sd.set_stock('jpm')

stock_info = sd.get_stock_info('jpm')
# Get stock company information
company_name = stock_info['name'][0]
company_sector = stock_info['sector'][0]

# Plot stock basic charts
linear_trends = sd.plot_basic_charts('jpm')

# Linear Regression Equations returned by the plot call
linear_trends

# Get last price quote
sd.quote('csco')

# Get last key stats on the stock
sd.key_stat('ibm')

# Plot another stock
sd.plot_basic_charts('amzn')

# # The data load functions

# Load stock End of Day prices file
price_df = sd.GetStockDataFrame('fb')

# Index the data frame by 'Timestamps' column
price_df.set_index('Timestamps',inplace=True)

# Load the stock fundamental data
keys, column_map, main_df_org = sd.GetFund_Dict('fb')


# # Searching for data by column name
# Search for a key word in the data column names
table,text = sd.FindColumn('Revenue')
table

# Now we know where 'Capex to Revenue' is, we can load it from the DataFrame source 'company-key-metrics' as 
# the code below
main_df_org['company-key-metrics']['Capex to Revenue'].plot(figsize=(12,8),
                                                            grid=True, 
                                                            title=sd.get_stock().upper()+' Capex to Revenue')


# # Selecting data range from price data frame

# Limit the stock prices data in the dates range
price_df = sd.DatesRange(price_df, '2016-01-01', '2018-06-01')

# Plot the prices chart
ax1 = price_df['AdjClose'].plot(figsize=(15,8),
                                title=sd.get_stock().upper()+" Plot selected date range").grid()


# # Merging DataFrames from multiple sources into one DF
import pandas as pd

# Create a subset of columns in a DataFrame from the stock fundamentals dataset
cols = ['Revenue','Earnings before Tax','Net Income','Free Cash Flow margin','Gross Profit','Net Profit Margin'] # [ 'Shares', 'Price', 'Revenue', 'Earnings']
main_df1 = sd.MakeFund_Subset(main_df_org['income-statement'],cols)

# Create another subset of columns from another fundamentals dataset
cols = ['Number of Shares','Stock Price']
main_df2 = sd.MakeFund_Subset(main_df_org['enterprise-value'],cols)

# Create a 3rd subject of columns from yet another fundamentals dataset
cols = ['Capital Expenditure','Free Cash Flow']
main_df3 = sd.MakeFund_Subset(main_df_org['cash-flow-statement'],cols)

# Create a 4th subject of columns from yet another fundamentals dataset
cols = ['Free Cash Flow per Share']
main_df4 = sd.MakeFund_Subset(main_df_org['company-key-metrics'],cols)


# Merge the 3 dataframes into a single dataframe
main_df = pd.merge(main_df1,main_df2,on='date')
main_df = pd.merge(main_df,main_df3,on='date')
main_df = pd.merge(main_df,main_df4,on='date')

main_df.plot(y=['Revenue','Earnings before Tax','Net Income','Free Cash Flow','Gross Profit'],
             figsize=(15,8),title=sd.get_stock().upper()).grid()


# # Using DatesRange() on fundamental data
sd.DatesRange(main_df, '2016-01-01', '2019-06-01').plot(figsize=(12,8),
                                                                     grid=True,
                                                                     title=sd.get_stock().upper()+' From 2016-01-01 TO 2019-06-01')

# Plot price chart with trend lines, EPS, and PE ratios
sd.plot_basic_charts('nflx')


# # Plot a single key stat item
main_df['Net Income'].plot(figsize=(12,8), title = sd.get_stock().upper()).grid()

price_data_df, maxtable_df = sd.TrendsPlot('c')

maxtable_df

price_data_df.describe()

f,s = sd.FindColumn('cash flow')
f

