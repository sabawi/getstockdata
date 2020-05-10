#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "sabawi"
__date__ = "$May 9, 2020 11:22:35 AM$"

import os
import sys
import json
from datetime import datetime
import pandas as pd
import stockdatalib as sd

# create a list of stock symbols
directory = './data/'

stocklist = []

def get_sp_constituents(filename):
    global stocklist
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



def main(argv):
    global stocklist
    tcount = bcount = scount = 0
    print(directory+argv)
    get_sp_constituents(directory+argv)
    stock_count, stock_fields, sp_df = sd.init_stocks_data('./data/',argv)
    #print(stocklist)
    for s in stocklist:
        price_df = sd.GetStockDataFrame(s)
        sd.set_stock(s)
        stock_info = sd.get_stock_info(s)
        if(price_df.empty):
            continue
            
        price_df = sd.DatesRange(price_df, '2019-08-01') # limit the data since a specific past date or a range
        rc, price_df2, low_df, hi_df, action_df = sd.GetBuySellEnvelope(s,price_df, 5)
        if(not rc):
            #print('Data not available for stock '+s.upper())
            continue
        tcount = tcount +1
        if(action_df['Recommendation'][0] == 'buy'):
            bcount = bcount +1
            print(str(bcount)+'-'+'BUY '+':'+s.upper()+','+stock_info['name'][0]+','+ stock_info['sector'][0],
            ',Close: $'+str(sd.quote(s).close[0]),', Statement : ',action_df['Statement'][0])
            
        if(action_df['Recommendation'][0] == 'sell'):
            scount = scount +1
        
    print('Stocks processes = ',tcount)
    print('Buy Recommendations = ',bcount)
    print('Sell Recommendations = ',scount)
        
            
    
if __name__ == "__main__":
    if(len(sys.argv)>1):
        a = sys.argv[1:]
        main(a[0])
    else:
        print("Error: Specify stocks filename as an argument ")
