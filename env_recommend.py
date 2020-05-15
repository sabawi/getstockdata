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
from collections import defaultdict 
from datetime import datetime


stocklist = []

def get_sp_constituents_delete(filename):
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

def SaveBuySellList2File_delete(recomm_filename,recomm_df):
    recomm_df.to_csv(recomm_filename,index=False)
    print('Table writen to file "'+recomm_filename+'"')
        
def GenerateBuySellList_delete(directory,argv):
    global stocklist
    tcount = bcount = scount = 0
    
    print('Using stocks list in : ',directory+argv)
    get_sp_constituents(directory+argv)
    stock_count, stock_fields, sp_df = sd.init_stocks_data(directory,argv)
    #print(stocklist)
    
    rec_columns=['Symbol','Name','Sector','Recommendation','Close','BuyAt','SellAt']
    record = defaultdict(list) 
    recomm_list = []
    for s in stocklist:
        price_df = sd.GetStockDataFrame(s)
        sd.set_stock(s)
        stock_info = sd.get_stock_info(s)
        if(price_df.empty):
            continue
            
        price_df = sd.DatesRange(price_df, '2019-08-01') # limit the data since a specific past date or a range
        period = 28
        rc, price_df2, low_df, hi_df, action_df = sd.GetBuySellEnvelope(s,price_df, period)
        if(not rc):
            #print('Data not available for stock '+s.upper())
            continue
        tcount = tcount +1
        if(action_df['Recommendation'][0] == 'buy'):
            bcount = bcount +1
            #print(str(bcount)+'-'+'BUY '+':'+s.upper()+','+stock_info['name'][0]+','+ stock_info['sector'][0],
            #',Close: $'+str(sd.quote(s).close[0]),', Statement : ',action_df['Statement'][0])
            
            record = {'Symbol':s.upper(),'Name': stock_info['name'][0], 'Sector':stock_info['sector'][0],
                'Recommendation':'BUY','Close':sd.quote(s).close[0],'BuyAt':action_df['BuyAt'][0],
                'SellAt':action_df['SellAt'][0],'Upside $': round( action_df['SellAt'][0] - sd.quote(s).close[0],2),
                'Upside %': str( round( 100 * ((action_df['SellAt'][0] - sd.quote(s).close[0] )/sd.quote(s).close[0] )  ,2) )+'%',
                'Ave. Hold (days)': 0}

            recomm_list.append(record)
            
        if(action_df['Recommendation'][1] == 'sell'):
            scount = scount +1
            ## ToDo: Add records of Sell Recommendations here
        
    recomm_df = pd.DataFrame(recomm_list)
    
    return tcount, bcount, scount, recomm_df


def main_deleteme(argv):
    global stocklist
    tcount = bcount = scount = 0
    print(directory+argv)
    get_sp_constituents(directory+argv)
    stock_count, stock_fields, sp_df = sd.init_stocks_data('./data/',argv)
    #print(stocklist)
    rec_columns=['Symbol','Name','Sector','Recommendation','Close','BuyAt','SellAt']
    record = defaultdict(list) 
    recomm_list = []
    for s in stocklist:
        price_df = sd.GetStockDataFrame(s)
        sd.set_stock(s)
        stock_info = sd.get_stock_info(s)
        if(price_df.empty):
            continue
            
        price_df = sd.DatesRange(price_df, '2019-08-01') # limit the data since a specific past date or a range
        period = 28
        rc, price_df2, low_df, hi_df, action_df = sd.GetBuySellEnvelope(s,price_df, period)
        if(not rc):
            #print('Data not available for stock '+s.upper())
            continue
        tcount = tcount +1
        if(action_df['Recommendation'][0] == 'buy'):
            bcount = bcount +1
            print(str(bcount)+'-'+'BUY '+':'+s.upper()+','+stock_info['name'][0]+','+ stock_info['sector'][0],
            ',Close: $'+str(sd.quote(s).close[0]),', Statement : ',action_df['Statement'][0])
            
            record = {'Symbol':s.upper(),'Name': stock_info['name'][0], 'Sector':stock_info['sector'][0],
                'Recommendation':'BUY','Close':sd.quote(s).close[0],'BuyAt':action_df['BuyAt'][0],
                'SellAt':action_df['SellAt'][0],'Upside $': round( action_df['SellAt'][0] - sd.quote(s).close[0],2),
                'Upside %': str( round( 100 * ((action_df['SellAt'][0] - sd.quote(s).close[0] )/sd.quote(s).close[0] )  ,2) )+'%',
                'Ave. Hold (days)': 0}

            recomm_list.append(record)
            
        if(action_df['Recommendation'][1] == 'sell'):
            scount = scount +1
        
    print('Stocks processes = ',tcount)
    print('Buy Recommendations = ',bcount)
    print('Sell Recommendations = ',scount)
    
    recomm_df = pd.DataFrame(recomm_list)
    if not recomm_df.empty:
        recomm_filename = logdir + "stock_recommendations_" + timestampStr + ".csv"
        recomm_df.to_csv(recomm_filename,index=False)
        print('Table writen to file "'+recomm_filename+'"')
    #print(recomm_df)
          
def main(argv):
    directory = './data/'

    # Get the Buy Sell List
    period = 28
    tcount, bcount, scount, recomm_df = sd.GenerateBuySellList(period,directory,argv)
    print(recomm_df)
    dateTimeObj = datetime.now()
    # create a list of stock symbols
    timestampStr = dateTimeObj.strftime("%Y-%b-%d-%H-%M-%S")
    logdir = "./logs/"    
    
    if not recomm_df.empty:
        recomm_filename = logdir + "stock_recommendations_" + timestampStr + ".csv"
        sd.SaveBuySellList2File(recomm_filename,recomm_df)
    
if __name__ == "__main__":
    if(len(sys.argv)>1):
        a = sys.argv[1:]
        main(a[0])
    else:
        print("Error: Specify stocks filename as an argument ")
