# /usr/bin/env python3
""" Auth: Al Sabawi Feb 2020 """
import os
import json
import logging
import time
import urllib
from datetime import datetime
from urllib import request as url_request
import pandas as pd
import requests
from urllib import request
from urllib.request import urlopen
from collections import defaultdict

# create a list of stock symbols
directory = './data/'
data_range = '5y'  # 2y to 5y

stocklist = []

dateTimeObj = datetime.now()

timestampStr = dateTimeObj.strftime("%Y-%b-%d-%H-%M-%S")
logdir = "./logs/"
logfilename = logdir + "log_" + timestampStr + ".txt"
print('Log file name : ' + logfilename)


def print_over(txt):
    print('\r', end='')
    print(txt, end='')


def logstart():
    # Make sure the log dir exists, if not create it
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    format_str = '%(asctime)s - %(funcName)s - %(lineno)d :  %(message)s'

    logging.basicConfig(format=format_str, filename=logfilename, level=logging.INFO)

    # create logger
    logger = logging.getLogger('mainlogger')

    ch = logging.StreamHandler()

    # create formatter
    formatter = logging.Formatter(format_str)

    # add formatter to ch
    ch.setFormatter(formatter)

    logging.info('Started')

    # add ch to logger
    logger.addHandler(ch)


def logevent(str, type='info'):
    loggername = logging.getLogger(__name__).name

    if type == 'info':
        logging.info(loggername + ' ' + str)
    if type == 'warning':
        logging.warning(loggername + ' ' + str)
    if type == 'error':
        logging.error(loggername + ' ' + str)


def get_sp_constituents():
    global stocklist
    sp_cons_csv = directory + 'sp_const.csv'
    sp_df = pd.read_csv(sp_cons_csv)
    sp_df.sort_values('Symbol', ascending=True, inplace=True)
    stocklist = sp_df['Symbol'].str.lower()


# This utility function myprint() is used only when you want to traverse through a JSON structure
# to find where the nodes are
def myprint(d, level=0):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v, level + 1)
        else:
            print('-' * level, "{0} : {1}".format(k, v))


def load_data_from_file(symbol):
    # create filename and read it
    fname = directory + symbol + '.json'
    with open(fname, 'r') as rfile:
        read_content = json.load(rfile)
    return read_content


def download_stock_data():
    # make 2 parts of the yahoo URL so we can insert the symbol in between them
    urlA = 'https://query1.finance.yahoo.com/v7/finance/chart/'
    urlB = '?range=' + data_range + '&interval=1d&indicators=quote&includeTimestamps=true'

    for s in stocklist:
        print_over('-- Processing ... ' + s.upper())
        # Create the yahoo finance URL
        url = urlA + s + urlB
        logevent(url)

        # Request the data
        try:
            stockdata = requests.get(url)
            data = stockdata.json()
        except  urllib.error.URLError as e:
            logevent('ERROR ' + s + ' ' + e.reason, 'error')
            continue
        except urllib.error.HTTPError as e:
            logevent('ERROR ' + s + ' ' + e.reason, 'error')
            continue

        # write data into a json filw
        try:
            with open(directory + s + '.json', 'w') as f:
                json.dump(data, f, indent=4)
        except:
            logevent("ERROR: Opening/Writing file ", directory + s + '.json', 'error')


def download_stock_fund(Source='STOCKPOP'):
    global stocklist
    logevent("Downloading " + str(len(stocklist)) + " stocks' fundementals FROM " + Source.upper())
    url = 'UN-INITIALIZED'
    tmp_csvfilename = 'Temporary' + '_fund.csv'
    url_list = []
    stock_count = 0
    for s in stocklist:
        time.sleep(2.0)
        s = s.upper()
        print_over('-- Processing ... ' + s)
        stock_count = stock_count + 1
        logevent('Processing ' + s + ' No.' + str(stock_count))

        if Source.upper() == 'STOCKPOP':
            replace_symb = {'AAL': 'AMR', 'ANDV': 'TSO', 'ANTM': 'WLP', 'AON': 'AOC', 'ARNC': 'AA', 'ATGE': 'DV',
                            'BEAM': 'FO'}
            exclude_symbols = ['GOOGL']
            if s in replace_symb.keys():
                logevent(s + ' is being replaced with ' + replace_symb[s])
                s = replace_symb[s]

            if s in exclude_symbols:
                logevent("Skipping " + s)
                continue
            url = 'http://www.stockpup.com/data/' + s + '_quarterly_financial_data.csv'
        elif Source.upper() == 'FINANCIALMODELINGPREP' or 'FMP':
            replace_symb = {}
            exclude_symbols = ['SCG']
            if s in replace_symb.keys():
                logevent(s + ' is being replaced with ' + replace_symb[s])
                s = replace_symb[s]

            if s in exclude_symbols:
                logevent("Skipping " + s)
                continue

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
            logevent('ERROR ' + s + ': ' + '*** Error: Invalid data source URL', url, ' ***')
            return

        try:
            if Source.upper() == 'STOCKPOP':
                logevent(url)
                s = s.lower()
                csvfilename = './data/' + s + '_fund.csv'
                fund_data = url_request.urlopen(url).read().decode('utf-8')
                with open(csvfilename, 'w') as csvfile:
                    csvfile.write(fund_data)

            elif Source.upper() == 'FINANCIALMODELINGPREP' or 'FMP':
                s = s.lower()
                for key in url_list:
                    url = url_list[key]
                    logevent(url)
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
                            logevent(s + ' data.dict Empty ERROR -- Skipped')
                            continue
                        elif 'Error' in data.keys():
                            # print(s + ' data.dict ', data['Error'], ' ERROR -- Skipped')
                            logevent(s + ' data.dict ' + str(data['Error']) + ' ERROR -- Skipped')
                            continue

                        l = list(data.keys())
                        keyl = l[1]

                        d = defaultdict(list)
                        if not isinstance(data, dict) or keyl not in data.keys():
                            logevent('data.dict ERROR -- File "' + key + '" Skipped')
                            continue
                        if not isinstance(data[keyl], dict):
                            if len(data[keyl]) > 0 and not isinstance(data[keyl][0], dict):
                                logevent(keyl, ' : data[key] ERROR -- File "' + key + '" Skipped')
                                continue
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
                            logevent('df.empty ERROR - File "' + key + '" Skipped')
                            continue

                        df.set_index('date', inplace=True)

                        csvfilename_out = directory + s + '_' + key + '.csv'
                        df.to_csv(csvfilename_out)

        except  urllib.error.URLError as e:
            logevent('ERROR ' + s + ' ' + e.reason, 'error')
            continue
        except urllib.error.HTTPError as e:
            logevent('ERROR ' + s + ' ' + e.reason, 'error')
            continue


def stock_info(symbol):
    data = load_data_from_file(symbol)
    df = pd.DataFrame()
    data_dic = {
        "currency": [data["chart"]["result"][0]["meta"]["currency"]],
        "symbol": [data["chart"]["result"][0]["meta"]["symbol"]],
        "exchangeName": [data["chart"]["result"][0]["meta"]["exchangeName"]],
        "instrumentType": [data["chart"]["result"][0]["meta"]["instrumentType"]],
        "firstTradeDate": [datetime.fromtimestamp(data["chart"]["result"][0]["meta"]["firstTradeDate"])],
        "regularMarketTime": [datetime.fromtimestamp(data["chart"]["result"][0]["meta"]["regularMarketTime"])],
        "gmtoffset": [data["chart"]["result"][0]["meta"]["gmtoffset"]],
        "timezone": [data["chart"]["result"][0]["meta"]["timezone"]],
        "exchangeTimezoneName": [data["chart"]["result"][0]["meta"]["exchangeTimezoneName"]],
        "regularMarketPrice": [data["chart"]["result"][0]["meta"]["regularMarketPrice"]],
        "chartPreviousClose": [data["chart"]["result"][0]["meta"]["chartPreviousClose"]],
        "priceHint": [data["chart"]["result"][0]["meta"]["priceHint"]]
    }

    df = pd.DataFrame.from_dict(data_dic, orient='index', columns=['Values'])
    return (df)


def stock_dataframe(symbol):
    data = load_data_from_file(symbol)
    df = pd.DataFrame()
    df['Timestamps'] = pd.to_datetime(data["chart"]["result"][0]["timestamp"], unit='s')
    df["Open"] = data["chart"]["result"][0]["indicators"]["quote"][0]["open"]
    df["High"] = data["chart"]["result"][0]["indicators"]["quote"][0]["high"]
    df["Low"] = data["chart"]["result"][0]["indicators"]["quote"][0]["low"]
    df["AdjClose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]

    return df


def download_sp_constituents():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    try:
        data = request.urlopen(url).read().decode('utf-8')
    except:
        print("Error: Downloading SP Symbols from " + url)
        return

    os.makedirs("./data", exist_ok=True)
    with open('./data/sp_const.csv', 'w') as csvfile:
        csvfile.write(data)

def download_stocks_list():
    url = "https://financialmodelingprep.com/api/v3/company/stock/list"
    try:
        data = request.urlopen(url).read().decode('utf-8')
    except:
        print("Error: Downloading General Symbols list from " + url)
        return

    os.makedirs("./data", exist_ok=True)

    # write data into a json file
    try:
        with open('./data/stocks_list.json', 'w') as f:
            json.dump(data, f, indent=4)
    except:
        logevent("ERROR: Opening/Writing file './data/stocks_list.json'", 'error')



def main():
    logstart()
    print("Wait! Download in progress ... this may take a while")
    # uncomment the line below when done debugging
    print("-- Downloading stocks S&P list!")
    download_sp_constituents()

    print("-- Downloading general stocks list!")
    download_stocks_list()

    # Get S&P 500 Constituents, their Sectors, and Industry
    get_sp_constituents()
    print("-- Done!")

    # Get S&P 500 historical prices
    print("-- Downloading latest End of Day pricing data ...")
    download_stock_data()
    print("-- Done!")

    # Get S&P 500 stock fundamentals
    print("-- Downloading fundamental company data ...")
    fund_sources = ['STOCKPOP', 'FMP']
    download_stock_fund(fund_sources[1])

    logevent("Done!")
    print('********* Data Download Completed *********')


if __name__ == "__main__":
    main()
