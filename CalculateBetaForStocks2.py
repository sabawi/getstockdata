import yfinance as yf
import numpy as np

def calculate_beta(ticker: str, market: str,strPeriod) -> float:
    # Get the stock data
    stock_data = yf.Ticker(ticker).history(period=strPeriod)
    
    # Calculate the stock's returns
    stock_returns = stock_data["Close"].pct_change().dropna()
    
    # If a market index is specified, get the market data and calculate the market's returns
    if market:
        market_data = yf.Ticker(market).history(period=strPeriod)
        market_returns = market_data["Close"].pct_change().dropna()
    else:
        market_returns = np.ones_like(stock_returns)
    
    # Calculate the stock's beta using the returns
    beta = stock_returns.cov(market_returns) / market_returns.var()
    
    return beta

# Example usage: calculate the beta for Apple stock relative to the S&P 500
beta = calculate_beta("AAPL", "^GSPC","1y")
print(beta)
