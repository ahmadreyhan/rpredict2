# Import some required packages.
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

# Override datareader with Yahoo Finance
yf.pdr_override()

# Build a helper function to load data and return a dataframe.
def load_data(
    stock_list: list,
    start_date: str = '2021-01-04',
    end_date: str = '2023-03-01',
    price_type: str = 'Close') -> pd.DataFrame:
    """Load and fetch the daily stock price data using pandas-datareader from Yahoo Finance.
    
    Args:
        stock_list (list): A list contains the stock tickers.
        start_date (str): The start date of stock price data, there is a default start date.
        end_date (str): The end date of stock price data, there is a default start date.
        price_type (str): The desired stock price type ['Open', 'Close', 'Adj Close', 'High', 'Low'], 'Close' as default.
   
    Returns:
        A dataframe with daily stock price data corresponding to passed stickers.
    
    """

    ## Add the '.JK' suffix as the format of a IDX ticker in Yahoo Finance.
    tickers = [stock + '.JK' for stock in stock_list]
    
    ## Get a empty pandas dataframe.
    portfolio = pd.DataFrame()

    ## Loop over the `tickers` list then read the data of each ticker depends on passed price type.
    portfolio = pdr.get_data_yahoo(tickers,
                                   start=start_date,
                                   end = end_date)[price_type]
    
    ## Set the columns name of dataframe.
    portfolio.columns = stock_list

    ## Return of function.
    return portfolio