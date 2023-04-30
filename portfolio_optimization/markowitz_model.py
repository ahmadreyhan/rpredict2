# Import some required packages.
import pandas as pd
import numpy as np

# Build a class for Markowitz Model method.
class MarkowitzModel:
    """A Class for the steps of Markowitz Model method for portfolio optimization.
    
    Attributes:

    
    Methods:


    """

    def __init__(
        self,
        data: pd.DataFrame,
        return_minimum: float,
        risk_free_rate: float) -> None:
        """The function to initialize some algorithm given parameters and hyperparameters as the object's attributes then serve within class.
        
        Args:

        Returns:


        """
        # Set some local attributes.
        self.data = data
        self.n_tickers = len(self.data.columns)
        self.given_minimum_return = return_minimum
        self.risk_free_rate = risk_free_rate
        self.expected_return = self.series_mean()[1]
        self.sd_expected_return = self.series_sd()
        self.portfolio_vcov = self.series_vcov_matrix()

    #%% Solver
    def  markowitz_solver(self) -> None:
        """Run """
        #
        one_vector = np.ones(self.n_tickers)
        sigma_invers = np.linalg.inv(np.array(self.portfolio_vcov))
        
        #
        a = np.dot(np.transpose(self.expected_return), np.dot(sigma_invers, self.expected_return))
        b = np.dot(np.transpose(self.expected_return), np.dot(sigma_invers, one_vector))
        c = np.dot(np.transpose(one_vector), np.dot(sigma_invers, one_vector))
        
        #
        self.optimum_weights = (1/(a*c - b**2)) * np.dot(sigma_invers, ((c*self.given_minimum_return-b)*self.expected_return + (a-b*self.given_minimum_return)*one_vector))
        
        #
        self.optimized_solution()

    #%% Properties
    def series_mean(self) -> np.array:
        """Compute the mean of each ticker.

        Returns:
            returns: A matrix contains the daily return of each ticker.
            tick_expected_return: An array contains the expected return of each ticker.

        """
        # Compute daily return of the stock price data for each tick.
        #returns = np.log(self.data / self.data.shift(1))
        returns = (self.data - self.data.shift(1)) / self.data.shift(1)
  
        # Get the average of daily return of each stock ticker.
        tick_expected_returns = np.array(returns.mean())
        
        # Give daily return and the average as the function returns.
        return returns, tick_expected_returns

    def series_sd(self) -> np.array:
        """Compute standard deviation of each ticker series mean.
        
        Returns:
            An array contains the standard deviation of each ticker's expected return.

        """
        # Compute standard deviation and give back as a function return.
        return np.array(np.std(self.data, axis=0))
    
    def series_vcov_matrix(self) -> np.array:
        """Compute the variance-covarince matrix of tickers daily return.
        
        Returns:
            The variance-covariance matrix of tickers expected return.

        """
        # Get the daily return from a helper function.
        daily_returns = self.series_mean()[0]

        # Get the variance-covariance matrix then annualize it by multiply with 252 as the number of effective days a year.
        vcov_matrix = daily_returns.cov() * 252
        
        # Give back variance-covariance matrix as the function return.
        return vcov_matrix
    
    #%% Result Display
    def optimized_solution(self) -> float:
        """Compute the average result (Sharpe ratio, portofolio return and risk)."""
        #
        print('Optimum Weights: {}'.format(np.round(self.optimum_weights, 4)))
        
        #
        print('Optimum Risk: {}'.format(np.round(np.sqrt(np.dot(np.transpose(self.optimum_weights), np.dot(self.portfolio_vcov, self.optimum_weights))), 4)))

        #
        print('Optimum Return: {}'.format(np.round(np.sum((self.expected_return * self.optimum_weights) * 252), 4)))

         #
        print('Sharpe Ratio: {}'.format(np.round((np.sum((self.expected_return * self.optimum_weights) * 252) - self.risk_free_rate) / np.sqrt(np.dot(np.transpose(self.optimum_weights), np.dot(self.portfolio_vcov, self.optimum_weights))), 4)))

        #
        print('Sum of Weights: {}'.format(np.round(np.sum(self.optimum_weights), 4)))