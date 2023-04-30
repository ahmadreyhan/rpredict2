# Import some reqiured packages.
import pandas as pd
import numpy as np

# Build a class for Markowitz Model method.
class MonteCarloSimulation:
    """A Class for the steps of Monte Carlo Simulation method for portfolio optimization.
    
    Attributes:

    
    Methods:


    """

    def __init__(
        self,
        data: pd.DataFrame,
        number_of_iterations: int,
        risk_free_rate: float) -> None:
        """The function to initialize some algorithm given parameters and hyperparameters as the object's attributes then serve within class.
        
        Args:


        Returns:


        """
        # Set some local attributes.
        self.data = data
        self.n_tickers = len(self.data.columns)
        self.iterations = number_of_iterations
        self.risk_free_rate = risk_free_rate
        self.all_weights = np.zeros((self.iterations, self.n_tickers))
        self.portfolio_return_array = np.zeros(self.iterations)
        self.portfolio_risk_array = np.zeros(self.iterations)
        self.sharpe_ratio_array = np.zeros(self.iterations)
        self.expected_return = self.series_mean()[1]
        self.sd_expected_return = self.series_sd()
        self.portfolio_vcov = self.series_vcov_matrix()

    #%% Solver
    def  monte_carlo(self) -> None:
        """Run """
        #
        for index in range(self.iterations):
            #
            weights = self.generate_weights()

            #
            self.all_weights[index, :] = weights
            self.portfolio_return_array[index] = self.return_risk_function(weights)[0]
            self.portfolio_risk_array[index] = self.return_risk_function(weights)[1]
            self.sharpe_ratio_array[index] = self.return_risk_function(weights)[2]

            #
            #print('Iterations: {} | Portfolio Return: {} | Portfolio Risk: {} | Sharpe Ratio: {}'.format(index+1, self.portfolio_return_array[index], self.portfolio_risk_array[index], self.sharpe_ratio_array[index]))
        
        #
        self.store_result()

        #
        self.display_summaries()

    #%% Each Step
    def generate_weights(self) -> None:
        """A function for generating initialization weights for each ticker as an init population."""
        # Generate random number as many as the population size then normalize by row in order to give sum to 1.
        weights = self.normalize(given_array=np.array(np.random.random(self.n_tickers)))
        
        #
        return weights
    
    def return_risk_function(
            self,
            weights: np.array) -> float:
        """The method to calculate fitness function for weights evaluation."""
        # Compute portfolio return for each weight combination.
        portfolio_return = np.sum((weights * self.expected_return) * 252)
        
        # Intialize an array for portfolio risk computation for each weight combination then looping over to compute it.
        portfolio_risk = np.sqrt(np.transpose(weights) @ self.portfolio_vcov @ weights)

        #
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        #
        return portfolio_return, portfolio_risk, sharpe_ratio
        
    def store_result(self) -> None:
        #
        self.simulation_result = pd.DataFrame([self.portfolio_return_array,
                                               self.portfolio_risk_array,
                                               self.sharpe_ratio_array,
                                               self.all_weights]).T
        
        #
        self.simulation_result.columns = [
            'Returns',
            'Risks',
            'Sharpe Ratios',
            'Portfolio Weights'
        ]

        #
        self.simulation_result = self.simulation_result.infer_objects()

    def display_summaries(self) -> None:
        #
        #max_return = np.max(self.simulation_result['Returns'])
        #max_return_weights = self.simulation_result.loc[self.simulation_result['Returns']==max_return, 'Portfolio Weights'].iloc[-1]
        #print('Maximum Return: {} with Weights: {}'.format(max_return, max_return_weights))

        #
        max_sharpe_ratio = np.max(self.simulation_result['Sharpe Ratios'])
        max_sharpe_return = self.simulation_result.loc[self.simulation_result['Sharpe Ratios']==max_sharpe_ratio, 'Returns'].iloc[-1]
        max_sharpe_risk = self.simulation_result.loc[self.simulation_result['Sharpe Ratios']==max_sharpe_ratio, 'Risks'].iloc[-1]
        max_sharpe_weights = self.simulation_result.loc[self.simulation_result['Sharpe Ratios']==max_sharpe_ratio, 'Portfolio Weights'].iloc[-1]
        print('Maximum Sharpe Ratio: {} with Return: {}, Risk: {}, Weights: {}'.format(np.round(max_sharpe_ratio, 4), np.round(max_sharpe_return, 4), np.round(max_sharpe_risk, 4), np.round(max_sharpe_weights, 4)))

    #%% Properties
    def normalize(
        self,
        given_array: np.array) -> None:
        """A scratch function to normalize an internal array.

        Returns:

        """

        # Return.
        return given_array / np.sum(given_array)
    
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