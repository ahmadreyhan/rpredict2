# Import some required packages.
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Build a class for Genetic Algorithm method.ß
class GeneticAlgorithm:
    """A Class for the steps of Genetic Algorithm method for portfolio optimization.
    
    Attributes:

    
    Methods:


    """

    def __init__(
        self,
        data: pd.DataFrame,
        populations_size: int,
        number_of_generations: int,
        risk_free_rate: float,
        elitism_rate: float,
        crossover_rate: float,
        mutation_rate: float) -> None:
        """The function to initialize some algorithm given parameters and hyperparameters as the object's attributes then serve within class.
        
        Args:

        Returns:


        """
        # Set some local attributes.
        self.data = data
        self.n_tickers = len(self.data.columns)
        self.population_size = populations_size
        self.number_of_generations = number_of_generations
        self.risk_free_rate = risk_free_rate
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.expected_return = self.series_mean()[1]
        self.sd_expected_return = self.series_sd()
        self.portfolio_vcov = self.series_vcov_matrix()

    #%% Solver
    def optimization_iteration(self) -> None:
        """Run """
        #
        self.generate_weights()

        # 
        for generation in range(0, self.number_of_generations):
            self.fitness_function()
            self.elitism()
            self.selection()
            self.crossover()
            self.mutation()
            self.iteration_result_average()
            #print('Generation: {} | Number of Genes: {} | Max Sharpe Ratio: {} | Max Portfolio Return: {} | Min Portfolio Risk: {}'.format(generation, len(self.weights), self.max_sharpe, self.max_portfolio_return, self.min_portfolio_risk))

        #
        self.fitness_function()

        #
        self.optimized_solution()

    #%% Each Step
    def generate_weights(self) -> None:
        """A function for generating initialization weights for each ticker as an init population."""
        # Generate random number as many as the population size then normalize by row in order to give sum to 1.
        self.weights = self.normalize(given_array = np.random.random(size = (self.population_size, self.n_tickers)), normalize_type = 'row')
        
    def fitness_function(self) -> None:
        """The method to calculate fitness function for weights evaluation."""
        # Compute portfolio return for each weight combination.
        self.portfolio_return = np.sum((self.weights * self.expected_return) * 252, axis=1)
        
        # Intialize an array for portfolio risk computation for each weight combination then looping over to compute it.
        self.portfolio_risk = np.zeros(len(self.weights))
        for index in range(len(self.weights)):
            self.portfolio_risk[index] = np.sqrt(np.transpose(self.weights[index]) @ self.portfolio_vcov @ self.weights[index])
    
        # Compute Sharpe ratio for each weight combination.
        self.sharpe_ratio = (self.portfolio_return - self.risk_free_rate) / self.portfolio_risk

    def elitism(self) -> None:
        """Execute the elitism step by finding n highest sharpe ratios that depends on a given elistism rate."""
        # Get the number of n depends on the given elitism rate.
        n_elite_gen = round(len(self.sharpe_ratio) * self.elitism_rate)
        
        # Get the index of Sharpe ratios that detected as elit genes.
        elite_gen_index = (-self.sharpe_ratio).argsort()[:n_elite_gen]
        
        # Get a list that contains the index of non elit genes.
        self.non_elite_index = np.setdiff1d(range(len(self.sharpe_ratio)), elite_gen_index).tolist()

    def selection(self) -> None:
        """Execute the selection step by dividing the population to 2 groups, crossover and non crossover, then selecting crossover genes by random numbers and their probabilities."""
        #
        n_selections = round(len(self.non_elite_index) / 2)

        # 
        self.crossover_index = np.array([])

        #
        self.acc_sharpes = self.normalize(given_array=np.cumsum(self.sharpe_ratio[self.non_elite_index]), normalize_type='cumsum')
        
        #
        for _ in range(n_selections):
            rw_prob = rd.random()
            index = (np.abs(self.acc_sharpes - rw_prob)).argmin()
            self.crossover_index = np.append(self.crossover_index, index)
    
    def crossover(self) -> None:
        """Execute """
        #
        for index in range(0, int(len(self.crossover_index)/2), 2):
            cross_gene1, cross_gene2 = self.crossover_index[index], self.crossover_index[index+1]
            cross_gene1_weights, cross_gene2_weights = self.uniform_crossover(cross_gene1, cross_gene2)
            self.weights[int(cross_gene1)] = self.normalize(given_array=cross_gene1_weights, normalize_type='array')
            self.weights[int(cross_gene2)] = self.normalize(given_array=cross_gene2_weights, normalize_type='array')

    def mutation(self) -> None:
        """Execute """
        # 
        weight_n = len(self.crossover_index) * self.n_tickers
        mutate_gens = round(weight_n * self.mutation_rate)

        #
        if self.mutation_rate != 0:
            for _ in range(mutate_gens):
                random_index = int(np.random.choice(self.crossover_index))
                generation = self.weights[random_index]
                random_ticker = rd.randint(0, self.n_tickers-1)
                mu_gen = generation[random_ticker]
                mutated_index = mu_gen * np.random.normal(0, 1)
                generation[random_ticker] = abs(mutated_index)
                generation = self.normalize(given_array=generation, normalize_type='array')
                self.weights[random_index] = generation

    #@property_method
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

    def normalize(
        self,
        given_array: np.array,
        normalize_type: str) -> None:
        """A scratch function to normalize an internal array.
        
        Args:


        Returns:

        """

        # Build some conditional logics corresponding to the type of normalization and their formulas.
        if normalize_type == 'cumsum':
            normalized_array = given_array / given_array[len(given_array) - 1]
        elif normalize_type == 'row':
            normalized_array = given_array / np.sum(given_array, axis=1)[:, np.newaxis]
        elif normalize_type == 'array':
            normalized_array = given_array / np.sum(given_array)

        # Give back the array result of normalization as the function return.
        return normalized_array 

    def uniform_crossover(
        self,
        gene1: np.array,
        gene2: np.array) -> np.array:
        """
        Args:

    
        Returns:

        """
        #
        w_one = self.weights[int(gene1)]
        w_two = self.weights[int(gene2)]

        #
        prob = np.random.normal(1, 1, self.n_tickers)

        #
        for index in range(0, len(prob)):
            if prob[index] > self.crossover_rate:
                w_one[index], w_two[index] = w_two[index], w_one[index]

        #
        return w_one, w_two

    #%% Result Display
    def iteration_result_average(self) -> float:
        """Compute the average result (Sharpe ratio, portofolio return and risk) each iteration."""
        #
        self.max_sharpe = round(np.max(self.sharpe_ratio), 5)

        #
        self.max_portfolio_return = round(np.max(self.portfolio_return), 5)

        #
        self.min_portfolio_risk = round(np.min(self.portfolio_risk), 5)

    def optimized_solution(self) -> None:
        """Find the optimal solution after iterations."""
        # Extract the index of optimal solution which is the highest of Sharpe ratio.
        optimal_index = np.argmax(self.sharpe_ratio)

        #
        self.maximum_sharpe = self.sharpe_ratio[optimal_index]
        self.return_maximum_sharpe = self.portfolio_return[optimal_index]
        self.risk_maximum_sharpe = self.portfolio_risk[optimal_index]
        self.optimal_weight = self.weights[optimal_index]

        #
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        print('Maximum Sharpe Ratio: {} with Return: {}, Risk: {}, Weights: {}'.format(np.round(self.maximum_sharpe, 4), np.round(self.return_maximum_sharpe,4), np.round(self.risk_maximum_sharpe,4), np.round(self.optimal_weight, 4)))

    def plot_efficient_frontier(self) -> None:
        """Plot ."""
        #
        used_cmap = plt.cm.get_cmap('RdY1Bu')

        #
        fig = plt.scatter(self.portfolio_return, self.portfolio_risk, c=self.sharpe_ratio, cmap=used_cmap)
        plt.colorbar(fig)
        plt.xlabel('Portfolio Risks')
        plt.ylabel('Portfolio Returns')
        plt.title('Portfolio GA Optimization & Efficient Frontier')
        plt.show()