# Import some reqiured packages.
import pandas as pd
import numpy as np

# Build a class for Particle Swarm Optimization method.
class ParticleSwarmOptimization:
    """A Class for the steps of Particle Swarm Optimization method for portfolio optimization.
    
    Attributes:
        data (pandas dataframe)     Used stock price in portfolio dataframe
        number_of_iterations (int)  Desired number of iterations in the method


    
    Methods:


    """

    def __init__(
        self,
        data: pd.DataFrame,
        number_of_iterations: int,
        risk_free_rate: float,
        return_minimum: float,
        number_of_particles: int,
        absolute_move_limit: float,
        c1: float,
        c2: float,
        inertia_moment: float) -> None:
        """The function to initialize some algorithm given parameters and hyperparameters as the object's attributes then serve within class.
        
        Args:


        Returns:


        """
        # Set some local attributes.
        self.data = data
        self.n_tickers = len(self.data.columns)
        self.iterations = number_of_iterations
        self.risk_free_rate = risk_free_rate
        self.return_minimum = return_minimum
        self.number_of_particles = number_of_particles
        self.absolute_move_limit = absolute_move_limit
        self.c1 = c1
        self.c2 = c2
        self.inertia_moment = inertia_moment
        self.expected_return = self.series_mean()[1]
        self.sd_expected_return = self.series_sd()
        self.portfolio_vcov = self.series_vcov_matrix()
        
    #%% Solver
    def optimization_iteration(self) -> None:
        """Run """
        #
        weights, velocity, particle_fit_value, particle_best_position, particle_best_sharpe, global_best_sharpe, global_best_position, particle_return, particle_return_best_sharpe, particle_risk, particle_risk_best_sharpe = self.initialize_swarm()
        
        #
        position_iteration = []
        velocity_iteration = []
        global_best_position_iterations = []
        
        #
        for iteration in range(self.iterations):
            #
            position_iteration.append(weights)
            velocity_iteration.append(velocity)
            global_best_position_iterations.append(global_best_position)

            #
            weights, velocity = self.update_velocity_and_weights(weights=weights,
                                                                 velocity=velocity,
                                                                 particle_best_position=particle_best_position,
                                                                 global_best_position=global_best_position)
            
            #
            particle_fit_value, particle_best_position, particle_best_sharpe, particle_return_best_sharpe, particle_risk_best_sharpe = self.find_best_portfolio(
                weights=weights,
                particle_fit_value=particle_fit_value,
                particle_best_position=particle_best_position,
                particle_best_sharpe=particle_best_sharpe,
                particle_return=particle_return,
                particle_return_best_sharpe=particle_return_best_sharpe,
                particle_risk=particle_risk,
                particle_risk_best_sharpe=particle_risk_best_sharpe
            )

            #
            global_best_sharpe, global_best_position, global_return_best_sharpe, global_risk_best_sharpe = self.get_global_best(sharpe_ratio
                                                                                                                                =particle_best_sharpe,
                                                                                                                                weights=particle_best_position,
                                                                                                                                returns=particle_return_best_sharpe,
                                                                                                                                risks=particle_risk_best_sharpe
            )
        #
        print('Maximum Sharpe Ratio: {} with Return: {}, Risk: {}, Weights: {}'.format(np.round(global_best_sharpe, 4), np.round(global_return_best_sharpe, 4), np.round(global_risk_best_sharpe, 4), np.round(global_best_position, 4)))
        
        #
        return [global_best_position, global_best_sharpe, global_return_best_sharpe, global_risk_best_sharpe]
    
    #%% Each Step
    def initialize_swarm(self) -> np.array:
        """ """
        # 
        all_weights = []
        all_velocities = []

        #
        particle_fit_value = []
        particle_return = []
        particle_risk = []

        # 
        for particle in range(self.number_of_particles):
            #
            weights = self.generate_weights()

            #
            self.fitness_function(weights)

            #
            while self.fitness_function(weights)[0] < self.return_minimum:
                weights = self.generate_weights()
                self.fitness_function(weights)
            
            #
            all_weights.append(weights)

            #
            all_velocities.append(np.random.rand(self.n_tickers) * self.absolute_move_limit)

            #
            self.fitness_function(weights)
            particle_fit_value.append(self.fitness_function(weights)[2])
            particle_return.append(self.fitness_function(weights)[0])
            particle_risk.append(self.fitness_function(weights)[1])

        #
        particle_best_position = all_weights[:]
        particle_best_sharpe = particle_fit_value[:]
        particle_return_best_sharpe = particle_return[:]
        particle_risk_best_sharpe = particle_risk[:]

        #
        global_best_sharpe, global_best_position, _, _ = self.get_global_best(  particle_best_sharpe[:], 
                                                                                particle_best_position[:],
                                                                                particle_return_best_sharpe,
                                                                                particle_risk_best_sharpe)

        return [all_weights, all_velocities, particle_fit_value, particle_best_position,
                particle_best_sharpe, global_best_sharpe, global_best_position, 
                particle_return, particle_return_best_sharpe, particle_risk, particle_risk_best_sharpe]

    def fitness_function(
            self,
            weights: np.array) -> None:
        """The method to calculate fitness function for weights evaluation."""
        # Compute portfolio return for each weight combination.
        portfolio_return = np.sum((weights * self.expected_return) * 252)
        
        # Intialize an array for portfolio risk computation for each weight combination then looping over to compute it.
        portfolio_risk = np.sqrt(np.transpose(weights) @ self.portfolio_vcov @ weights)
    
        # Compute Sharpe ratio for each weight combination.
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk

        #
        return portfolio_return, portfolio_risk, sharpe_ratio
    
    def update_velocity_and_weights(
        self,
        weights: np.array,
        velocity: np.array,
        particle_best_position: np.array,
        global_best_position: np.array) -> np.array:
        """ Up """
        #
        r1, r2 = np.random.rand(), np.random.rand()

        #
        #print(particle_best_position)
        velocity_inertia = np.multiply(self.inertia_moment, velocity[:])
        velocity_cognitive = np.multiply(self.c1 * r1, np.subtract(particle_best_position[:], weights[:]))
        velocity_social = np.multiply(self.c2 * r2, np.subtract(global_best_position[:], weights[:]))
        
        #
        new_velocity = velocity_inertia[:] + velocity_cognitive[:] + velocity_social[:]
        
        #
        for particle in range(self.number_of_particles):
            for stock in range(self.n_tickers):
                #
                if new_velocity[particle][stock] > self.absolute_move_limit:
                    new_velocity[particle][stock] = self.absolute_move_limit

                #
                if new_velocity[particle][stock] < self.absolute_move_limit:
                    new_velocity[particle][stock] = -self.absolute_move_limit
        
        #
        new_weights = weights[:] + new_velocity[:]

        #
        for particle in range(self.number_of_particles):
            for stock in range(self.n_tickers):
                #
                if new_weights[particle][stock] < 0:
                    new_weights[particle][stock] = 1e-16

                #
                if new_weights[particle][stock] > 1:
                    new_weights[particle][stock] = 1
        
        #
        for particle in range(self.number_of_particles):
            #
            new_weights[particle] = self.normalize(given_array=np.array(new_weights[particle]))

        #
        new_weights = new_weights.tolist()
        new_velocity = new_velocity.tolist()

        #
        return [new_weights, new_velocity]
    
    def find_best_portfolio(
        self,
        weights: np.array,
        particle_fit_value: np.array,
        particle_best_position: np.array,
        particle_best_sharpe: np.array,
        particle_return: np.array,
        particle_return_best_sharpe: np.array,
        particle_risk: np.array,
        particle_risk_best_sharpe: np.array) -> np.array:
        """ Fi """
        #
        for particle in range(self.number_of_particles):
            #
            particle_fit_value[particle] = self.fitness_function(weights[:][particle])[2]

            #
            if particle_fit_value[particle] < particle_best_sharpe[particle]:
                #
                particle_best_sharpe[particle] = particle_fit_value[particle]

                #
                particle_return_best_sharpe[particle] = particle_return[particle]

                #
                particle_risk_best_sharpe[particle] = particle_risk[particle]

                #
                particle_best_position[particle] = weights[:][particle]
            
        #
        return [particle_fit_value, particle_best_position, particle_best_sharpe, particle_return_best_sharpe, particle_risk_best_sharpe]

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
    
    def generate_weights(self) -> None:
        """A function for generating initialization weights for each ticker as an init population."""
        # Generate random number as many as the population size then normalize by row in order to give sum to 1.
        weights = self.normalize(given_array = np.random.random(size = self.n_tickers))

        #
        return weights

    def normalize(
        self,
        given_array: np.array) -> None:
        """A scratch function to normalize an internal array.

        Returns:

        """

        # Return.
        return given_array / np.sum(given_array)

    def get_global_best(
            self,
            sharpe_ratio: np.array,
            weights: np.array,
            returns: np.array,
            risks: np.array) -> np.array:
        #
        max_value = np.max(sharpe_ratio)

        #
        max_index = sharpe_ratio.index(max_value)

        #
        max_position = weights[max_index][:]

        #
        return_max_value = returns[max_index]

        #
        risk_max_value = risks[max_index]

        #
        return [max_value, max_position, return_max_value, risk_max_value]