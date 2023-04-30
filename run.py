# Import some required packages and dependencies.
from data_loader import load_data
from portfolio_optimization.markowitz_model import *
from portfolio_optimization.monte_carlo_simulation import *
from portfolio_optimization.genetic_algorithm import *
from portfolio_optimization.particle_swarm_optimization import *

# Define some used IDX tickers in the analysis.
tickers = ['ASII', 'BBCA', 'BBNI', 'BBRI', 'BMRI', 'DSNG', 
           'INDF', 'JSMR', 'KLBF', 'PGAS', 'PTPP', 'SIDO',
           'SMGR', 'TLKM', 'UNTR', 'UNVR', 'WIKA']

# Load the stock price data using a built function
stock_data = load_data(
    stock_list=tickers,
    start_date='2021-01-04',
    end_date='2022-12-14')

# Execute Markowitz Model method
markowitz_model = MarkowitzModel(
    data=stock_data,
    return_minimum=0.02,
    risk_free_rate=0.0)
print('=' * 80)
print('MARKOWITZ MODEL SUMMARIES')
markowitz_model.markowitz_solver()

# Execute Monte Carlo Simulation method
monte_carlo_simulation = MonteCarloSimulation(
    data=stock_data,
    number_of_iterations=5000,
    risk_free_rate=0.0)
print('=' * 80)
print('MONTE CARLO SIMULATION SUMMARIES')
monte_carlo_simulation.monte_carlo()

# Execute Genetic Algorithm method
ga = GeneticAlgorithm(
    data = stock_data,
    populations_size = 1000,
    number_of_generations = 1000,
    risk_free_rate = 0.0,
    elitism_rate = 0.25,
    crossover_rate = 0.4,
    mutation_rate = 0.01
)
print('=' * 80)
print('GENETIC ALGORITHM SUMMARIES')
ga.optimization_iteration()

#
#ga.plot_efficient_frontier()

# Execute Particle Swarm Optimization method
pso = ParticleSwarmOptimization(
    data=stock_data,
    number_of_iterations=1000,
    risk_free_rate=0.0,
    return_minimum=0.2,
    number_of_particles=1000,
    absolute_move_limit=0.2,
    c1=0.4,
    c2=1.7,
    inertia_moment=0.9
)
print('=' * 80)
print('PARTICLE SWARM OPTIMIZATION SUMMARIES')
pso.optimization_iteration()