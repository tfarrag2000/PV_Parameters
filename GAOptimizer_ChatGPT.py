import numpy as np
from helpingclasses import pygad_modified
from CombinedParametersForecasting import CombinedParametersForecasting

# Define the desired output
desired_output = 44

# Define the input values
x = [4, -2, 3.5, 5, -11, -4.7]

# Define the fitness function
def fitness_func(solution, solution_idx):
    # Convert solution to integer values
    solution_int = [int(i) for i in solution]
    # Create an instance of CombinedParametersForecasting
    forecasting = CombinedParametersForecasting(ANN_arch=solution_int[:6],
                                                 n_batch=solution_int[6],
                                                 n_epochs=1000,
                                                 earlystop=True,
                                                 optimizer='adam',
                                                 ActivationFunctions='tanh')
    # Get the output
    output = forecasting.predict(x)
    # Calculate the fitness
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

# Define the callback function
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Define the GA parameters
num_generations = 20
num_parents_mating = 3
sol_per_pop = 20
num_genes = 7
genes_range_dict = {"Layer1": [0, 0, 32, 64, 256, 512],
                    "Layer2": [0, 0, 8, 32, 64, 256, 512],
                    "Layer3": [0, 0, 8, 32, 64, 256, 512],
                    "Layer4": [0, 0, 8, 32, 64, 256, 512],
                    "Layer5": [0, 0, 8, 32, 64, 256, 512],
                    "Layer6": [0, 0, 8, 32, 64, 256, 512],
                    "n_batch": [32, 128, 256]}
parent_selection_type = "sss"
keep_parents = -1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20

# Create an instance of the GA class
ga_instance = pygad_modified.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_func,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                genes_range_dict=genes_range_dict,
                                parent_selection_type=parent_selection_type,
                                keep_parents=keep_parents,
                                crossover_type=crossover_type,
                                mutation_type=mutation_type,
                                mutation_percent_genes=mutation_percent_genes,
                                callback_generation=callback_generation)

# Run the GA
ga_instance.run()

# Print the best solution found
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution : ", solution)
print("Best solution fitness : ", solution_fitness)
