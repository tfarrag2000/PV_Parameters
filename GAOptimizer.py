import numpy

import ParametersForecasting
import CombinedParametersForecasting
import pygad_modified

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""

desired_output = 0  # Function output.
ii = 0


def Predict(solution, solution_idx=-1):
    ga_instance.count = ga_instance.count + 1
    outputIndex = -1
    s = "output:{} generation:{} index:{} count:{}".format(outputIndex, ga_instance.generations_completed, solution_idx,
                                                           ga_instance.count, ga_instance.generations_completed)
    print("****** " + str(s))

    ANN_arch = [int(solution[0]), int(solution[1]), int(solution[2]), int(solution[3]), int(solution[4]), int(solution[5])]
    Forecasting = CombinedParametersForecasting.CombinedParametersForecasting(#Dropout=float(solution[7]),
                                                              ANN_arch=ANN_arch,
                                                              n_batch=int(solution[6]),
                                                              comment="Genetic Optimizer combined model: " + s,
                                                              model_train_verbose=0,
                                                              n_epochs=1000, earlystop=True, optimizer='adam',
                                                              outputIndex=outputIndex, ActivationFunctions='tanh')
    return Forecasting.start_experiment()


def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    print(solution)
    output = Predict(solution, solution_idx)
    print("output : " + str(output))
    fitness = 1.0 / numpy.abs(output - desired_output)
    print("fitness : " + str(fitness))
    return fitness


fitness_function = fitness_func

num_generations = 20  # Number of generations.
num_parents_mating = 3  # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:

# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 20  # Number of solutions in the population.

# init_range_low = -2
# init_range_high = 5
genes_range_dict = {"Layer1": [0,0, 32, 64, 256, 512],
                    "Layer2": [0,0, 8, 32, 64, 256, 512],
                    "Layer3": [0,0, 8, 32, 64, 256, 512],
                    "Layer4": [0,0, 8, 32, 64, 256, 512],
                    "Layer5": [0,0,8, 32, 64, 256, 512],
                    "Layer6": [0, 0, 8, 32, 64, 256, 512],
                    "n_batch": [32, 128, 256]
                   # "Dropout": [0, 0.2, 0.5, 0.8]
                    # "ActivationFunctions": ['relu', 'tanh']
                    }

num_genes = len(genes_range_dict)

parent_selection_type = "sss"  # Type of parent selection.
keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

# Type of the crossover operator.
crossover_type="single_point"
# Parameters of the mutation operation.
mutation_type = "random_dict"  # Type of the mutation operator.
mutation_percent_genes = 20  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

last_fitness = 0


def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad_modified.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_function,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                # init_range_low=init_range_low,
                                # init_range_high=init_range_high,
                                parent_selection_type=parent_selection_type,
                                keep_parents=keep_parents,
                                crossover_type=crossover_type,
                                mutation_type=mutation_type,
                                mutation_percent_genes=mutation_percent_genes,
                                callback_generation=callback_generation,
                                genes_range_dict=genes_range_dict)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

prediction = Predict(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'genetic4'  # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad_modified.load(filename=filename)
loaded_ga_instance.plot_result()
