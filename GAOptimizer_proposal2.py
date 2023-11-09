import numpy as np
import logging

from CombinedParametersForecasting import CombinedParametersForecasting
from ParametersForecasting import ParametersForecasting
from helpingclasses import pygad_modified


# Define constant values
General_Commment= "Major4_proposal2"
DESIRED_OUTPUT = 0
NUM_GENERATIONS = 20
NUM_PARENTS_MATING = 3
SOL_PER_POP = 20
GENES_RANGE_DICT = {"Layer1": [0, 0, 32, 64, 256, 512],
                    "Layer2": [0, 0, 8, 32, 64, 256, 512],
                    "Layer3": [0, 0, 8, 32, 64, 256, 512],
                    "Layer4": [0, 0, 8, 32, 64, 256, 512],
                    "Layer5": [0, 0, 8, 32, 64, 256, 512],
                    "Layer6": [0, 0, 8, 32, 64, 256, 512],
                    "n_batch": [32, 128, 256]}

NUM_GENES = len(GENES_RANGE_DICT)
PARENT_SELECTION_TYPE = "sss"
KEEP_PARENTS = -1
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random_dict"
MUTATION_PERCENT_GENES = 20


def predict(solution, solution_idx=-1):
    """
    Runs a prediction using the given solution.

    Args:
        solution (array-like): The solution to use for the prediction.
        solution_idx (int, optional): The index of the solution. Defaults to -1.

    Returns:
        float: The predicted output.
    """
    s = f"output:{output_index} generation:{ga_instance.generations_completed} index:{solution_idx} count:{ga_instance.count}"
    logging.info("****** %s", s)
    ANN_arch = [int(solution[0]), int(solution[1]), int(solution[2]), int(solution[3]), int(solution[4]), int(solution[5])]
    Forecasting = ParametersForecasting(
        ANN_arch=ANN_arch,
        n_batch=int(solution[6]),
        comment=General_Commment+": " + s,
        model_train_verbose=0,
        n_epochs=500,
        earlystop=True,
        optimizer='adam',
        save_to_database=True,
        outputIndex=output_index,
        ActivationFunctions='tanh',
        checkdatabase=True  
    )
    return Forecasting.start_experiment()

def fitness_func(solution, solution_idx):
    """
    Calculates the fitness value of the given solution.

    Args:
        solution (array-like): The solution to calculate the fitness value for.
        solution_idx (int): The index of the solution.

    Returns:
        float: The fitness value of the solution.
    """
    # logging.debug(solution)
    output = predict(solution, solution_idx)
    # logging.debug("output: %s", output)
    fitness = 1.0 / np.abs(output - DESIRED_OUTPUT)
    # logging.debug("fitness: %s", fitness)
    return fitness

def callback_generation(ga_instance):
    global last_fitness
    # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.


output_index = -1 

for out in [3]:
    output_index=out    
    print('****************************')
    print("output_index = "+ str(output_index))
    print('****************************')

    ga_instance = pygad_modified.GA(num_generations=NUM_GENERATIONS,
                                    num_parents_mating=NUM_PARENTS_MATING,
                                    fitness_func=fitness_func,
                                    sol_per_pop=SOL_PER_POP,
                                    num_genes=NUM_GENES,
                                    # init_range_low=init_range_low,
                                    # init_range_high=init_range_high,
                                    parent_selection_type=PARENT_SELECTION_TYPE,
                                    keep_parents=KEEP_PARENTS,
                                    crossover_type=CROSSOVER_TYPE,
                                    mutation_type=MUTATION_TYPE,
                                    mutation_percent_genes=MUTATION_PERCENT_GENES,
                                    callback_generation=callback_generation,
                                    genes_range_dict=GENES_RANGE_DICT)

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
    # ga_instance.plot_result()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    prediction = predict(solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    # Saving the GA instance.
    filename = 'genetic4'  # The filename to which the instance is saved. The name is without extension.
    ga_instance.save(filename=filename)

    # Loading the saved GA instance.
    # loaded_ga_instance = pygad_modified.load(filename=filename)
    # loaded_ga_instance.plot_result()
