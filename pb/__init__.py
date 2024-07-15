import concurrent.futures
import re
import logging
from typing import List

from rich import print
import time

from pb.mutation_operators import mutate, llm
from pb import gsm
from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')


def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set) * len(mutator_set),
        'age': 0,
        'problem_description': problem_description,
        'elites': [],
        'units': [EvolutionUnit(**{
            'T': t,
            'M': m,
            'P': '',
            'fitness': 0,
            'history': []
        }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)


def init_run(population: Population, num_evals: int):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

    prompts = []
    results = []

    for unit in population.units:
        template = f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        prompts.append(template)

    for p in prompts:
        result = llm(p)
        results.append(result)

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(results):
        population.units[i].P = item

    _evaluate_fitness(population, num_evals)

    return population


def run_for_n(n: int, population: Population, num_evals: int):
    """ Runs the genetic algorithm for n generations.
    """
    p = population
    for i in range(n):
        print(f"================== Population {i} ================== ")
        mutate(p)
        print("done mutation")
        _evaluate_fitness(p, num_evals)
        print("done evaluation")

    return p


def _evaluate_fitness(population: Population, num_evals: int) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.

    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    # batch = random.sample(gsm8k_examples, num_evals)
    # instead of random, its better for reproducibility 
    batch = gsm8k_examples[:num_evals]

    elite_fitness = -1
    examples = []
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        examples.append([unit.P + '\n' + example['question'] for example in batch])

    results = []
    for example_batch in examples:
        try:
            data = [llm(example, temperature=0) for example in example_batch]
            results.append(data)
        except Exception as exc:
            print(f"Exception: {exc}")

    # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition
    # the LLM before further input Q.
    for unit_index, fitness_results in enumerate(results):
        for i, x in enumerate(fitness_results):
            valid = re.search(gsm.gsm_extract_answer(batch[i]['answer']), str(x))
            if valid:
                # 0.25 = 1 / 4 examples
                population.units[unit_index].fitness += (1 / num_evals)

            if population.units[unit_index].fitness > elite_fitness:
                # Copy the unit to preserve it against future mutations
                unit = population.units[unit_index]
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness

    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population
