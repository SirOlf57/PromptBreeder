import logging
import random
import re
from typing import List

from colorama import Fore
from sentence_transformers import SentenceTransformer, util
from rich import print
import time

from ollama_client import OllamaClient
from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')
model = SentenceTransformer('bert-base-nli-mean-tokens')


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
            'Q': [],
            'A': '',
            'EA': '',
            'fitness': 0,
            'history': []
        }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)


def init_run(population: Population, num_evals: int, client: OllamaClient):
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
        result = client.prompt(p)
        results.append(result)

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(results):
        population.units[i].P = item

    _evaluate_fitness(population, num_evals, client)

    return population


def run_for_n(n: int, population: Population, num_evals: int, client: OllamaClient):
    """ Runs the genetic algorithm for n generations.
    """
    p = population
    for i in range(n):
        print(f"================== Population {i} ================== ")
        mutate(p, client)
        print("done mutation")
        _evaluate_fitness(p, num_evals, client)
        print("done evaluation")

    return p


def extract_numeric_answers(text):
    """Extract the numeric answer from a given text."""
    numbers = re.findall(r'\d+', text)
    return numbers if numbers else None

def is_correct_answer(llm_response, correct_answer):
    """Check if the correct answer is in the LLM's response."""
    llm_answers = extract_numeric_answers(llm_response)
    return str(correct_answer) in llm_answers

def _evaluate_fitness(population: Population, num_evals: int, client: OllamaClient) -> Population:
    """Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values."""
    print(Fore.CYAN + "Starting fitness evaluation...")
    start_time = time.time()

    # batch = gsm8k_examples[:num_evals]
    batch = random.sample(gsm8k_examples, num_evals)
    elite_fitness = -1
    examples = []
    for unit in population.units:
        unit.fitness = 0
        unit.Q.append([example['question'] for example in batch])
        examples.append([unit.P + ' \n' + example['question'] for example in batch])

    results = []
    for example_batch in examples:
        batch_results = []
        for example in example_batch:
            try:
                result = client.prompt(example, temperature=0)
                batch_results.append(result)
            except Exception as exc:
                print(Fore.RED + f"Exception: {exc}")
                batch_results.append(None)
        results.append(batch_results)

    for unit_index, fitness_results in enumerate(results):
        if fitness_results is None:
            continue
        for index, llm_answer in enumerate(fitness_results):
            if llm_answer is None:
                continue

            print(Fore.MAGENTA + f"Generated result: {llm_answer}")
            print(Fore.YELLOW + f"Expected answer: {batch[index]['answer']}")

            answer = batch[index]['answer']
            extracted_answer = gsm.gsm_extract_answer(answer)

            population.units[unit_index].A = str(llm_answer)
            population.units[unit_index].EA = str(answer)
            if is_correct_answer(llm_answer, extracted_answer):
                population.units[unit_index].fitness += (1 / num_evals)

            # Calculate BERT-based similarity as secondary validation
            embeddings1 = model.encode(extracted_answer, convert_to_tensor=True)
            embeddings2 = model.encode(str(llm_answer), convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

            similarity_score = cosine_scores.item()
            print(Fore.GREEN + f"Similarity score: {similarity_score}")

            if similarity_score > 0.1:
                population.units[unit_index].fitness += (
                            0.5 / num_evals)  # Assign partial credit for semantic similarity

            if population.units[unit_index].fitness > elite_fitness:
                current_elite = population.units[unit_index].model_copy()
                elite_fitness = population.units[unit_index].fitness

    population.elites.append(current_elite)
    end_time = time.time()
    print(Fore.CYAN + f"Done fitness evaluation. {end_time - start_time}s")

    return population
