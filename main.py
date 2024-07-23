from ollama_client import OllamaClient
from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import logging
import argparse

from dotenv import load_dotenv
from rich import print

load_dotenv() # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=2, type=int)
parser.add_argument('-ts', '--num_thinking_styles', default=4, type=int)
parser.add_argument('-e', '--num_evals', default=10, type=int)
parser.add_argument('-n', '--simulations', default=10, type=int)
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       
parser.add_argument('-m', '--model', default="llama3:8b")

args = vars(parser.parse_args())


ollama_client = OllamaClient(host="http://localhost:11434", model=args['model'])
total_evaluations = args['num_mutation_prompts']*args['num_thinking_styles']*args['num_evals']

tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set = thinking_styles[:int(args['num_thinking_styles'])]

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, int(args['num_evals']), ollama_client)

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, num_evals=int(args['num_evals']), client=ollama_client)

print("%"*80)
print("done processing! final gen:")
print(p.units)
