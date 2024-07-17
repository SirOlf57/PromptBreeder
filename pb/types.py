import pydantic
from typing import List

class EvolutionUnit(pydantic.BaseModel):
    """ A individual unit of the overall population.

    Attributes:
        'T': the thinking_style.

        'M': the mutation_prompt.

        'P': the task_prompt.

        'Q': the question prompted the llm.

        'A': the answer from the llm.

        'EA': the expected answer from the dataset.

        'fitness': the estimated performance of the unit.
        
        'history': historical prompts for analysis. 
    """
    T: str
    M: str
    P: str
    Q: list
    A: list
    EA: list
    fitness: float
    history: List[str]

class Population(pydantic.BaseModel):
    """ Population model that holds the age of the population, its size, and a list of individuals.
    
    Attributes:
        'size' (int): the size of the population.

        'age' (int): the age of the population.
        
        'units' (List[EvolutionUnit]): the individuals of a population.
    """
    size: int
    age: int
    problem_description: str
    elites: List[EvolutionUnit]
    units: List[EvolutionUnit]