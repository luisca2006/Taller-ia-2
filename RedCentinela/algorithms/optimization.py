import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    coverage, redundancy, exposure = problem.score_components(configuration)
    return coverage - redundancy - exposure
    
def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    actual = initial_configuration
    evals = 0

    score_actual = configuration_score(problem, actual)
    evals += 1

    historia: list[Configuration] = [actual]
    score_historia: list[float] = [score_actual]

    iteraciones = 0
    while iteraciones < max_iterations:
        m_vecino = None
        m_score = score_actual

        for vecino in problem.neighbors(actual):
            score_vecino = configuration_score(problem, vecino)
            evals += 1

            if score_vecino > m_score:
                m_score = score_vecino
                m_vecino = vecino  

        iteraciones += 1

        if m_vecino is None:
            break
        actual = m_vecino
        score_actual = m_score
        historia.append(actual)
        score_historia.append(score_actual)

    return OptimizationResult(best_configuration=actual, best_score=score_actual, evaluations=evals, iterations=iteraciones, history=historia, score_history=score_historia)

def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    # TODO: Add your code here
    temperture = initial_temperature * (cooling_rate ** iteration)
    return temperture
    
    


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9

    temperature = initial_temperature
    iteration = 0

    actual = initial_configuration
    best = initial_configuration

    current_score = configuration_score(problem, actual)
    best_score = current_score

    while temperature > minimum_temperature and iteration < max_iterations:

        temperature = cooling_schedule(
            initial_temperature,
            cooling_rate,
            iteration
        )

        candidato = rng.choice(problem.neighbors(actual))

        candidate_score = configuration_score(problem, candidato)

        delta = candidate_score - current_score

        if delta > 0:
            actual = candidato
            current_score = candidate_score

        else:
            if rng.random() < math.exp(delta / temperature):
                actual = candidato
                current_score = candidate_score

        if current_score > best_score:
            best = actual
            best_score = current_score

        iteration += 1

    return OptimizationResult(
        best_configuration=best,
        best_score=best_score,
    )



def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    
    cut = rng.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return tuple(child1), tuple(child2)


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    active = []
    inactive = []
    i = 0
    if rng.random() < mutation_probability:
        for i in range(len(individual)):
            if individual[i] == 1:
                active.append(i)
            elif individual[i] == 0:
                inactive.append(i)
        if not active or not inactive:
            return individual
        
        i1 = rng.choice(active)
        i0 = rng.choice(inactive)
        mutated = list(individual)
        mutated[i1] = 0
        mutated[i0] = 1
        return tuple(mutated)
        
        


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    # TODO: Add your code here
    raise NotImplementedError("Punto 3: implemente genetic_algorithm")
