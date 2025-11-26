function optimizeDSMWithStewards(dsmModel) {
    // Create a deep copy to avoid modifying the original
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    let subElements = optimizedModel.rootElement.subElements;
    
    // Build adjacency matrix for dependency analysis
    function buildAdjacencyMatrix(elements) {
        let matrix = {};
        let elementMap = {};
        
        // Create element mapping
        elements.forEach((element, index) => {
            elementMap[element.identifier] = index;
            matrix[element.identifier] = {};
        });
        
        // Initialize matrix with zeros
        elements.forEach(element => {
            elements.forEach(otherElement => {
                matrix[element.identifier][otherElement.identifier] = 0;
            });
        });
        
        // Fill in dependencies
        elements.forEach(element => {
            element.dependencies.forEach(dep => {
                let [depId, strength] = dep;
                if (matrix[element.identifier] && matrix[element.identifier][depId] !== undefined) {
                    matrix[element.identifier][depId] = strength;
                }
            });
        });
        
        return { matrix, elementMap };
    }
    
    // Calculate reachability matrix w/ Floyd-Warshall
    function calculateReachabilityMatrix(adjMatrix, elements) {
        let reachMatrix = {};
        
        // Initialize reachability matrix
        elements.forEach(element => {
            reachMatrix[element.identifier] = {};
            elements.forEach(otherElement => {
                if (element.identifier === otherElement.identifier) {
                    reachMatrix[element.identifier][otherElement.identifier] = 1;
                } else {
                    reachMatrix[element.identifier][otherElement.identifier] = 
                        adjMatrix[element.identifier][otherElement.identifier] > 0 ? 1 : 0;
                }
            });
        });
        
        // Floyd-Warshall algorithm
        elements.forEach(k => {
            elements.forEach(i => {
                elements.forEach(j => {
                    if (reachMatrix[i.identifier][k.identifier] && 
                        reachMatrix[k.identifier][j.identifier]) {
                        reachMatrix[i.identifier][j.identifier] = 1;
                    }
                });
            });
        });
        
        return reachMatrix;
    }
    
    // Find SCCs
    function findStronglyConnectedComponents(reachMatrix, elements) {
        let sccs = [];
        let visited = new Set();
        
        elements.forEach(element => {
            if (!visited.has(element.identifier)) {
                let scc = [];
                elements.forEach(otherElement => {
                    if (!visited.has(otherElement.identifier) &&
                        reachMatrix[element.identifier][otherElement.identifier] &&
                        reachMatrix[otherElement.identifier][element.identifier]) {
                        scc.push(otherElement);
                        visited.add(otherElement.identifier);
                    }
                });
                if (scc.length > 0) {
                    sccs.push(scc);
                }
            }
        });
        
        return sccs;
    }
    
    // Steward's partitioning algorithm
    function stewardsPartitioning(elements) {
        let { matrix: adjMatrix } = buildAdjacencyMatrix(elements);
        let reachMatrix = calculateReachabilityMatrix(adjMatrix, elements);
        let sccs = findStronglyConnectedComponents(reachMatrix, elements);
        
        // Separate elements into levels
        let levels = [];
        let remaining = [...elements];
        let processed = new Set();
        
        while (remaining.length > 0) {
            let currentLevel = [];
            
            // Find elements with no unprocessed dependencies
            remaining.forEach(element => {
                let hasUnprocessedDeps = element.dependencies.some(dep => {
                    let depId = dep[0];
                    return remaining.some(rem => rem.identifier === depId) && 
                           !processed.has(depId);
                });
                
                if (!hasUnprocessedDeps) {
                    currentLevel.push(element);
                }
            });
            
            // If no elements found (circular dependencies), break the cycle
            if (currentLevel.length === 0) {
                // Find SCC with minimum feedback and break it
                let minFeedback = Infinity;
                let elementToBreak = null;
                
                remaining.forEach(element => {
                    let feedbackCount = 0;
                    element.dependencies.forEach(dep => {
                        let depElement = remaining.find(r => r.identifier === dep[0]);
                        if (depElement && reachMatrix[dep[0]][element.identifier]) {
                            feedbackCount += dep[1];
                        }
                    });
                    
                    if (feedbackCount < minFeedback) {
                        minFeedback = feedbackCount;
                        elementToBreak = element;
                    }
                });
                
                if (elementToBreak) {
                    currentLevel.push(elementToBreak);
                }
            }
            
            currentLevel.forEach(element => {
                processed.add(element.identifier);
            });
            
            remaining = remaining.filter(element => !processed.has(element.identifier));
            levels.push(currentLevel);
        }
        
        return levels;
    }
    
    // Apply Steward's algorithm
    let levels = stewardsPartitioning(subElements);
    
    // Flatten levels and reassign orders
    let optimizedElements = [];
    let newOrder = 1;
    
    levels.forEach(level => {
        level.forEach(element => {
            element.order = newOrder++;
            optimizedElements.push(element);
        });
    });
    
    optimizedModel.rootElement.subElements = optimizedElements;
    
    // Calculate optimization metrics
    function calculateFeedback(elements) {
        let feedbackCount = 0;
        let feedbackStrength = 0;
        
        elements.forEach(element => {
            element.dependencies.forEach(dep => {
                let depElement = elements.find(e => e.identifier === dep[0]);
                if (depElement && depElement.order > element.order) {
                    feedbackCount++;
                    feedbackStrength += dep[1];
                }
            });
        });
        
        return { count: feedbackCount, strength: feedbackStrength };
    }
    
    let originalFeedback = calculateFeedback(subElements);
    let optimizedFeedback = calculateFeedback(optimizedElements);
    
    console.log(`Steward's Optimization Results:`);
    console.log(`Original feedback - Count: ${originalFeedback.count}, Strength: ${originalFeedback.strength}`);
    console.log(`Optimized feedback - Count: ${optimizedFeedback.count}, Strength: ${optimizedFeedback.strength}`);
    console.log(`Improvement - Count: ${originalFeedback.count - optimizedFeedback.count}, Strength: ${originalFeedback.strength - optimizedFeedback.strength}`);
    
    return optimizedModel;
}

function optimizeDSMWithGeneticAlgorithm(dsmModel, options = {}) {
    const {
        populationSize = 50,
        generations = 100,
        mutationRate = 0.1,
        crossoverRate = 0.8,
        elitismRate = 0.1
    } = options;
    
    let subElements = [...dsmModel.rootElement.subElements];
    
    // Fitness function - minimize feedback
    function calculateFitness(ordering) {
        let feedback = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                if (depIndex > i) { // Feedback dependency
                    feedback += dep[1];
                }
            });
        });
        return 1 / (1 + feedback); // Higher fitness = lower feedback
    }
    
    // Initialize population
    function createRandomOrdering() {
        let shuffled = [...subElements];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
    
    let population = Array.from({length: populationSize}, () => createRandomOrdering());
    
    // Evolution loop
    for (let gen = 0; gen < generations; gen++) {
        // Calculate fitness for all individuals
        let fitnessScores = population.map(individual => ({
            individual,
            fitness: calculateFitness(individual)
        }));
        
        // Sort by fitness (descending)
        fitnessScores.sort((a, b) => b.fitness - a.fitness);
        
        // Create next generation
        let newPopulation = [];
        
        // Elitism - keep best individuals
        let eliteCount = Math.floor(populationSize * elitismRate);
        for (let i = 0; i < eliteCount; i++) {
            newPopulation.push([...fitnessScores[i].individual]);
        }
        
        // Crossover and mutation
        while (newPopulation.length < populationSize) {
            // Tournament selection
            let parent1 = tournamentSelection(fitnessScores, 3);
            let parent2 = tournamentSelection(fitnessScores, 3);
            
            let offspring = crossover(parent1, parent2);
            if (Math.random() < mutationRate) {
                offspring = mutate(offspring);
            }
            
            newPopulation.push(offspring);
        }
        
        population = newPopulation;
    }
    
    // Return best solution
    let finalFitness = population.map(ind => ({
        individual: ind,
        fitness: calculateFitness(ind)
    }));
    finalFitness.sort((a, b) => b.fitness - a.fitness);
    
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    optimizedModel.rootElement.subElements = finalFitness[0].individual.map((el, i) => ({
        ...el,
        order: i + 1
    }));
    
    return optimizedModel;
}

function tournamentSelection(fitnessScores, tournamentSize) {
    let tournament = [];
    for (let i = 0; i < tournamentSize; i++) {
        tournament.push(fitnessScores[Math.floor(Math.random() * fitnessScores.length)]);
    }
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0].individual;
}

function crossover(parent1, parent2) {
    // Order crossover (OX)
    let start = Math.floor(Math.random() * parent1.length);
    let end = Math.floor(Math.random() * (parent1.length - start)) + start;
    
    let offspring = new Array(parent1.length);
    
    // Copy segment from parent1
    for (let i = start; i <= end; i++) {
        offspring[i] = parent1[i];
    }
    
    // Fill remaining positions with parent2's order
    let parent2Filtered = parent2.filter(item => 
        !offspring.slice(start, end + 1).some(offItem => 
            offItem && offItem.identifier === item.identifier));
    
    let index = 0;
    for (let i = 0; i < offspring.length; i++) {
        if (!offspring[i]) {
            offspring[i] = parent2Filtered[index++];
        }
    }
    
    return offspring;
}

function mutate(individual) {
    let mutated = [...individual];
    let i = Math.floor(Math.random() * mutated.length);
    let j = Math.floor(Math.random() * mutated.length);
    [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
    return mutated;
}

function optimizeDSMWithSimulatedAnnealing(dsmModel, options = {}) {
    const {
        initialTemp = 1000,
        coolingRate = 0.95,
        minTemp = 1,
        maxIterations = 1000
    } = options;
    
    let currentSolution = [...dsmModel.rootElement.subElements];
    let currentCost = calculateCost(currentSolution);
    let bestSolution = [...currentSolution];
    let bestCost = currentCost;
    
    let temperature = initialTemp;
    let iteration = 0;
    
    function calculateCost(ordering) {
        let cost = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                if (depIndex > i) {
                    cost += dep[1] * dep[1]; // Square to penalize stronger feedback more
                }
            });
        });
        return cost;
    }
    
    function generateNeighbor(solution) {
        let neighbor = [...solution];
        let i = Math.floor(Math.random() * neighbor.length);
        let j = Math.floor(Math.random() * neighbor.length);
        [neighbor[i], neighbor[j]] = [neighbor[j], neighbor[i]];
        return neighbor;
    }
    
    while (temperature > minTemp && iteration < maxIterations) {
        let neighbor = generateNeighbor(currentSolution);
        let neighborCost = calculateCost(neighbor);
        
        let deltaE = neighborCost - currentCost;
        
        if (deltaE < 0 || Math.random() < Math.exp(-deltaE / temperature)) {
            currentSolution = neighbor;
            currentCost = neighborCost;
            
            if (currentCost < bestCost) {
                bestSolution = [...currentSolution];
                bestCost = currentCost;
            }
        }
        
        temperature *= coolingRate;
        iteration++;
    }
    
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    optimizedModel.rootElement.subElements = bestSolution.map((el, i) => ({
        ...el,
        order: i + 1
    }));
    
    return optimizedModel;
}

function optimizeDSMWithAntColony(dsmModel, options = {}) {
    const {
        numAnts = 20,
        iterations = 100,
        alpha = 1, // Pheromone importance
        beta = 2,  // Heuristic importance
        evaporationRate = 0.5,
        pheromoneDeposit = 100
    } = options;
    
    let elements = dsmModel.rootElement.subElements;
    let n = elements.length;
    
    // Initialize pheromone matrix
    let pheromones = Array(n).fill().map(() => Array(n).fill(1));
    
    // Heuristic information (inverse of dependency strength)
    let heuristics = Array(n).fill().map(() => Array(n).fill(1));
    
    function buildSolution() {
        let solution = [];
        let available = [...elements];
        
        while (available.length > 0) {
            let probabilities = available.map((element, i) => {
                let elementIndex = elements.findIndex(e => e.identifier === element.identifier);
                let pheromone = solution.length > 0 ? 
                    pheromones[elements.findIndex(e => e.identifier === solution[solution.length - 1].identifier)][elementIndex] : 1;
                let heuristic = heuristics[elementIndex][0];
                
                return Math.pow(pheromone, alpha) * Math.pow(heuristic, beta);
            });
            
            let totalProb = probabilities.reduce((sum, p) => sum + p, 0);
            probabilities = probabilities.map(p => p / totalProb);
            
            // Roulette wheel selection
            let random = Math.random();
            let cumulative = 0;
            let selectedIndex = 0;
            
            for (let i = 0; i < probabilities.length; i++) {
                cumulative += probabilities[i];
                if (random <= cumulative) {
                    selectedIndex = i;
                    break;
                }
            }
            
            solution.push(available[selectedIndex]);
            available.splice(selectedIndex, 1);
        }
        
        return solution;
    }
    
    let bestSolution = null;
    let bestCost = Infinity;
    
    for (let iter = 0; iter < iterations; iter++) {
        let solutions = [];
        
        // Generate solutions with ants
        for (let ant = 0; ant < numAnts; ant++) {
            let solution = buildSolution();
            let cost = calculateFeedbackCost(solution);
            solutions.push({ solution, cost });
            
            if (cost < bestCost) {
                bestCost = cost;
                bestSolution = [...solution];
            }
        }
        
        // Update pheromones
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                pheromones[i][j] *= (1 - evaporationRate);
            }
        }
        
        // Add pheromones from solutions
        solutions.forEach(({ solution, cost }) => {
            let deposit = pheromoneDeposit / cost;
            for (let i = 0; i < solution.length - 1; i++) {
                let from = elements.findIndex(e => e.identifier === solution[i].identifier);
                let to = elements.findIndex(e => e.identifier === solution[i + 1].identifier);
                pheromones[from][to] += deposit;
            }
        });
    }
    
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    optimizedModel.rootElement.subElements = bestSolution.map((el, i) => ({
        ...el,
        order: i + 1
    }));
    
    return optimizedModel;
}

function calculateFeedbackCost(ordering) {
    let cost = 0;
    ordering.forEach((element, i) => {
        element.dependencies.forEach(dep => {
            let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
            if (depIndex > i) {
                cost += dep[1];
            }
        });
    });
    return cost;
}

function optimizeDSMMultiObjective(dsmModel, objectives = ['feedback', 'coupling', 'cohesion']) {
    let elements = dsmModel.rootElement.subElements;
    
    // Calculate multiple objectives
    function evaluateObjectives(ordering) {
        let results = {};
        
        if (objectives.includes('feedback')) {
            results.feedback = calculateFeedback(ordering);
        }
        
        if (objectives.includes('coupling')) {
            results.coupling = calculateCoupling(ordering);
        }
        
        if (objectives.includes('cohesion')) {
            results.cohesion = calculateCohesion(ordering);
        }
        
        if (objectives.includes('bandwidth')) {
            results.bandwidth = calculateBandwidth(ordering);
        }
        
        return results;
    }
    
    function calculateFeedback(ordering) {
        let feedback = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                if (depIndex > i) feedback += dep[1];
            });
        });
        return feedback;
    }
    
    function calculateCoupling(ordering) {
        let totalDependencies = 0;
        ordering.forEach(element => {
            totalDependencies += element.dependencies.length;
        });
        return totalDependencies / ordering.length;
    }
    
    function calculateCohesion(ordering) {
        // Measure how close related elements are
        let cohesion = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                cohesion += Math.abs(i - depIndex) * dep[1];
            });
        });
        return cohesion;
    }
    
    function calculateBandwidth(ordering) {
        let maxDistance = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                maxDistance = Math.max(maxDistance, Math.abs(i - depIndex));
            });
        });
        return maxDistance;
    }
    
    // Use NSGA-II for multi-objective optimization
    return nsgaII(elements, evaluateObjectives, objectives);
}

function nsgaII(elements, evaluateObjectives, objectives, options = {}) {
    const {
        populationSize = 50,
        generations = 100,
        mutationRate = 0.1,
        crossoverRate = 0.8
    } = options;
    
    // Initialize population with random orderings
    function createRandomOrdering() {
        let shuffled = [...elements];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
    
    // Create individual with objectives evaluated
    function createIndividual(ordering = null) {
        let individual = {
            ordering: ordering || createRandomOrdering(),
            objectives: {},
            dominationCount: 0,
            dominatedSolutions: [],
            rank: 0,
            crowdingDistance: 0
        };
        
        individual.objectives = evaluateObjectives(individual.ordering);
        return individual;
    }
    
    // Check if solution1 dominates solution2
    function dominates(sol1, sol2) {
        let atLeastOneBetter = false;
        let allBetterOrEqual = true;
        
        for (let obj of objectives) {
            let val1 = sol1.objectives[obj];
            let val2 = sol2.objectives[obj];
            
            // Assuming minimization for all objectives
            if (val1 < val2) {
                atLeastOneBetter = true;
            } else if (val1 > val2) {
                allBetterOrEqual = false;
                break;
            }
        }
        
        return atLeastOneBetter && allBetterOrEqual;
    }
    
    // Fast non-dominated sorting
    function fastNonDominatedSort(population) {
        let fronts = [[]];
        
        // Initialize domination properties
        population.forEach(individual => {
            individual.dominationCount = 0;
            individual.dominatedSolutions = [];
        });
        
        // Calculate domination relationships
        for (let i = 0; i < population.length; i++) {
            for (let j = 0; j < population.length; j++) {
                if (i !== j) {
                    if (dominates(population[i], population[j])) {
                        population[i].dominatedSolutions.push(population[j]);
                    } else if (dominates(population[j], population[i])) {
                        population[i].dominationCount++;
                    }
                }
            }
            
            if (population[i].dominationCount === 0) {
                population[i].rank = 1;
                fronts[0].push(population[i]);
            }
        }
        
        // Build subsequent fronts
        let currentFront = 0;
        while (fronts[currentFront].length > 0) {
            let nextFront = [];
            
            fronts[currentFront].forEach(individual => {
                individual.dominatedSolutions.forEach(dominated => {
                    dominated.dominationCount--;
                    if (dominated.dominationCount === 0) {
                        dominated.rank = currentFront + 2;
                        nextFront.push(dominated);
                    }
                });
            });
            
            currentFront++;
            if (nextFront.length > 0) {
                fronts.push(nextFront);
            } else {
                break;
            }
        }
        
        return fronts;
    }
    
    // Calculate crowding distance for diversity preservation
    function calculateCrowdingDistance(front) {
        if (front.length <= 2) {
            front.forEach(individual => {
                individual.crowdingDistance = Infinity;
            });
            return;
        }
        
        // Initialize crowding distance
        front.forEach(individual => {
            individual.crowdingDistance = 0;
        });
        
        // Calculate distance for each objective
        objectives.forEach(objective => {
            // Sort by objective value
            front.sort((a, b) => a.objectives[objective] - b.objectives[objective]);
            
            // Set boundary points to infinity
            front[0].crowdingDistance = Infinity;
            front[front.length - 1].crowdingDistance = Infinity;
            
            // Calculate range
            let objRange = front[front.length - 1].objectives[objective] - front[0].objectives[objective];
            
            if (objRange === 0) return; // Avoid division by zero
            
            // Calculate crowding distance for intermediate points
            for (let i = 1; i < front.length - 1; i++) {
                let distance = (front[i + 1].objectives[objective] - front[i - 1].objectives[objective]) / objRange;
                front[i].crowdingDistance += distance;
            }
        });
    }
    
    // Tournament selection based on rank and crowding distance
    function tournamentSelection(population, tournamentSize = 2) {
        let tournament = [];
        
        for (let i = 0; i < tournamentSize; i++) {
            tournament.push(population[Math.floor(Math.random() * population.length)]);
        }
        
        // Select based on rank first, then crowding distance
        tournament.sort((a, b) => {
            if (a.rank !== b.rank) {
                return a.rank - b.rank; // Lower rank is better
            }
            return b.crowdingDistance - a.crowdingDistance; // Higher crowding distance is better
        });
        
        return tournament[0];
    }
    
    // Order crossover (OX) for permutation representation
    function orderCrossover(parent1, parent2) {
        let p1Order = parent1.ordering;
        let p2Order = parent2.ordering;
        let length = p1Order.length;
        
        // Select random crossover points
        let start = Math.floor(Math.random() * length);
        let end = Math.floor(Math.random() * (length - start)) + start;
        
        let offspring1 = new Array(length);
        let offspring2 = new Array(length);
        
        // Copy segments
        for (let i = start; i <= end; i++) {
            offspring1[i] = p1Order[i];
            offspring2[i] = p2Order[i];
        }
        
        // Fill remaining positions
        fillOffspring(offspring1, p2Order, start, end);
        fillOffspring(offspring2, p1Order, start, end);
        
        return [
            createIndividual(offspring1),
            createIndividual(offspring2)
        ];
    }
    
    function fillOffspring(offspring, parent, start, end) {
        let parentFiltered = parent.filter(item => 
            !offspring.slice(start, end + 1).some(offItem => 
                offItem && offItem.identifier === item.identifier));
        
        let index = 0;
        for (let i = 0; i < offspring.length; i++) {
            if (!offspring[i]) {
                offspring[i] = parentFiltered[index++];
            }
        }
    }
    
    // Swap mutation
    function mutate(individual) {
        let mutated = [...individual.ordering];
        let i = Math.floor(Math.random() * mutated.length);
        let j = Math.floor(Math.random() * mutated.length);
        [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
        return createIndividual(mutated);
    }
    
    // Environmental selection - select next generation
    function environmentalSelection(population, populationSize) {
        let fronts = fastNonDominatedSort(population);
        let nextGeneration = [];
        let frontIndex = 0;
        
        // Add complete fronts
        while (frontIndex < fronts.length && 
               nextGeneration.length + fronts[frontIndex].length <= populationSize) {
            calculateCrowdingDistance(fronts[frontIndex]);
            nextGeneration.push(...fronts[frontIndex]);
            frontIndex++;
        }
        
        // Add partial front if needed
        if (nextGeneration.length < populationSize && frontIndex < fronts.length) {
            calculateCrowdingDistance(fronts[frontIndex]);
            fronts[frontIndex].sort((a, b) => b.crowdingDistance - a.crowdingDistance);
            
            let remaining = populationSize - nextGeneration.length;
            nextGeneration.push(...fronts[frontIndex].slice(0, remaining));
        }
        
        return nextGeneration;
    }
    
    // Main NSGA-II algorithm
    console.log("Starting NSGA-II optimization...");
    
    // Initialize population
    let population = Array.from({length: populationSize}, () => createIndividual());
    
    // Evolution loop
    for (let generation = 0; generation < generations; generation++) {
        let offspring = [];
        
        // Generate offspring through crossover and mutation
        while (offspring.length < populationSize) {
            let parent1 = tournamentSelection(population);
            let parent2 = tournamentSelection(population);
            
            if (Math.random() < crossoverRate) {
                let children = orderCrossover(parent1, parent2);
                
                children.forEach(child => {
                    if (Math.random() < mutationRate) {
                        child = mutate(child);
                    }
                    offspring.push(child);
                });
            } else {
                // Direct copy with possible mutation
                let child1 = createIndividual([...parent1.ordering]);
                let child2 = createIndividual([...parent2.ordering]);
                
                if (Math.random() < mutationRate) child1 = mutate(child1);
                if (Math.random() < mutationRate) child2 = mutate(child2);
                
                offspring.push(child1, child2);
            }
        }
        
        // Limit offspring to population size
        offspring = offspring.slice(0, populationSize);
        
        // Combine parent and offspring populations
        let combinedPopulation = [...population, ...offspring];
        
        // Environmental selection
        population = environmentalSelection(combinedPopulation, populationSize);
        
        if (generation % 20 === 0) {
            console.log(`Generation ${generation}: Population size = ${population.length}`);
        }
    }
    
    // Get Pareto front (first front after final sorting)
    let finalFronts = fastNonDominatedSort(population);
    let paretoFront = finalFronts[0];
    
    console.log(`NSGA-II completed. Pareto front size: ${paretoFront.length}`);
    
    // Return the solution with best compromise (you can modify this selection)
    // For now, we'll return the one with minimum sum of normalized objectives
    let bestCompromise = paretoFront[0];
    let minScore = Infinity;
    
    // Normalize objectives
    let objRanges = {};
    objectives.forEach(obj => {
        let values = paretoFront.map(sol => sol.objectives[obj]);
        let min = Math.min(...values);
        let max = Math.max(...values);
        objRanges[obj] = { min, max, range: max - min };
    });
    
    paretoFront.forEach(solution => {
        let normalizedSum = 0;
        objectives.forEach(obj => {
            if (objRanges[obj].range > 0) {
                normalizedSum += (solution.objectives[obj] - objRanges[obj].min) / objRanges[obj].range;
            }
        });
        
        if (normalizedSum < minScore) {
            minScore = normalizedSum;
            bestCompromise = solution;
        }
    });
    
    // Log Pareto front solutions
    console.log("Pareto Front Solutions:");
    paretoFront.forEach((solution, index) => {
        console.log(`Solution ${index + 1}:`, solution.objectives);
    });
    
    return bestCompromise.ordering;
}

// Complete Multi-Objective DSM Optimization
function optimizeDSMMultiObjective(dsmModel, objectives = ['feedback', 'coupling', 'cohesion']) {
    let elements = dsmModel.rootElement.subElements;
    
    function evaluateObjectives(ordering) {
        let results = {};
        
        if (objectives.includes('feedback')) {
            results.feedback = calculateFeedback(ordering);
        }
        
        if (objectives.includes('coupling')) {
            results.coupling = calculateCoupling(ordering);
        }
        
        if (objectives.includes('cohesion')) {
            results.cohesion = calculateCohesion(ordering);
        }
        
        if (objectives.includes('bandwidth')) {
            results.bandwidth = calculateBandwidth(ordering);
        }
        
        return results;
    }
    
    function calculateFeedback(ordering) {
        let feedback = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                if (depIndex > i) feedback += dep[1];
            });
        });
        return feedback;
    }
    
    function calculateCoupling(ordering) {
        let totalDependencies = 0;
        ordering.forEach(element => {
            totalDependencies += element.dependencies.reduce((sum, dep) => sum + dep[1], 0);
        });
        return totalDependencies / ordering.length;
    }
    
    function calculateCohesion(ordering) {
        let cohesion = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                cohesion += Math.abs(i - depIndex) * dep[1];
            });
        });
        return cohesion;
    }
    
    function calculateBandwidth(ordering) {
        let maxDistance = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                maxDistance = Math.max(maxDistance, Math.abs(i - depIndex));
            });
        });
        return maxDistance;
    }
    
    // Run NSGA-II
    let optimizedOrdering = nsgaII(elements, evaluateObjectives, objectives, {
        populationSize: 50,
        generations: 100,
        mutationRate: 0.1,
        crossoverRate: 0.8
    });
    
    // Create optimized model
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    optimizedModel.rootElement.subElements = optimizedOrdering.map((el, i) => ({
        ...el,
        order: i + 1
    }));
    
    return optimizedModel;
}

function optimizeDSMBandwidth(dsmModel) {
    let elements = dsmModel.rootElement.subElements;
    
    function calculateBandwidth(ordering) {
        let bandwidth = 0;
        ordering.forEach((element, i) => {
            element.dependencies.forEach(dep => {
                let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
                bandwidth = Math.max(bandwidth, Math.abs(i - depIndex));
            });
        });
        return bandwidth;
    }
    
    // Use Cuthill-McKee algorithm
    function cutHillMcKee(elements) {
        let visited = new Set();
        let result = [];
        let degrees = new Map();
        
        // Calculate degrees
        elements.forEach(element => {
            degrees.set(element.identifier, element.dependencies.length);
        });
        
        // Find peripheral node (minimum degree)
        let startNode = elements.reduce((min, element) => 
            degrees.get(element.identifier) < degrees.get(min.identifier) ? element : min
        );
        
        let queue = [startNode];
        visited.add(startNode.identifier);
        
        while (queue.length > 0) {
            let current = queue.shift();
            result.push(current);
            
            // Add unvisited neighbors sorted by degree
            let neighbors = current.dependencies
                .map(dep => elements.find(e => e.identifier === dep[0]))
                .filter(neighbor => neighbor && !visited.has(neighbor.identifier))
                .sort((a, b) => degrees.get(a.identifier) - degrees.get(b.identifier));
            
            neighbors.forEach(neighbor => {
                if (!visited.has(neighbor.identifier)) {
                    visited.add(neighbor.identifier);
                    queue.push(neighbor);
                }
            });
        }
        
        // Add any remaining unvisited nodes
        elements.forEach(element => {
            if (!visited.has(element.identifier)) {
                result.push(element);
            }
        });
        
        return result;
    }
    
    let optimizedOrdering = cutHillMcKee(elements);
    
    let optimizedModel = JSON.parse(JSON.stringify(dsmModel));
    optimizedModel.rootElement.subElements = optimizedOrdering.map((el, i) => ({
        ...el,
        order: i + 1
    }));
    
    return optimizedModel;
}

// DSM generation function wrapper that applies optimization
function generateOptimizedDSM(dsmModel, parentElement, optimizationOption = "SPS") {
    let modelToUse;
    switch (optimizationOption) {
        case "SPS":
            // code to execute if expression === value1
            break;
        case "GA":
            // code to execute if expression === value2
            break;
        case "ACO":
            // code to execute if expression === value2
            break;
        case "MOB":
            // code to execute if expression === value2
            break;
        case "BW":
            // code to execute if expression === value2
            break;
        case "BW":
            // code to execute if expression === value2
            break;
        case "BW":
            // code to execute if expression === value2
            break;
        default:
            modelToUse = optimizeDSMWithStewards(dsmModel)
            console.log(`Optimization option not recognized. Defaulting to SPS.`)
        break; 
    }
    generateDSM(modelToUse, parentElement);
    return modelToUse;
}