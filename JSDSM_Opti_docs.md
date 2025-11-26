# DSM Optimization Methods Documentation

## Table of Contents
1. [Steward's Path Searching](#stewards-path-searching)
2. [Genetic Algorithm](#genetic-algorithm)
3. [Simulated Annealing](#simulated-annealing)
4. [Ant Colony Optimization](#ant-colony-optimization)
5. [Multi-Objective Optimization (NSGA-II)](#multi-objective-optimization-nsga-ii)
6. [Matrix Bandwidth Minimization](#matrix-bandwidth-minimization)
7. [Comparing Methods](#comparing-methods)

---

## Steward's Path Searching

### Overview
Steward's method is a systematic approach that identifies and breaks dependency cycles by partitioning elements into levels. It's fast and deterministic, making it ideal for hierarchical systems.

### When to Use
- **Best for**: Systems with clear hierarchical structures
- **Advantages**: Fast execution, deterministic results, handles circular dependencies well
- **Disadvantages**: May not find globally optimal solutions for complex systems

### Usage

#### Basic Usage
```javascript
// Simple optimization with default settings
let optimizedModel = optimizeDSMWithStewards(StartingDm);
generateDSM(optimizedModel, document.getElementById('container'));
```

#### How It Works
1. **Dependency Analysis**: Builds adjacency and reachability matrices
2. **Cycle Detection**: Identifies strongly connected components (circular dependencies)
3. **Level Partitioning**: Groups elements into dependency levels
4. **Cycle Breaking**: Breaks cycles at points of minimum feedback

#### Expected Results
```javascript
// Console output example:
// Steward's Optimization Results:
// Original feedback - Count: 4, Strength: 8
// Optimized feedback - Count: 1, Strength: 2
// Improvement - Count: 3, Strength: 6
```

#### Use Cases
- Software architecture with clear module hierarchies
- Manufacturing processes with sequential dependencies
- Project task scheduling
- Organizational structure optimization

---

## Genetic Algorithm

### Overview
Mimics natural evolution to find optimal solutions through selection, crossover, and mutation. Excellent for complex optimization landscapes with multiple constraints.

### When to Use
- **Best for**: Complex systems with multiple constraints and large search spaces
- **Advantages**: Handles complex constraints, can escape local optima, highly customizable
- **Disadvantages**: Computationally intensive, requires parameter tuning

### Usage

#### Basic Usage
```javascript
let optimizedModel = optimizeDSMWithGeneticAlgorithm(StartingDm);
```

#### Advanced Configuration
```javascript
let optimizedModel = optimizeDSMWithGeneticAlgorithm(StartingDm, {
    populationSize: 100,    // Larger population = better exploration
    generations: 200,       // More generations = better convergence
    mutationRate: 0.15,     // Higher rate = more exploration
    crossoverRate: 0.85,    // Higher rate = more exploitation
    elitismRate: 0.1        // Percentage of best solutions to keep
});
```

### Parameters Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `populationSize` | 50 | 20-200 | Larger = better exploration, slower |
| `generations` | 100 | 50-500 | More = better results, slower |
| `mutationRate` | 0.1 | 0.05-0.3 | Higher = more exploration |
| `crossoverRate` | 0.8 | 0.6-0.9 | Higher = more information sharing |
| `elitismRate` | 0.1 | 0.05-0.2 | Higher = faster convergence |

### Tuning Guidelines

#### For Quick Results (Small DSMs < 10 elements)
```javascript
{
    populationSize: 30,
    generations: 50,
    mutationRate: 0.2,
    crossoverRate: 0.7
}
```

#### For High Quality (Large DSMs > 20 elements)
```javascript
{
    populationSize: 100,
    generations: 300,
    mutationRate: 0.1,
    crossoverRate: 0.9,
    elitismRate: 0.15
}
```

#### For Balanced Performance
```javascript
{
    populationSize: 50,
    generations: 150,
    mutationRate: 0.12,
    crossoverRate: 0.8
}
```

---

## Simulated Annealing

### Overview
Probabilistic optimization that gradually reduces randomness, allowing escape from local optima early while converging to good solutions later.

### When to Use
- **Best for**: Medium-complexity problems where you want to avoid local optima
- **Advantages**: Simple to implement, good balance of exploration/exploitation, single solution focus
- **Disadvantages**: Requires careful temperature schedule tuning

### Usage

#### Basic Usage
```javascript
let optimizedModel = optimizeDSMWithSimulatedAnnealing(StartingDm);
```

#### Advanced Configuration
```javascript
let optimizedModel = optimizeDSMWithSimulatedAnnealing(StartingDm, {
    initialTemp: 2000,      // Higher = more initial exploration
    coolingRate: 0.95,      // Slower cooling = more thorough search
    minTemp: 0.1,           // Lower = more final refinement
    maxIterations: 2000     // More iterations = better results
});
```

### Parameters Guide

| Parameter | Default | Typical Range | Effect |
|-----------|---------|---------------|--------|
| `initialTemp` | 1000 | 500-5000 | Higher = accepts more bad moves initially |
| `coolingRate` | 0.95 | 0.9-0.99 | Lower = faster cooling, quicker convergence |
| `minTemp` | 1 | 0.01-10 | Lower = more final optimization |
| `maxIterations` | 1000 | 500-5000 | More = better results, slower |

### Temperature Schedules

#### Fast Cooling (Quick Results)
```javascript
{
    initialTemp: 500,
    coolingRate: 0.9,
    minTemp: 1,
    maxIterations: 500
}
```

#### Slow Cooling (High Quality)
```javascript
{
    initialTemp: 2000,
    coolingRate: 0.98,
    minTemp: 0.1,
    maxIterations: 3000
}
```

#### Adaptive Cooling
```javascript
{
    initialTemp: 1000,
    coolingRate: 0.95,
    minTemp: 0.5,
    maxIterations: 1500
}
```

### Monitoring Progress
```javascript
// Add logging to see optimization progress
function optimizeDSMWithSimulatedAnnealingVerbose(dsmModel, options = {}) {
    // ... (same as original function)
    
    if (iteration % 100 === 0) {
        console.log(`Iteration ${iteration}: Temp=${temperature.toFixed(2)}, Cost=${currentCost}`);
    }
    
    // ... (rest of function)
}
```

---

## Ant Colony Optimization

### Overview
Inspired by ant foraging behavior, uses pheromone trails to find good paths through the solution space. Excellent for dependency-heavy systems.

### When to Use
- **Best for**: Systems with complex interdependencies and path-finding aspects
- **Advantages**: Good for dependency networks, naturally handles constraints, adaptive
- **Disadvantages**: Many parameters to tune, can be slow to converge

### Usage

#### Basic Usage
```javascript
let optimizedModel = optimizeDSMWithAntColony(StartingDm);
```

#### Advanced Configuration
```javascript
let optimizedModel = optimizeDSMWithAntColony(StartingDm, {
    numAnts: 30,            // More ants = better exploration
    iterations: 150,        // More iterations = better convergence
    alpha: 1.5,             // Pheromone importance (1-3)
    beta: 2.5,              // Heuristic importance (1-5)
    evaporationRate: 0.4,   // Pheromone decay (0.1-0.9)
    pheromoneDeposit: 150   // Reward strength (50-200)
});
```

### Parameters Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `numAnts` | 20 | 10-50 | More ants = better exploration |
| `iterations` | 100 | 50-300 | More = better convergence |
| `alpha` | 1 | 0.5-3 | Higher = more pheromone influence |
| `beta` | 2 | 1-5 | Higher = more heuristic influence |
| `evaporationRate` | 0.5 | 0.1-0.9 | Higher = faster pheromone decay |
| `pheromoneDeposit` | 100 | 50-300 | Higher = stronger reinforcement |

### Configuration Strategies

#### Exploration-Focused (Complex Dependencies)
```javascript
{
    numAnts: 40,
    iterations: 200,
    alpha: 0.8,      // Lower pheromone influence
    beta: 3,         // Higher heuristic influence
    evaporationRate: 0.6,
    pheromoneDeposit: 80
}
```

#### Exploitation-Focused (Refinement)
```javascript
{
    numAnts: 25,
    iterations: 150,
    alpha: 2,        // Higher pheromone influence
    beta: 1.5,       // Lower heuristic influence
    evaporationRate: 0.3,
    pheromoneDeposit: 200
}
```

#### Balanced Approach
```javascript
{
    numAnts: 30,
    iterations: 120,
    alpha: 1.2,
    beta: 2.2,
    evaporationRate: 0.5,
    pheromoneDeposit: 120
}
```

---

## Multi-Objective Optimization (NSGA-II)

### Overview
Optimizes multiple conflicting objectives simultaneously (e.g., minimize feedback while maximizing cohesion). Returns a set of trade-off solutions called the Pareto front.

### When to Use
- **Best for**: When you have multiple, conflicting optimization goals
- **Advantages**: Finds multiple trade-off solutions, no need to pre-weight objectives
- **Disadvantages**: Most computationally expensive, complex result interpretation

### Available Objectives
- **`feedback`**: Minimizes dependencies that go "backwards" in order
- **`coupling`**: Minimizes overall dependency strength
- **`cohesion`**: Minimizes distance between dependent elements
- **`bandwidth`**: Minimizes maximum distance between dependencies

### Usage

#### Basic Multi-Objective
```javascript
let optimizedModel = optimizeDSMMultiObjective(StartingDm, ['feedback', 'coupling']);
```

#### Full Multi-Objective Analysis
```javascript
let optimizedModel = optimizeDSMMultiObjective(StartingDm, 
    ['feedback', 'coupling', 'cohesion', 'bandwidth']
);
```

#### Advanced Configuration
```javascript
// Note: This requires modifying the function to accept options
let optimizedModel = optimizeDSMMultiObjective(StartingDm, 
    ['feedback', 'coupling'], 
    {
        populationSize: 100,    // Larger for better Pareto front
        generations: 200,       // More for better convergence
        mutationRate: 0.1,
        crossoverRate: 0.9
    }
);
```

### Objective Combinations

#### Software Architecture Focus
```javascript
['feedback', 'coupling', 'cohesion']
// Minimizes circular dependencies, reduces coupling, keeps related modules close
```

#### Manufacturing Process Focus
```javascript
['feedback', 'bandwidth']
// Minimizes rework loops, keeps process steps close together
```

#### Project Management Focus
```javascript
['feedback', 'coupling']
// Minimizes task dependencies that create scheduling conflicts
```

### Interpreting Results
The algorithm logs the Pareto front solutions:
```
Pareto Front Solutions:
Solution 1: { feedback: 2, coupling: 1.5, cohesion: 12 }
Solution 2: { feedback: 3, coupling: 1.2, cohesion: 10 }
Solution 3: { feedback: 1, coupling: 2.1, cohesion: 15 }
```

**How to Choose:**
- **Solution 1**: Balanced approach
- **Solution 2**: Lower coupling priority
- **Solution 3**: Feedback minimization priority

---

## Matrix Bandwidth Minimization

### Overview
Focuses on keeping dependencies close to the diagonal of the DSM matrix, reducing the "bandwidth" or spread of dependencies.

### When to Use
- **Best for**: When you want dependencies to be close together in the ordering
- **Advantages**: Fast execution, good for visualization, reduces matrix sparsity
- **Disadvantages**: May not minimize feedback optimally

### Usage

#### Basic Usage
```javascript
let optimizedModel = optimizeDSMBandwidth(StartingDm);
```

### Algorithm Details
Uses the **Cuthill-McKee algorithm**:
1. Starts with the element having the fewest dependencies
2. Processes elements in breadth-first order
3. Orders neighbors by their degree (number of dependencies)

### Best Applications
- **Circuit design**: Keeps related components close
- **Software modules**: Groups tightly coupled modules
- **Manufacturing**: Minimizes material flow distances
- **Data structures**: Improves cache locality

### Visual Benefits
Before bandwidth minimization:
```
X . . X . . X
. X . . X . .
X . X . . X .
. . . X X . X
X . . . X X .
. X . X . X .
. . X . . . X
```

After bandwidth minimization:
```
X X . . . . .
X X X . . . .
. X X X . . .
. . X X X . .
. . . X X X .
. . . . X X X
. . . . . X X
```

---

## Comparing Methods

### Performance Comparison Function
```javascript
function compareOptimizationMethods(dsmModel, includeAll = false) {
    console.log("=== DSM Optimization Method Comparison ===");
    
    let methods = [
        { 
            name: "Steward's Path Searching", 
            func: optimizeDSMWithStewards,
            options: {}
        },
        { 
            name: "Genetic Algorithm", 
            func: optimizeDSMWithGeneticAlgorithm,
            options: { generations: 50 } // Reduced for comparison
        },
        { 
            name: "Simulated Annealing", 
            func: optimizeDSMWithSimulatedAnnealing,
            options: { maxIterations: 500 }
        },
        { 
            name: "Bandwidth Minimization", 
            func: optimizeDSMBandwidth,
            options: {}
        }
    ];
    
    if (includeAll) {
        methods.push(
            { 
                name: "Ant Colony", 
                func: optimizeDSMWithAntColony,
                options: { iterations: 50 }
            },
            { 
                name: "Multi-Objective", 
                func: (model) => optimizeDSMMultiObjective(model, ['feedback', 'coupling']),
                options: {}
            }
        );
    }
    
    let results = methods.map(method => {
        console.log(`Running ${method.name}...`);
        let startTime = performance.now();
        
        let optimized = method.func(dsmModel, method.options);
        
        let endTime = performance.now();
        let feedback = calculateFeedbackCost(optimized.rootElement.subElements);
        let bandwidth = calculateBandwidthMetric(optimized.rootElement.subElements);
        
        return {
            method: method.name,
            feedback: feedback,
            bandwidth: bandwidth,
            time: Math.round(endTime - startTime),
            model: optimized
        };
    });
    
    // Sort by feedback (primary) then by time (secondary)
    results.sort((a, b) => {
        if (a.feedback !== b.feedback) {
            return a.feedback - b.feedback;
        }
        return a.time - b.time;
    });
    
    console.table(results.map(r => ({
        Method: r.method,
        Feedback: r.feedback,
        Bandwidth: r.bandwidth,
        "Time (ms)": r.time,
        "Rank": results.indexOf(r) + 1
    })));
    
    console.log(`\n🏆 Best Method: ${results[0].method}`);
    console.log(`📊 Best Feedback: ${results[0].feedback}`);
    console.log(`⚡ Fastest: ${results.reduce((min, r) => r.time < min.time ? r : min).method}`);
    
    return {
        bestOverall: results[0],
        fastest: results.reduce((min, r) => r.time < min.time ? r : min),
        allResults: results
    };
}

// Helper functions
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

function calculateBandwidthMetric(ordering) {
    let maxBandwidth = 0;
    ordering.forEach((element, i) => {
        element.dependencies.forEach(dep => {
            let depIndex = ordering.findIndex(e => e.identifier === dep[0]);
            maxBandwidth = Math.max(maxBandwidth, Math.abs(i - depIndex));
        });
    });
    return maxBandwidth;
}
```

### Usage Example
```javascript
// Quick comparison (recommended for initial testing)
let comparison = compareOptimizationMethods(StartingDm);

// Full comparison (more comprehensive but slower)
let fullComparison = compareOptimizationMethods(StartingDm, true);

// Use the best result
generateDSM(comparison.bestOverall.model, document.getElementById('optimized-container'));

// Or use the fastest if time is critical
generateDSM(comparison.fastest.model, document.getElementById('fast-container'));
```

### Method Selection Guide

| DSM Size | Time Critical | Quality Critical | Recommended Method |
|----------|---------------|------------------|-------------------|
| < 10 elements | Yes | No | Steward's or Bandwidth |
| < 10 elements | No | Yes | Genetic Algorithm |
| 10-20 elements | Yes | No | Simulated Annealing |
| 10-20 elements | No | Yes | Multi-Objective |
| > 20 elements | Yes | No | Steward's |
| > 20 elements | No | Yes | Genetic Algorithm (high params) |
| Any size | - | Multiple goals | Multi-Objective |
| Complex dependencies | - | - | Ant Colony |

### Integration with Your Existing Code

```javascript
// Replace your existing generateDSM call:
// generateDSM(StartingDm, parentElement);

// With optimized version:
function generateOptimizedDSM(dsmModel, parentElement, method = 'auto') {
    let optimizedModel;
    
    switch(method) {
        case 'stewards':
            optimizedModel = optimizeDSMWithStewards(dsmModel);
            break;
        case 'genetic':
            optimizedModel = optimizeDSMWithGeneticAlgorithm(dsmModel);
            break;
        case 'annealing':
            optimizedModel = optimizeDSMWithSimulatedAnnealing(dsmModel);
            break;
        case 'ant':
            optimizedModel = optimizeDSMWithAntColony(dsmModel);
            break;
        case 'bandwidth':
            optimizedModel = optimizeDSMBandwidth(dsmModel);
            break;
        case 'multi':
            optimizedModel = optimizeDSMMultiObjective(dsmModel);
            break;
        case 'auto':
        default:
            let elementCount = dsmModel.rootElement.subElements.length;
            if (elementCount <= 10) {
                optimizedModel = optimizeDSMWithStewards(dsmModel);
            } else if (elementCount <= 20) {
                optimizedModel = optimizeDSMWithSimulatedAnnealing(dsmModel);
            } else {
                optimizedModel = optimizeDSMWithGeneticAlgorithm(dsmModel, { generations: 50 });
            }
            break;
    }
    
    generateDSM(optimizedModel, parentElement);
    return optimizedModel;
}

// Usage:
generateOptimizedDSM(StartingDm, document.getElementById('container'), 'genetic');
```

This documentation provides comprehensive guidance for using each optimization method effectively with your DSM system.