import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load the returns data ===
stocks_df = pd.read_csv("all_stocks_5yr.csv") # Download dataset from https://www.kaggle.com/datasets/camnugent/sandp500
price_data = stocks_df.pivot(index='date', columns='Name', values='close')
price_data = price_data.sort_index().ffill()
returns_all = price_data.pct_change().dropna()

# === User-defined stock selection ===
TOP_N_STOCKS = int(input("Enter number of top stocks to consider: "))
top_stocks = returns_all.mean().sort_values(ascending=False).head(TOP_N_STOCKS).index.tolist()
returns = returns_all[top_stocks]

# === Sharpe Ratio Objective Function ===
def sharpe_ratio(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -portfolio_return / portfolio_std  # Negative for minimization

# === Normalize weights ===
def normalize(weights):
    weights = np.maximum(weights, 0)
    total = np.sum(weights)
    if total == 0:
        return np.ones_like(weights) / len(weights)
    return weights / total

# === Improved Bat Algorithm ===
def bat_algorithm(n=30, max_gen=100):
    dim = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    A = np.random.rand(n)
    r = np.random.rand(n)
    Qmin, Qmax = 0, 5
    freq = np.zeros(n)
    v = np.zeros((n, dim))
    Sol = np.random.dirichlet(np.ones(dim), size=n)
    Fitness = np.array([sharpe_ratio(sol, mean_returns, cov_matrix) for sol in Sol])

    best_idx = np.argmin(Fitness)
    best = Sol[best_idx]
    best_fitness = Fitness[best_idx]

    history = []

    for t in range(max_gen):
        for i in range(n):
            freq[i] = Qmin + (Qmax - Qmin) * np.random.rand()
            v[i] += (Sol[i] - best) * freq[i]
            S = Sol[i] + v[i]
            S = normalize(S)

            if np.random.rand() > r[i]:
                eps = 0.05 * np.random.randn(dim)
                S = normalize(best + eps)

            Fnew = sharpe_ratio(S, mean_returns, cov_matrix)

            if (Fnew <= Fitness[i]) and (np.random.rand() < A[i]):
                Sol[i] = S
                Fitness[i] = Fnew
                A[i] *= 0.95
                r[i] = r[i] * (1 - np.exp(-0.1 * t))

                if Fnew <= best_fitness:
                    best = S
                    best_fitness = Fnew

        history.append(-best_fitness)

    return best, -best_fitness, history

# === Genetic Algorithm ===
def genetic_algorithm(pop_size=50, generations=100, mutation_rate=0.1):
    dim = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    population = np.random.dirichlet(np.ones(dim), size=pop_size)
    history = []

    for gen in range(generations):
        fitness = np.array([sharpe_ratio(ind, mean_returns, cov_matrix) for ind in population])
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        history.append(-sharpe_ratio(population[0], mean_returns, cov_matrix))

        offspring = []
        for _ in range(pop_size // 2):
            parent1, parent2 = population[np.random.randint(0, pop_size//2, 2)]
            crossover = np.random.rand(dim) < 0.5
            child = np.where(crossover, parent1, parent2)
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.1, dim)
            child = normalize(child)
            offspring.append(child)

        population = np.vstack((population[:pop_size//2], offspring))

    best = population[0]
    return best, -sharpe_ratio(best, mean_returns, cov_matrix), history

# === Particle Swarm Optimization ===
def pso(num_particles=30, max_iter=100):
    dim = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    particles = np.random.dirichlet(np.ones(dim), size=num_particles)
    velocities = np.zeros((num_particles, dim))
    personal_best = particles.copy()
    personal_best_val = np.array([sharpe_ratio(p, mean_returns, cov_matrix) for p in particles])
    global_best = personal_best[np.argmin(personal_best_val)]
    history = []

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = 0.5 * velocities[i] + r1 * (personal_best[i] - particles[i]) + r2 * (global_best - particles[i])
            particles[i] += velocities[i]
            particles[i] = normalize(particles[i])

            score = sharpe_ratio(particles[i], mean_returns, cov_matrix)
            if score < personal_best_val[i]:
                personal_best[i] = particles[i]
                personal_best_val[i] = score
                if score < sharpe_ratio(global_best, mean_returns, cov_matrix):
                    global_best = particles[i]

        history.append(-sharpe_ratio(global_best, mean_returns, cov_matrix))

    return global_best, -sharpe_ratio(global_best, mean_returns, cov_matrix), history

# === Grey Wolf Optimizer ===
def gwo(num_wolves=30, max_iter=100):
    dim = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    wolves = np.random.dirichlet(np.ones(dim), size=num_wolves)
    fitness = np.array([sharpe_ratio(w, mean_returns, cov_matrix) for w in wolves])
    alpha, beta, delta = wolves[np.argsort(fitness)[:3]]
    history = []

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        for i in range(num_wolves):
            A1, C1 = a * (2*np.random.rand(dim) - 1), 2*np.random.rand(dim)
            D_alpha = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            A2, C2 = a * (2*np.random.rand(dim) - 1), 2*np.random.rand(dim)
            D_beta = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            A3, C3 = a * (2*np.random.rand(dim) - 1), 2*np.random.rand(dim)
            D_delta = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            wolves[i] = normalize((X1 + X2 + X3) / 3)

        fitness = np.array([sharpe_ratio(w, mean_returns, cov_matrix) for w in wolves])
        alpha, beta, delta = wolves[np.argsort(fitness)[:3]]
        history.append(-sharpe_ratio(alpha, mean_returns, cov_matrix))

    return alpha, -sharpe_ratio(alpha, mean_returns, cov_matrix), history

# === Run all optimizers multiple times and compare best results ===
if __name__ == "__main__":
    algorithms = {
        'Bat Algorithm': bat_algorithm,
        'Genetic Algorithm': genetic_algorithm,
        'Particle Swarm': pso,
        'Grey Wolf': gwo
    }

    num_runs = 5
    results = {name: [] for name in algorithms}
    best_portfolios = {}
    histories = {}

    for _ in range(num_runs):
        for name, algo in algorithms.items():
            weights, sr, history = algo()
            results[name].append(sr)
            if name not in best_portfolios or sr > max(results[name]):
                best_portfolios[name] = weights
                histories[name] = history

    avg_results = {name: np.max(scores) for name, scores in results.items()}

    for name, score in avg_results.items():
        print(f"{name} Sharpe Ratio: {score:.4f}")

    # === Bar Chart of Best Sharpe Ratios ===
    plt.figure(figsize=(10, 6))
    plt.bar(avg_results.keys(), avg_results.values(), color='skyblue')
    plt.title(f"Best Sharpe Ratio after {num_runs} Runs")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # === Convergence Graph ===
    plt.figure(figsize=(12, 6))
    for name, history in histories.items():
        plt.plot(history, label=name)
    plt.title("Best Convergence per Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Bar Charts of Best Portfolio Weights ===
    for name, weights in best_portfolios.items():
        top_indices = np.argsort(weights)[-TOP_N_STOCKS:][::-1]
        top_weights = weights[top_indices]
        top_labels = [returns.columns[i] for i in top_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_labels[::-1], top_weights[::-1], color='lightcoral')
        plt.xlabel("Weight")
        plt.title(f"Top {TOP_N_STOCKS} Portfolio Weights - {name}")
        plt.tight_layout()
        plt.show()
