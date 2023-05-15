import numpy as np


def f(x):
    # 定义函数和适应度函数
    return x**2 - 4 * x + 4


def fitness(x):
    return 1 / (f(x) + 1e-6)


# 遗传算法参数
pop_size = 100  # 种群大小
cross_rate = 0.8  # 交叉概率
mutate_rate = 0.1  # 变异概率

# 初始化种群
pop = np.random.uniform(-5, 5, size=(pop_size, 1))

# 计算适应度值
fitness_values = np.array([fitness(x) for x in pop])

# 迭代
for i in range(100):
    # 选择
    fitness_values += 1e-6  # 防止fitness_values全为0
    p = fitness_values/fitness_values.sum()
    idx = np.random.choice(np.arange(pop_size), size=pop_size,
                           replace=True, p=np.ravel(fitness_values/fitness_values.sum()))
    parents = pop[idx]

    # 交叉
    for parent in parents:
        if np.random.rand() < cross_rate:
            i_ = np.random.randint(0, pop_size, size=1)
            cross_points = np.random.randint(0, 2, size=1).astype(bool)
            parent[cross_points] = pop[i_, cross_points]

    # 变异
    for parent in parents:
        if np.random.rand() < mutate_rate:
            parent += np.random.normal(0, 0.1, size=1)

    # 计算适应度值
    pop = parents
    fitness_values = np.array([fitness(x) for x in pop])

# 找到最优解
best_idx = np.argmax(fitness_values)
best_x = pop[best_idx]
best_fitness = fitness_values[best_idx]

print("最优解：", best_x)
print("最优解的适应度值：", best_fitness)
