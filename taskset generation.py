import csv
import random
import numpy as np


def generate_random_periods_discrete(num_periods: int, num_sets: int, available_periods: list):
    try:
        periods = np.random.choice(available_periods, size=(num_sets, num_periods)).tolist()
    except AttributeError:
        p = np.array(available_periods)
        periods = p[np.random.randint(len(p), size=(num_sets, num_periods))].tolist()

    return periods


def generate_uunifastdiscard(nsets: int, u: float, n: int):
    sets = []
    while len(sets) < nsets:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)

        if all(ut <= 1 for ut in utilizations):
            sets.append(utilizations)

    return sets


def generate_tasksets(utilizations, periods, filename):
    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    result = [[(trunc(ui * pi, 6), trunc(pi, 6)) for ui, pi in zip(us, ps)]
              for us, ps in zip(utilizations, periods)]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)

    return result


def myfunc(e):
    return e[0][0]


def assign_preemption_level(taskset: list):
    taskset = sorted(taskset, key=lambda x: x[0], reverse=True)
    i = 1
    j = len(taskset) - 1
    while j >= 0:
        taskset[j][0] += (i,)
        if j != 0:
            if taskset[j-1][0][1] != taskset[j][0][1]:
                i = i + 1
        j = j - 1

    return taskset

tmp = generate_uunifastdiscard(1, 5, 100)

x =  generate_random_periods_discrete(100, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
temp = generate_tasksets(generate_uunifastdiscard(1, 5, 100),
                         generate_random_periods_discrete(100, 100, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "a.csv")
temp = temp[0]
assign_preemption_level(temp)
print(temp)
