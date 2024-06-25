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


def assign_preemption_level(taskset: list):
    taskset = sorted(taskset, key=lambda x: x[0], reverse=True)
    i = 1
    j = len(taskset) - 1
    while j >= 0:        
        taskset[j] += (i,)
        if j != 0:            
            if taskset[j-1][1] != taskset[j][1]:
                i = i + 1        
        j = j - 1
    return taskset



def assign_resources(temp, res_num):
    for t in temp:
        executionTime = t[0]
        res_points = [random.uniform(0, executionTime) for _ in range(0, 2*res_num)]
        res_points.sort()
        tuples_list = [(res_points[i], res_points[i + 1]) for i in range(0, len(res_points), 2)]
        random.shuffle(tuples_list)
        yield tuples_list


def createTasks(temp, resources):
    for i in range(0, len(temp)):
        a = temp[i]
        yield MyTask(a[0]/a[1], a[1], a[0], a[2], resources[i])


def assign_tasks_to_cores(tasks, number_of_cores):
    sorted_tasks = sorted(tasks, key=lambda task: task.utilization, reverse=True)
    result = [[] for _ in range(0, number_of_cores)]
    core_utilizations = [0 for _ in range(0, number_of_cores)]

    for t in sorted_tasks:
        assigned_core = core_utilizations.index(min(core_utilizations))
        core_utilizations[assigned_core] += t.utilization
        result[assigned_core].append(t)

    return result


class MyTask:
    task_count = 0
    def __init__(self, utilization, period, executionTime, preemptionLevel, resourceAllocationTimes):
        self.utilization = utilization
        self.period = period
        self.execution_time = executionTime
        self.preemption_level = preemptionLevel
        self.resource_allocation_times = resourceAllocationTimes
        self.task_number = MyTask.task_count
        MyTask.task_count += 1


n = 100
number_of_cores = 4

temp = generate_tasksets(generate_uunifastdiscard(1, number_of_cores-2, 100),
                         generate_random_periods_discrete(n, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "a.csv")


temp = temp[0]

temp = assign_preemption_level(temp)
res_num = random.randint(2, 2)
resources = assign_resources(temp, res_num)
resources = list(resources)

tasks = list(createTasks(temp, resources))

core_tasks = assign_tasks_to_cores(tasks, number_of_cores)

print(resources)
