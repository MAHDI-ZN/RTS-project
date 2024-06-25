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

def trunc(x, p):
    return int(x * 10 ** p) / float(10 ** p)

def generate_tasksets(utilizations, periods, filename):
    z = 3
    result = [[(trunc(ui * pi, z), trunc(pi, z)) for ui, pi in zip(us, ps)]
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
        res_points = [trunc(random.uniform(0, executionTime), 3) for _ in range(0, 2*res_num)]
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

def get_next_arriving_tasks_from_t(tasks, t):
    remaining_times_to_arrival_from_t = [task.period - (t % task.period) for task in tasks]
    min_remaining_time_i = remaining_times_to_arrival_from_t.index(min(remaining_times_to_arrival_from_t))
    return tasks[min_remaining_time_i]

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

        self.executed_time = 0
        self.IsArrived = True
    
    def get_allocated_resources(self):
        for res in self.resource_allocation_times:
            if res[0] <= self.executed_time and res[1] > self.executed_time:
                yield self.resource_allocation_times.index(res)
    

    def get_next_releasing_resource_and_remaining_time_to_release(self):
        if not self.IsArrived:
            return
        
        allocated_resouce_indices = self.get_allocated_resources()
        remaining_resource_allocating_times = \
            [(self.resource_allocation_times[res_index][1]- self.executed_time) for res_index in allocated_resouce_indices] 
        next_relesing_resource_index_in_list = \
            remaining_resource_allocating_times.index(min(remaining_resource_allocating_times))
        
        next_releasing_resource = allocated_resouce_indices[next_relesing_resource_index_in_list]
        time_to_release = remaining_resource_allocating_times[next_relesing_resource_index_in_list]
        
        return next_releasing_resource, time_to_release    
    
    def get_next_allocating_resource_and_remaining_time_to_allocate(self):
        if not self.IsArrived:
            return
        
        not_allocated_resouce_indices = filter(lambda i : self.executed_time < self.resource_allocation_times[i][0], \
                                               range(0, len(self.resource_allocation_times)))
        
        remaining_time_to_allocate = \
            [(self.executed_time - self.resource_allocation_times[res_index][0]) for res_index in not_allocated_resouce_indices] 
        next_allocating_resource_index_in_list = \
            remaining_time_to_allocate.index(min(remaining_time_to_allocate))
        
        next_allocating_resource = not_allocated_resouce_indices[next_allocating_resource_index_in_list]
        time_to_allocate = remaining_time_to_allocate[next_allocating_resource_index_in_list]
        
        return next_allocating_resource, time_to_allocate


n = 5
number_of_cores = 2

temp = generate_tasksets(generate_uunifastdiscard(1, number_of_cores-1, n),
                         generate_random_periods_discrete(n, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "a.csv")


temp = temp[0]

temp = assign_preemption_level(temp)
res_num = random.randint(2, 2)
resources = assign_resources(temp, res_num)
resources = list(resources)

tasks = list(createTasks(temp, resources))

core_tasks = assign_tasks_to_cores(tasks, number_of_cores)

print(resources)


t = 0
current_tasks = [None for _ in range(0, number_of_cores)]
tasks_to_run = core_tasks.copy()

for core_num in range(0, number_of_cores):
    to_runs = tasks_to_run[core_num]
    task_to_run = sorted(to_runs, key=lambda t: t.period)[0]
    current_tasks[core_num] = task_to_run

while True:
    

    








