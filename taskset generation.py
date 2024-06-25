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
        t.core = assigned_core

    return result

def get_time_remaining_to_next_arriving_tasks_from_t(tasks, time_):
    remaining_times_to_arrival_from_t = []
    for task in tasks:
        r = (task.period - (time_ % task.period)) % task.period
        if r == 0:
            if task.IsArrived:
                r = task.period
            else:
                r = 0

        remaining_times_to_arrival_from_t.append(r)
    
    min_remaining_time_i = remaining_times_to_arrival_from_t.index(min(remaining_times_to_arrival_from_t))
    return tasks[min_remaining_time_i], remaining_times_to_arrival_from_t[min_remaining_time_i]

class MyTask:
    task_count = 0
    resources_allocated = []

    def __init__(self, utilization, period, executionTime, preemptionLevel, resourceAllocationTimes, core = -1):
        self.utilization = utilization
        self.period = period
        self.execution_time = executionTime
        self.preemption_level = preemptionLevel
        self.resource_allocation_times = resourceAllocationTimes
        self.task_number = MyTask.task_count
        MyTask.task_count += 1

        self.core = core

        self.executed_time = 0
        self.IsArrived = True
        self.IsWaiting = False
        self.NeededResource = -1
    
    def get_allocated_resources(self):
        for res in self.resource_allocation_times:
            if res[0] <= self.executed_time and res[1] > self.executed_time:
                yield self.resource_allocation_times.index(res)
    
    def get_next_releasing(self):
        return self.get_next_releasing_resource_and_remaining_time_to_release()


    def get_next_releasing_resource_and_remaining_time_to_release(self):
        if not self.IsArrived:
            return [None, None]
        
        allocated_resouce_indices = list(self.get_allocated_resources())

        if len(allocated_resouce_indices) == 0:
            return [None, None]
        
        remaining_resource_allocating_times = \
            [(self.resource_allocation_times[res_index][1]- self.executed_time) for res_index in allocated_resouce_indices] 
        next_relesing_resource_index_in_list = \
            remaining_resource_allocating_times.index(min(remaining_resource_allocating_times))
        
        next_releasing_resource = allocated_resouce_indices[next_relesing_resource_index_in_list]
        time_to_release = remaining_resource_allocating_times[next_relesing_resource_index_in_list]
        
        if self.NeededResource == next_releasing_resource:
            return [None, None]
        
        return next_releasing_resource, time_to_release    
    
    def get_next_allocating(self):
        return self.get_next_allocating_resource_and_remaining_time_to_allocate()
    
    def get_next_allocating_resource_and_remaining_time_to_allocate(self):
        if not self.IsArrived:
            return [None, None]
        
        not_allocated_resouce_indices2 = filter(lambda i : self.executed_time <= self.resource_allocation_times[i][0], \
                                               range(0, len(self.resource_allocation_times)))
        
        not_allocated_resouce_indices = []
        for i in not_allocated_resouce_indices2:
            if not (MyTask.resources_allocated[i] == self or self.NeededResource == i):
                not_allocated_resouce_indices.append(i)
        
        not_allocated_resouce_indices = list(not_allocated_resouce_indices)
        if len(not_allocated_resouce_indices) == 0:
            return [None, None]

        remaining_time_to_allocate = \
            [(self.resource_allocation_times[res_index][0] - self.executed_time) for res_index in not_allocated_resouce_indices] 
        next_allocating_resource_index_in_list = \
            remaining_time_to_allocate.index(min(remaining_time_to_allocate))
        
        next_allocating_resource = not_allocated_resouce_indices[next_allocating_resource_index_in_list]
        time_to_allocate = remaining_time_to_allocate[next_allocating_resource_index_in_list]
        
        return next_allocating_resource, time_to_allocate
    
    def execute(self, executed):
        if self.IsWaiting:
            return
        self.executed_time += executed
    
    def start_critical_section(self, resource_index):
        if MyTask.resources_allocated[resource_index] is None:
            MyTask.resources_allocated[resource_index] = self
        else:
            self.NeededResource = resource_index
            self.IsWaiting = True

    def try_continue(self, resource_index):
        if self.IsWaiting:
            if self.NeededResource == resource_index:
                self.NeededResource = -1
                self.IsWaiting = False
                MyTask.resources_allocated[resource_index] = self
                return True
        return False
    
    def finish_critical_section(self, resource_index):
        if MyTask.resources_allocated[resource_index] != self:
            print('finished')
        MyTask.resources_allocated[resource_index] = None
    

    def finish_Task(self):
        self.IsArrived = False
        self.executed_time = 0

    def in_critical_scetion(self):
        return self.IsWaiting or len(list(self.get_allocated_resources())) > 0




n = 5
number_of_cores = 2

temp = generate_tasksets(generate_uunifastdiscard(1, number_of_cores-1, n),
                         generate_random_periods_discrete(n, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "a.csv")


temp = temp[0]

temp = assign_preemption_level(temp)
res_num = random.randint(1, 2)
MyTask.resources_allocated = [None for _ in range(0, res_num)]
resources = assign_resources(temp, res_num)
resources = list(resources)

tasks = list(createTasks(temp, resources))

core_tasks = assign_tasks_to_cores(tasks, number_of_cores)

print(resources)


time_ = 0
delta_time = 0
current_tasks = [None for _ in range(0, number_of_cores)]

tasks_to_run = core_tasks.copy()
# tasks_to_run = [[MyTask(0.5, 20, 5, 100, [(0,3)], 0), MyTask(0.5, 15, 2, 100, [(1,2)], 0)],\
#                 [MyTask(0.5, 20, 5, 100, [(0,3)], 1), MyTask(0.5, 20, 5, 100, [(0,3)], 1)]]

# tasks = np.array(tasks_to_run).flatten().tolist()

for core_num in range(0, number_of_cores):
    to_runs = tasks_to_run[core_num]
    task_to_run = sorted(to_runs, key=lambda t: t.period)[0]
    current_tasks[core_num] = task_to_run

while True:
    closest_releasings = [10000 if t == None or t.get_next_releasing()[1] == None else t.get_next_releasing()[1] for t in current_tasks]
    closest_allocatings = [10000 if t == None or t.get_next_allocating()[1] == None else t.get_next_allocating()[1] for t in current_tasks]

    remaining_time_to_finishs = [10000 if t is None else t.execution_time - t.executed_time for t in current_tasks]

    closest_arriving_time_and_task = get_time_remaining_to_next_arriving_tasks_from_t(tasks, time_)
    closest_arriving = 1000 if closest_arriving_time_and_task[1] is None else closest_arriving_time_and_task[1]

    time_points = \
        [min(closest_releasings), min(closest_allocatings), min(remaining_time_to_finishs), closest_arriving]
    
    closest_time_point_index = time_points.index(min(time_points))

    if closest_time_point_index == 0:
        delta_time = min(closest_releasings)

        closest_releasing = closest_releasings.index(min(closest_releasings))
        t : MyTask = current_tasks[closest_releasing]
        resource_index_to_release = t.get_next_releasing()[0]

        for current_task in current_tasks:
            current_task.execute(delta_time)

        t.finish_critical_section(resource_index_to_release)

        for current_task in current_tasks:
            if current_task.try_continue(delta_time):
                break
        
        to_runs = tasks_to_run[t.core]
        to_runs = sorted([task for task in to_runs if task.IsArrived], key=lambda t: t.period)
        for to_run in to_runs:
            if to_run.preemptionLevel > t.preemptionLevel:
                current_tasks[t.core] = to_run
                break


    if closest_time_point_index == 1:
        closest_allocating = closest_allocatings.index(min(closest_allocatings))
        t : MyTask = current_tasks[closest_allocating]
        resource_index_to_allocate = t.get_next_allocating()[0]

        delta_time = min(closest_allocatings) 
        for current_task in current_tasks:
            current_task.execute(delta_time)

        t.start_critical_section(resource_index_to_allocate)


    if closest_time_point_index == 2:
        delta_time = min(remaining_time_to_finishs)

        for current_task in current_tasks:
            current_task.execute(delta_time)
        
        finished_task_core_number = remaining_time_to_finishs.index(min(remaining_time_to_finishs))
        task_to_finish = current_tasks[finished_task_core_number]
        task_to_finish.finish_Task()
        
        to_runs = tasks_to_run[finished_task_core_number]
        task_to_run = sorted([task for task in to_runs if task.IsArrived], key=lambda t: t.period)[0]
        current_tasks[finished_task_core_number] = task_to_run


    if closest_time_point_index == 3:
        delta_time = closest_arriving
        
        for current_task in current_tasks:
            current_task.execute(delta_time)
        
        arrived_task = closest_arriving_time_and_task[0]
        arrived_task.IsArrived = True
        
        running_task = current_tasks[arrived_task.core]
        if not running_task.in_critical_scetion():
            if arrived_task.preemption_level > running_task.preemption_level:
                current_tasks[arrived_task.core] = arrived_task


    time_ += delta_time


    








