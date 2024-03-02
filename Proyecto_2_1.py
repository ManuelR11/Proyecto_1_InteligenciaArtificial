'''
Manuel Rodas 21509
Universidad del Valle 
Inteligencia Artificial
Laboratorio 01
Solving Problems by Searching
'''


import pandas as pd
from collections import deque
import heapq
import time



#----------------- Implemente estructuras de colas FIFO, LIFO y PRIORITY ------------
# Clase FIFO
class QueueFIFO:
    def __init__(self):
        self.queue = deque()

    def empty(self):
        return len(self.queue) == 0

    def first(self):
        return self.queue[0]

    def remove_first(self):
        return self.queue.popleft()

    def insert(self, item):
        self.queue.append(item)
        return self.queue

# Clase LIFO
class StackLIFO:
    def __init__(self):
        self.stack = deque()

    def empty(self):
        return len(self.stack) == 0

    def first(self):
        return self.stack[-1]

    def remove_first(self):
        return self.stack.pop()

    def insert(self, item):
        self.stack.append(item)
        return self.stack

# Clase de prioridad
class PriorityQueue:
    def __init__(self):
        self.queue = []
        heapq.heapify(self.queue)

    def empty(self):
        return len(self.queue) == 0

    def first(self):
        return self.queue[0]

    def remove_first(self):
        return heapq.heappop(self.queue)

    def insert(self, item, priority):
        heapq.heappush(self.queue, (priority, item))
        return self.queue


# ----------------- Preparar datos ----------------
def load_maze(file_path):
    maze = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(char) for char in line.strip()]
            maze.append(row)
    return maze


def convertir_maze_a_grafo(maze):
    graph = {}
    rows = len(maze)
    cols = len(maze[0])
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                graph[(i, j)] = []
                if i > 0 and maze[i - 1][j] == 1:
                    graph[(i, j)].append((i - 1, j))
                if i < rows - 1 and maze[i + 1][j] == 1:
                    graph[(i, j)].append((i + 1, j))
                if j > 0 and maze[i][j - 1] == 1:
                    graph[(i, j)].append((i, j - 1))
                if j < cols - 1 and maze[i][j + 1] == 1:
                    graph[(i, j)].append((i, j + 1))
    return graph

def crear_costos(graph):
    costs = {}
    for node in graph:
        for neighbor in graph[node]:
            costs[(node, neighbor)] = 1
    return costs

#funcion que regresa el inicio del laberitno, y es donde encuentra un numero 2
def inicio_maze(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 2:
                return (i, j)
            
#funcion que regresa el final del laberitno, y es donde encuentra un numero 3
def final_maze(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 3:
                return (i, j)
            
def heuristic(maze):
    rows = len(maze)
    cols = len(maze[0])
    heuristic = {}
    for i in range(rows):
        for j in range(cols):
            heuristic[(i, j)] = abs(i - goal[0]) + abs(j - goal[1])
    return heuristic
# ----------------- Implementacion de algoritmos ----------------
def breadth_first_search(start, goal, graph):
    iterations = 0
    frontier = QueueFIFO()
    frontier.insert(start)
    came_from = {start: None}
    visited = set()

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()
        visited.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from and next_node not in visited:
                frontier.insert(next_node)
                came_from[next_node] = current

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def depth_first_search(start, goal, graph):
    iterations = 0
    frontier = StackLIFO()
    frontier.insert(start)
    came_from = {start: None}
    visited = set()

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()
        visited.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from and next_node not in visited:
                frontier.insert(next_node)
                came_from[next_node] = current

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def uniform_cost_search(start, goal, graph, costs):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]
        visited.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + costs[(current, next_node)]
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost
                if next_node not in visited:
                    frontier.insert(next_node, priority)
                    came_from[next_node] = current

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def greedy_best_first_search(start, goal, graph, heuristic):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, heuristic[start])
    came_from = {start: None}
    visited = set()

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]
        visited.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from and next_node not in visited:
                priority = heuristic[next_node]
                frontier.insert(next_node, priority)
                came_from[next_node] = current

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def a_star_search(start, goal, graph, costs, heuristic):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, heuristic[start])
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]
        visited.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + costs[(current, next_node)]
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic[next_node]
                if next_node not in visited:
                    frontier.insert(next_node, priority)
                    came_from[next_node] = current

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


maze_file_path = "laberinto.txt"
maze = load_maze(maze_file_path)
graph = convertir_maze_a_grafo(maze)
costs = crear_costos(graph)
heuristic = {}  # Aquí debes definir tu heurística si estás utilizando algoritmos que la requieren


start = list(graph.keys())[0]
goal = list(graph.keys())[-1]

algorithms = {
    'Búsqueda en amplitud': breadth_first_search,
    'Búsqueda en profundidad': depth_first_search,
    'Búsqueda de costo uniforme': uniform_cost_search,
    'Búsqueda greedy': greedy_best_first_search,
    'Búsqueda A*': a_star_search
}

for name, algorithm in algorithms.items():
    print(f'\n{name}:')
    start_time = time.time()
    path, iterations = algorithm(start, goal, graph)
    end_time = time.time()
    print(f'El camino encontrado: {path}')
    print(f'Iteraciones utilizadas: {iterations}')
    print(f'Tiempo: {end_time - start_time} s')

