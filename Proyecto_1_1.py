import pandas as pd
from collections import deque, defaultdict
import heapq
import time
import turtle

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
# ----------------- Implementacion de algoritmos ----------------
def load_maze(file_path):
    maze = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(char) for char in line.strip()]
            maze.append(row)
    return maze

def convertir_maze_a_grafo(maze):
    graph = defaultdict(dict)
    rows, cols = len(maze), len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1 or maze[i][j] == 2 or maze[i][j] == 3:  
                neighbors = []
                if i > 0 and maze[i - 1][j] != 0:  
                    neighbors.append((i - 1, j))
                if i < rows - 1 and maze[i + 1][j] != 0:
                    neighbors.append((i + 1, j))
                if j > 0 and maze[i][j - 1] != 0:
                    neighbors.append((i, j - 1))
                if j < cols - 1 and maze[i][j + 1] != 0:
                    neighbors.append((i, j + 1))

                graph[(i, j)] = {neighbor: 1 for neighbor in neighbors}

    return graph

def draw_maze(maze):
    screen = turtle.Screen()
    screen.setup(width=600, height=600)
    screen.setworldcoordinates(0, len(maze), len(maze[0]), 0)

    turtle.tracer(0, 0)

    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                turtle.fillcolor("black")
            elif maze[y][x] == 0:
                turtle.fillcolor("white")
            elif maze[y][x] == 2:
                turtle.fillcolor("green")
            elif maze[y][x] == 3:
                turtle.fillcolor("red")

            turtle.penup()
            turtle.goto(x, len(maze) - y - 1)
            turtle.pendown()
            turtle.begin_fill()
            for _ in range(4):
                turtle.forward(1)
                turtle.right(90)
            turtle.end_fill()

    turtle.update() 
    turtle.done()


def Breadth_first_search(graph, start, goal):
    iterations = 0
    frontier = QueueFIFO()
    frontier.insert(start)
    came_from = {start: None}
    explored = set() 

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()
        explored.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from and next_node not in explored:
                frontier.insert(next_node)
                came_from[next_node] = current
    else:
        return [], iterations  

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations

def depth_first_search(graph, start, goal):
    iterations = 0
    frontier = StackLIFO()
    frontier.insert(start)
    came_from = {start: None}
    explored = set()  

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()
        explored.add(current)

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from and next_node not in explored:
                frontier.insert(next_node)
                came_from[next_node] = current
    else:
        return [], iterations

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def check_path_existence(graph, start, goal):
    frontier = QueueFIFO()
    frontier.insert(start)
    explored = set()  

    while not frontier.empty():
        current = frontier.remove_first()
        explored.add(current)

        if current == goal:
            return True

        for next_node in graph[current]:
            if next_node not in explored:
                frontier.insert(next_node)
    return False

def uniform_cost_search(graph, start, goal):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + 1  
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost
                frontier.insert(next_node, priority)
                came_from[next_node] = current
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def a_star_search(graph, start, goal, heuristic_func):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]

        if current == goal:
            break

        for next_node in graph[current]:
            new_cost = cost_so_far[current] + 1  
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic_func(next_node, goal)
                frontier.insert(next_node, priority)
                came_from[next_node] = current
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations

def depth_delimited_search(graph, start, goal, depth_limit):
    iterations = 0
    frontier = StackLIFO()
    frontier.insert((start, 0))  
    came_from = {start: None}

    while not frontier.empty():
        iterations += 1
        current, depth = frontier.remove_first()

        if current == goal:
            break

        if depth < depth_limit:
            for next_node in graph[current]:
                if next_node not in came_from:
                    frontier.insert((next_node, depth + 1))
                    came_from[next_node] = current

    if goal not in came_from:
        return [], iterations 

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations

def greedy_best_first_search(graph, start, goal, heuristic_func):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, 0)
    came_from = {start: None}

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()[1]

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from:
                priority = heuristic_func(next_node, goal)
                frontier.insert(next_node, priority)
                came_from[next_node] = current
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()

    return path, iterations


def euclidean_distance(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def manhattan_distance(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)

def find_start(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 2:
                return (i, j)
    return None 

def find_goal(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 3: 
                return (i, j)  
    return None  




def check_path_existence(graph, start, goal):
    frontier = QueueFIFO()
    frontier.insert(start)
    explored = set()  

    while not frontier.empty():
        current = frontier.remove_first()
        explored.add(current)

        if current == goal:
            return True

        for next_node in graph[current]:
            if next_node not in explored:
                frontier.insert(next_node)
    return False



maze_file = 'Prueba_3.txt'
maze = load_maze(maze_file)

maze_graph = convertir_maze_a_grafo(maze)

start = list(maze_graph.keys())[0]
goal = list(maze_graph.keys())[-1]

if start is None:
    print("No se encontró el punto de inicio (2) en el laberinto.")
elif goal is None:
    print("No se encontró el punto de salida (3) en el laberinto.")
elif not check_path_existence(maze_graph, start, goal):
    print("No hay un camino posible entre el punto de inicio y el punto de salida.")
else:
    # Dibujar el laberinto
    draw_maze(maze)
    
    algorithms = {
        'Breadth First Search': Breadth_first_search,
        'Depth First Search': depth_first_search,
        'greedy Best First Search (Euclidean Distance Heuristic)': lambda graph, start, goal: greedy_best_first_search(graph, start, goal, euclidean_distance),
        'Depth-delimited Search (Depth Limit = 10)': lambda graph, start, goal: depth_delimited_search(graph, start, goal, 500),
        'A* Search (Euclidean Distance Heuristic)': lambda graph, start, goal: a_star_search(graph, start, goal, euclidean_distance),
        'A* Search (Manhattan Distance Heuristic)': lambda graph, start, goal: a_star_search(graph, start, goal, manhattan_distance)
    }

    for name, algorithm in algorithms.items():
        print(f'\n{name}:')
        start_time = time.time()
        path, iterations = algorithm(maze_graph, start, goal)
        end_time = time.time()
        print(f'Path found: {path}')
        print(f'Steps:', len(path))
        print(f'Iterations:', iterations)
        print(f'Time: {end_time - start_time} s')