from collections import deque
import heapq
import math

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

# Heurísticas
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Función para leer el laberinto desde un archivo
def read_maze(filename):
    with open(filename, 'r') as f:
        maze = []
        start_pos = None
        exit_pos = None
        for i, line in enumerate(f):
            row = [int(x) for x in line.strip().split()]
            maze.append(row)
            if 2 in row:
                start_pos = (i, row.index(2))
            if 3 in row:
                exit_pos = (i, row.index(3))
    return maze, start_pos, exit_pos

# Función para obtener los vecinos contiguos de una celda en el laberinto
def get_neighbors(maze, i, j):
    neighbors = []
    if i > 0 and maze[i - 1][j] == 1:
        neighbors.append((i - 1, j))
    if i < len(maze) - 1 and maze[i + 1][j] == 1:
        neighbors.append((i + 1, j))
    if j > 0 and maze[i][j - 1] == 1:
        neighbors.append((i, j - 1))
    if j < len(maze[0]) - 1 and maze[i][j + 1] == 1:
        neighbors.append((i, j + 1))
    return neighbors

# Búsqueda en anchura
def breadth_first_search(start, goal, graph):
    iterations = 0
    frontier = QueueFIFO()
    frontier.insert(start)
    came_from = {start: None}

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from:
                frontier.insert(next_node)
                came_from[next_node] = current
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    if path:
        path.reverse()

    return path, iterations

# Búsqueda en profundidad
def depth_first_search(start, goal, graph):
    iterations = 0
    frontier = StackLIFO()
    frontier.insert(start)
    came_from = {start: None}

    while not frontier.empty():
        iterations += 1
        current = frontier.remove_first()

        if current == goal:
            break

        for next_node in graph[current]:
            if next_node not in came_from:
                frontier.insert(next_node)
                came_from[next_node] = current
    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    if path:
        path.reverse()

    return path, iterations

# Búsqueda con profundidad limitada
def depth_limited_search(start, goal, graph, depth_limit):
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

    path = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    if path:
        path.reverse()

    return path, iterations

# Búsqueda de mejor primero
def greedy_best_first_search(start, goal, graph, heuristic_func):
    iterations = 0
    frontier = PriorityQueue()
    frontier.insert(start, heuristic_func(start, goal))
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
    if path:
        path.reverse()

    return path, iterations

# Búsqueda A*
def a_star_search(start, goal, graph, heuristic_func):
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
            new_cost = cost_so_far[current] + 1  # Assuming uniform cost for simplicity
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
    if path:
        path.reverse()

    return path, iterations

# Lee el laberinto desde el archivo
maze_filename = 'laberinto.txt'
maze, start_pos, exit_pos = read_maze(maze_filename)

# Encuentra la posición de entrada y salida
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i][j] == 2:
            start_pos = (i, j)
        elif maze[i][j] == 3:
            exit_pos = (i, j)

# Define el grafo del laberinto
graph = {}
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i][j] == 1:
            neighbors = get_neighbors(maze, i, j)
            graph[(i, j)] = neighbors

# Calcula las heurísticas
heuristics = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance
}

# Aplica los algoritmos de búsqueda sin heurísticas
for algorithm, algorithm_name in [
    (breadth_first_search, "Breadth-first search"),
    (depth_first_search, "Depth-first search"),
    (depth_limited_search, "Depth-limited search")
]:
    if algorithm == depth_limited_search:
        depth_limit = 10  # Puedes ajustar el límite de profundidad según sea necesario
        path, iterations = algorithm(start_pos, exit_pos, graph, depth_limit)
    else:
        path, iterations = algorithm(start_pos, exit_pos, graph)
    print(f"{algorithm_name}: Path found: {path}, Iterations: {iterations}")

# Aplica los algoritmos de búsqueda con heurísticas
for heuristic_name, heuristic_func in heuristics.items():
    print(f"Applying {heuristic_name} heuristic:")
    for algorithm, algorithm_name in [
        (greedy_best_first_search, "Greedy best-first search"),
        (a_star_search, "A* search")
    ]:
        path, iterations = algorithm(start_pos, exit_pos, graph, heuristic_func)
        print(f"{algorithm_name}: Path found: {path}, Iterations: {iterations}")
    print("\n")
