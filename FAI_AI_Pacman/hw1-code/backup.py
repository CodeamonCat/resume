import heapq
from collections import deque

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    Q = deque([])
    visited = set()
    parentDict = {}
    mazeStart = maze.getStart()
    Q.append(mazeStart)
    visited.add(mazeStart)
    parentDict[mazeStart] = None
    mazeObjectives = maze.getObjectives()[0]
    while(len(Q) > 0):
        u = Q.popleft()
        uNeighbors = maze.getNeighbors(u[0], u[1])
        if (u == mazeObjectives):
            path = u
            result = []
            while(path != None):
                result.append(path)
                if parentDict[path] == maze.getStart():
                    result.append(parentDict[path])
                    result.reverse()
                    # print(result)
                    # print(maze.isValidPath(result))
                    return result
                elif parentDict[path] == None:
                    print("Error path")
                else:
                    path = parentDict[path]
        for vertex in uNeighbors:
            if not (vertex in visited):
                visited.add(vertex)
                parentDict[vertex] = u
                Q.append(vertex)
    # return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    """
    f(n) = g(n)+h(n)
    g(n) = Euclidean Distance (from pos to start)
    h(n) = Manhattan Distance (from pos to end)
    """
    def calH(start, end): return abs(start[0]-end[0])+abs(start[1]-end[1])
    h = []
    visited = set()
    parentDict = {}
    mazeStart = maze.getStart()
    mazeObjectives = maze.getObjectives()[0]
    startF = calH(mazeStart, mazeObjectives)
    heapq.heappush(h, tuple([startF, 0, mazeStart]))
    visited.add(mazeStart)
    parentDict[mazeStart] = None
    while(len(h) > 0):
        _, mazeG, u = heapq.heappop(h)
        uNeighbors = maze.getNeighbors(u[0], u[1])
        if (u == mazeObjectives):
            path = u
            result = []
            while(path != None):
                result.append(path)
                if parentDict[path] == maze.getStart():
                    result.append(parentDict[path])
                    result.reverse()
                    # print(result)
                    print(maze.isValidPath(result))
                    return result
                elif parentDict[path] == None:
                    print("Error path")
                else:
                    path = parentDict[path]
        for vertex in uNeighbors:
            if not (vertex in visited):
                visited.add(vertex)
                parentDict[vertex] = u
                heapq.heappush(h, tuple([mazeG+calH(vertex, mazeObjectives), mazeG+1, vertex]))
    # return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    """
    astar with special heuristics
    Heuristics: find the closest
    """
    def calH(start, neverVisit):
        H = float('inf')
        cornerFlag = tuple()
        tmp = [item for item in neverVisit]
        if len(tmp) <= 1:
            return abs(start[0]-tmp[0][0])+abs(start[1]-tmp[0][1])
        elif len(tmp) == 2:
            return min((abs(start[0]-tmp[0][0])+abs(start[1]-tmp[0][1])), \
                (abs(start[0]-tmp[1][0])+abs(start[1]-tmp[1][1])))
        else:
            for corner in tmp:
                diff = abs(start[0]-corner[0])+abs(start[1]-corner[1])
                if diff < H:
                    H = diff; cornerFlag = corner
            tmp.pop(tmp.index(cornerFlag))
            return H + calH(cornerFlag, tmp)

    mazeStart = maze.getStart()
    neverVisit = maze.getObjectives()
    result = list()
    while(neverVisit):
        h = []
        visited = set()
        parentDict = {}
        heapq.heappush(h, tuple([calH(mazeStart, neverVisit), 0, mazeStart]))
        visited.add(mazeStart)
        parentDict[mazeStart] = None
        next = tuple()
        while(len(h) > 0):
            _, mazeG, u = heapq.heappop(h)
            uNeighbors = maze.getNeighbors(u[0], u[1])

            if (u in neverVisit):
                path = u
                subResult = []
                while(path != None):
                    subResult.append(path)
                    if parentDict[path] == mazeStart:
                        subResult.reverse()
                        result.extend(subResult)
                        next = u
                        break
                    elif parentDict[path] == None:
                        print("Error path")
                    else:
                        path = parentDict[path]
                break
            for vertex in uNeighbors:
                if not (vertex in visited):
                    visited.add(vertex)
                    parentDict[vertex] = u
                    heapq.heappush(h, tuple([mazeG+calH(vertex, neverVisit), mazeG+1, vertex]))
        # Update infors
        mazeStart = next
        neverVisit.pop(neverVisit.index(mazeStart))
    result.insert(0, maze.getStart())
    # print(maze.isValidPath(result))
    return result
    # return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def calM(p1, p2): return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
    
    def find(parent, index):
        if parent[index] == None: return index
        return find(parent, parent[index])
    
    def union(parent, rank, left, right):
        leftRoot = find(parent, left)
        rightRoot = find(parent, right)
        if rank[leftRoot] > rank[rightRoot]:
            parent[rightRoot] = leftRoot
        elif rank[leftRoot] < rank[rightRoot]:
            parent[leftRoot] = rightRoot
        else:
            parent[rightRoot] = leftRoot
            rank[leftRoot] = rank[leftRoot] + 1

    def calH(start, neverVisit):

        if len(neverVisit) == 0: return 0
        tmp = [item for item in neverVisit]
        graph = list()
        for i in range(len(tmp)):
            for j in range(1, len(tmp)):
                graph.append([tmp[i], tmp[j], calM(tmp[i], tmp[j])])
        
        parentDict = {}
        rankDict = {}
        # initialization of MST
        for food in tmp: parentDict[food] = None ;rankDict[food] = 0
        MSTLength = 0
        explore = 0
        index = 0
        while(explore < (len(tmp)-1)):
            prev, next, weight = graph[index]
            index = index + 1
            left = find(parentDict, prev)
            right = find(parentDict, next)
            if left != right:
                explore += 1
                MSTLength += weight
                union(parentDict, rankDict, left, right)
        
        choose = min([calM(start, food) for food in tmp])
        return choose + MSTLength

    mazeStart = maze.getStart()
    neverVisit = maze.getObjectives()
    result = list()
    mazeStart = maze.getStart()
    neverVisit = maze.getObjectives()
    result = list()
    while(neverVisit):
        h = []
        visited = set()
        parentDict = {}
        heapq.heappush(h, tuple([calH(mazeStart, neverVisit), 0, mazeStart]))
        visited.add(mazeStart)
        parentDict[mazeStart] = None
        next = tuple()
        while(len(h) > 0):
            _, mazeG, u = heapq.heappop(h)
            uNeighbors = maze.getNeighbors(u[0], u[1])

            if (u in neverVisit):
                path = u
                subResult = []
                while(path != None):
                    subResult.append(path)
                    if parentDict[path] == mazeStart:
                        subResult.reverse()
                        result.extend(subResult)
                        next = u
                        break
                    elif parentDict[path] == None:
                        print("Error path")
                    else:
                        path = parentDict[path]
                break
            for vertex in uNeighbors:
                if not (vertex in visited):
                    visited.add(vertex)
                    parentDict[vertex] = u
                    heapq.heappush(h, tuple([mazeG+calH(vertex, neverVisit), mazeG+1, vertex]))
        # Update infors
        mazeStart = next
        neverVisit.pop(neverVisit.index(mazeStart))
    result.insert(0, maze.getStart())
    # print(maze.isValidPath(result))
    return result
    # return []