# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
import math
import numpy as np
from collections import deque

from sklearn.neighbors import RadiusNeighborsTransformer

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

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
                if diff < H: H = diff; cornerFlag = corner
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
    def calH(start, neverVisit):
        H0 = float("inf")
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
                if diff < H0:
                    H0 = diff; cornerFlag = corner
            tmp.pop(tmp.index(cornerFlag))
            H1 = float("inf")
            for corner in tmp:
                diff = abs(start[0]-corner[0])+abs(start[1]-corner[1])
                if diff < H1: H1 = diff; cornerFlag = corner
            return H0 + H1

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


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def calH(start, neverVisit):
        H0 = float("inf")
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
                if diff < H0:
                    H0 = diff; cornerFlag = corner
            tmp.pop(tmp.index(cornerFlag))
            H1 = float("inf")
            for corner in tmp:
                diff = abs(start[0]-corner[0])+abs(start[1]-corner[1])
                if diff < H1: H1 = diff; cornerFlag = corner
            return H0 + H1

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
    return result
    # return []