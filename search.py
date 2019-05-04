# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    cache = util.Stack()
    cache.push((start, []))
    arr = []
    sol = ""
    while not cache.isEmpty():
        current, sol = cache.pop()
        if problem.isGoalState(current):
            break
        arr.append(current)
        for node, action, weight in problem.getSuccessors(current):
            if not node in arr:
                cache.push((node, sol + [action]))
    return sol

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    pacman_direction = []
    start_state = problem.getStartState()
    visit = util.Queue()
    visit.push(start_state)
    tovisit = []
    parent = {}
    direction = {}
    goal, goalDir, direction, parent = bfsHelper(problem, visit, tovisit, parent, direction)
    pacman_direction.append(goalDir)
    curr = goal
    while curr != start_state:
        p = parent[curr]
        if p in direction:
            pacman_direction.insert(0, direction[p])
        curr = p
    return pacman_direction

def bfsHelper(problem, visit, tovisit, parent, direction):
    curr = visit.pop()
    if problem.isGoalState(curr):
        return curr, direction[curr], direction, parent
    tovisit.append(curr)
    descend = problem.getSuccessors(curr)
    for (desc, action, cost) in descend:
        if desc not in tovisit and desc not in visit.list:
            visit.push(desc)
            direction[desc] = action
            parent[desc] = curr
    return bfsHelper(problem, visit, tovisit, parent, direction)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    pacman_direction = []
    start_state = problem.getStartState()
    visit = util.PriorityQueue()
    visit.push(start_state, 0)
    tovisit = []
    parent = {}
    direction = {}
    pathcost = {start_state: 0}
    goal, goalDir, direction, parent = uniformHelper(problem, visit, tovisit, parent, direction, pathcost)
    pacman_direction.append(goalDir)
    curr = goal
    while curr != start_state:
        p = parent[curr]
        if p in direction:
            pacman_direction.insert(0, direction[p])
        curr = p
    return pacman_direction


def uniformHelper(problem, visit, tovisit, parent, direction, costs):
    curr = visit.pop()
    if problem.isGoalState(curr):
        return curr, direction[curr], direction, parent
    tovisit.append(curr)
    descend = problem.getSuccessors(curr)
    for (desc, action, cost) in descend:
        if desc not in tovisit:
            total_cost = costs[curr] + cost
            if desc in costs:
                if total_cost < costs[desc]:
                    costs[desc] = total_cost
                    visit.push(desc, total_cost)
                    parent[desc] = curr
                    direction[desc] = action
            else:
                costs[desc] = total_cost
                visit.push(desc, total_cost)
                parent[desc] = curr
                direction[desc] = action
    return uniformHelper(problem, visit, tovisit, parent, direction, costs)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    openList = util.PriorityQueue()
    start = problem.getStartState()
    openList.push((start, [], 0), 0)
    closedList = []
    while not openList.isEmpty():
        curr, solution, cost = openList.pop()
        if problem.isGoalState(curr):
            break
        if curr not in closedList:
            closedList.append(curr)
            for node, move, weight in problem.getSuccessors(curr):
                if node not in closedList:
                    sum = (cost + weight) + (heuristic(node, problem))
                    openList.push((node, solution + [move], (cost + weight)), sum)
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
