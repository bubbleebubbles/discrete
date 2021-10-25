'''
Implements the Bellman Ford and Dijkstra's shortest path algorithms. 
Authors: Adi Sudhakar and Dasha Chadiuk
'''

import time
import matplotlib.pyplot as plt
from sys import maxsize
import numpy as np

#### VARIABLES ####
def redefine_variables():
    max_val = maxsize #placeholder super big number to represent infinity
    visit_dist = [[0, 0]] #[[visited?, distance to], [visited?, distance to], ...]
    return max_val, visit_dist

'''
#### DEFINE GRAPHS #####
def make_graph():
    # define connections between verticies
    connections = [[0 , 1 , 0 , 1],
                   [1 , 1 , 1 , 0],
                   [0 , 1 , 1 , 1],
                   [1 , 0 , 1 , 0]]

    #define weight of each edge generated
    edge_weight =  [[0 , 10 , 0 , 1],
                    [10 , 10 , 1 , 0],
                    [0 , 1 , 10 , 5],
                    [1 , 0 , 5 , 0]]

    total_vertices = len(connections[0])

    return connections, edge_weight, total_vertices
'''

def visit_next():
    #Identify which vertex to visit next to achieve shortest path
    visited = -1

    # Choosing the vertex with the minimum distance
    for vertex in range(total_vertices):
        if visit_dist[vertex][0] == 0 and (visited < 0 or visit_dist[vertex][1] <= visit_dist[visited][1]):
            visited = vertex

    return visited

def run_dijkstra(connections, edge_weight, total_vertices):
    start = time.time()
    #default min length to inf
    # visit_dist = []
    for vertex in range(total_vertices-1):
        visit_dist.append([0, max_val])

    for vertex in range(total_vertices):
    # Finding the next vertex to be visited.
        to_visit = visit_next()
        for adj_vertex in range(total_vertices):
            # Calculate distance to unvisited adjacent vertices
            if connections[to_visit][adj_vertex] == 1 and visit_dist[adj_vertex][0] == 0:
                updated_dist = visit_dist[to_visit][1] + edge_weight[to_visit][adj_vertex]
            # Updating the distance of the adjacent vertex
                if visit_dist[adj_vertex][1] > updated_dist:
                    visit_dist[adj_vertex][1] = updated_dist
        # mark as visited
        visit_dist[to_visit][0] = 1
    loc = 0 

    # Printing out the shortest distance from the source to each vertex 
    for distance in visit_dist:
        # FOR DEBUG ONLY. COMMENT OUT 'PRINT' LINE FOR GETTING TIME DATA
        # print("The shortest path to vertex [",(ord('a') + loc),"] from vertex [a] is:",distance[1])
        loc = loc + 1
    end = time.time()
    
    duration = end - start
    return duration


def bellman_ford(graph, total_vertices, total_edges, source):
    '''
    Runs the Bellman Ford algorithm on a given graph, determining the distance
    of all vertices from a source vertex. 
    Args:
        graph: a graph in the form of [[u1,v1,w1],...,[un,vn,wn]] where each
        edge is from u to v, and w is the weight of the edge. 
        total_vertices:an int representing the total number of vertices in the
        graph
        total_edges: an int representing the total number of edges in the graph
        source: an int representing the source vertex
    '''

    start = time.time()
    #define all distances between verteces as infinite
    dist = [maxsize]*total_vertices
    #make the source vertex distance as 0
    dist[source] = 0

    '''
    Go through each of the edges and check whether the recorded distance
    between the edges is smaller than the path distance being calculate.
    Replace it with the new distance if it is smaller.
    '''
    for vertex in range (total_vertices-1):
        for edge in range (total_edges):
            if dist[graph[edge][0]] + graph[edge][2] < dist[graph[edge][1]]:
                dist[graph[edge][1]] = dist[graph[edge][0]] + graph[edge][2]
        
        #check if there is a negative weight
        for edge in range (total_edges):
            start_vertex = graph[edge][0]
            end_vertex = graph[edge][1]
            
            weight = graph[edge][2]
            if dist[start_vertex] != maxsize and \
                    dist[start_vertex] + weight < dist[end_vertex]:
                       # print("Contains negative weight")
                       pass
    '''
    print("Vertex Distance")

    #print the vertex and the distance to that vertex from the source
    for vertex in range(total_vertices):
        print(str(vertex) + " " + str(dist[vertex]))
    '''

    end = time.time()
    duration = end - start
    return duration

def graph_to_adjacency(graph, total_vertices, total_edges):
    '''
    Converts from a u,v,w graph (where each edge is from u to v, and w is the
    weight between them) into two adjacency matrices connections, and
    edge_weights. 
    Args: 
        graph: a graph in the form of [[u1,v1,w1],...,[un,vn,wn]] where each
        edge is from u to v, and w is the weight of the edge. 
        total_vertices: an int representing the total number of vertices in
        the graph
        total_edges: an int representing the total number of edges in the graph
    '''
    connections = np.zeros((total_vertices, total_vertices),int)
    edge_weights = np.zeros((total_vertices, total_vertices),int)
    for edge in range(total_edges):
        connections[graph[edge][0]][graph[edge][1]] = 1
        connections[graph[edge][1]][graph[edge][0]] = 1
        edge_weights[graph[edge][0]][graph[edge][1]] = graph[edge][2]
        edge_weights[graph[edge][1]][graph[edge][0]] = graph[edge][2]
   
    return connections, edge_weights
        

def plot_times_hist(dij_times, bell_times):
    plt.hist([dij_times, bell_times], label=['Dijkstra', 'Bellman-Ford'])
    plt.legend(loc='upper right')
    plt.title('Distribution of computation time: Dijkstra vs Bellman-Ford')
    plt.xlabel('Time to Run [s]')
    plt.ylabel('Occurrence [dimless]')
    plt.show()


if __name__ == "__main__":

    dij_times, bell_times = [], []
    max_val, visit_dist = redefine_variables()
    
    total_vertices = 7
    total_edges = 20
    

    '''
    create a u,v,w graph where u is the starting vertex, v is the ending
    vertex, and w is the weight of the edge between them
    '''
    
    '''
    graph = [
	    [0,1,10],
	    [1,1,10],[1,2,1],
	    [2,2,10],[2,3,5],
	    [3,0,1]
            ]
    '''
    graph = [
	    [0,1,10],[0,4,10],[0,5,6],[0,6,20],
	    [1,1,10],[1,2,1],[1,4,13],[1,5,20],
	    [2,2,10],[2,3,5],[2,5,4],[2,4,8],[2,6,5],
	    [3,0,1],[3,6,5],[3,4,5],[3,5,20], 
            [4,5,20], [4,6,20],
            [5,6,10]
            ]
      
    #convert graph into 2 matrices of vertex connections and edge weights: 
    connections, edge_weight = graph_to_adjacency(graph, total_vertices, total_edges) 

    for j in range(1000):
        max_val, visit_dist = redefine_variables()
        duration = run_dijkstra(connections, edge_weight, total_vertices)
        dij_times.append(duration)

    for j in range (1000):
        #run the bellman ford algorithm and determine the time it takes to run
        duration = bellman_ford(graph, total_edges, total_vertices, 0)
        bell_times.append(duration)

    plot_times_hist(dij_times, bell_times)
