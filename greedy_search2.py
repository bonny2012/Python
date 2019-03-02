import numpy as np
import time

import matplotlib.pyplot as plt

inf = float('inf')

# read file from certain directory
def read_file(filepath):
    with open(filepath, 'r') as f:
        x = f.readlines()  # read file line by lie

    city = x[7:]  # generate array form line 8 of the file

    city_array = []
    index = 0

    for index in range(0, len(city)):  # split each line with space
        firstWord, secondWord, thirdWord, *otherWords = city[index].split()
        # print(firstWord)
        # print(secondWord)
        # print(thirdWord)
        # print(index)
        city_array.append([])
        # city_array[index].append(float(firstWord))
        city_array[index].append(float(secondWord))
        city_array[index].append(float(thirdWord))

    cities = np.asarray(city_array)  # generate array with value of coordinates

    return cities


cities = read_file("/Users/bonnyxin/PycharmProjects/project3/Random40.tsp")
print(cities)


# fomular of cacularing distance of two cities
def calDist(index1, index2):
    return (np.sum(np.power(cities[index1] - cities[index2], 2))) ** 0.5


# now need insert inf to array a

def gen_weight_array(cities):

    array= [[inf for i in range(len(cities))] for j in range(len(cities))]

    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            array[i][j] = calDist(i, j)

    return array


graph_weight = gen_weight_array(cities)

for i in range(0, len(graph_weight)):
   print(graph_weight[i])

# fomular of cacularing distance of two cities
def calDist(index1, index2):
    return (np.sum(np.power(cities[index1] - cities[index2], 2))) ** 0.5

# now need insert inf to array a

def gen_weight_array(array):
    for i in range(0,len(array)):
        for j in range(0,len(array[i])):
            if array[i][j] != inf:
                array[i][j] = calDist(i, j)
            else:
                continue

    return array


#ancestor_dic = {}  # a dictionary for node ancestor
distance_dic = {}  # a dictionary for node current weight
visited = [] # create an empty list for visited nodes

color_dic = {} # a color dictionary for the node color

def distEdge_node(edgeNode1,edgeNode2,node): #calculate distance between an edge to a node
    x1 = cities[edgeNode1][0]
    x2 = cities[edgeNode2][0]
    x0 = cities[node][0]
    y1 = cities[edgeNode1][1]
    y2 = cities[edgeNode2][1]
    y0 = cities[node][1]
    d1 = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    d2 = ((x2-x1)**2+(y2-y1)**2)**0.5
    d = d1/d2
    return d



def greedySearch(graph,start):
    vetex = []
    vetex.extend(range(len(graph)))  # create node list

    distance = 0
    for i in range(start, len(graph)):
        color_dic[i] = 'WHITE'

    #start from start node
    global visited
    u = start -1
    vetex.remove(u)
    color_dic[u]= 'BLACK'
    print("this is vetex: " + str(vetex))
    temp_short = inf
    visited.append(u)

    # add the second node
    for current_neighbour in graph[u]:
        if current_neighbour != 0 and color_dic[graph[u].index(current_neighbour)] == 'WHITE' \
                and current_neighbour < temp_short:
            temp_short = current_neighbour
            temp = graph[u].index(current_neighbour)
            print("this is temp short: ", temp_short)
            print("This is temp: ", temp)

    distance += temp_short
    temp_short = inf
    visited.append(temp)
    print("I am distance : ", distance)
    print("I am visited: ", visited)
    print(" I am i: ", temp)
    print("This is vetex:", vetex)
    vetex.remove(temp)
    print("vetex after remove temp:", vetex)
    color_dic[u] = 'BLACK'
    u = temp

    distance += graph[visited[-1]][0]

    #start traverse the third node till end,compare the ditance between edge and node
    while vetex:
        for current_neighbour in graph[u]:
            if current_neighbour != 0 and color_dic[graph[u].index(current_neighbour)] == 'WHITE':
                d = distEdge_node(u,visited[-2],graph[u].index(current_neighbour))
                if d < temp_short:
                    temp_short = d
                    nextNode = graph[u].index(current_neighbour)
                    short_dis = current_neighbour
                    print("this is d: ", temp_short)
                    print("I am temp dis: ",short_dis)
                    print("This is nextNode: ", nextNode)

        distance += short_dis
        temp_short = inf
        visited.append(nextNode)
        print("I am distance : ", distance)
        print("I am visited: ", visited)
        print(" I am i: ", nextNode)
        print("This is vetex:", vetex)
        vetex.remove(nextNode)
        print("vetex after remove temp:",vetex)
        color_dic[u] = 'BLACK'
        u = nextNode

    distance+=graph[visited[-1]][0]
    visited.append(start-1)
    visited = np.array(visited)+1

    return visited,distance


def draw(finalPath):

    axis = plt.subplot(111, aspect='equal')
    axis.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    for i,city in enumerate(cities):
        axis.text(city[0], city[1], str(i+1))
    finalPath = np.array(finalPath)-1
    axis.plot(cities[finalPath, 0], cities[finalPath, 1], color='red')
    plt.suptitle('Route of Greedy search 1', fontsize=16)
    plt.show()



path, dis = greedySearch(graph_weight, 1)

print("Distance is : " + str(dis))
print("Path is : " + str(path))
print("\nRunning time is: " + str(time.process_time()) + " seconds." )
draw(path)







