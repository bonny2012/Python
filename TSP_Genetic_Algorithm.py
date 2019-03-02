#####Genetic Algorithm for AI project 4, 10072018 ######
#####Xin Zheng#######


import numpy as np
import time
import matplotlib.pyplot as plt
import random


inf = float('inf')
# read file from certain directory
def read_file(filepath):
    with open(filepath, 'r') as f:
        x = f.readlines()  # read file line by line

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


cities = read_file("/Users/bonnyxin/PycharmProjects/project4_GA/Random100.tsp")
#print(cities)


# fomular of cacularing distance of two cities
def calDist(index1, index2):
    return (np.sum(np.power(cities[index1] - cities[index2], 2))) ** 0.5



# function to generation population with 20 individuals
def generate_population(cityNumber):
    # generate 30 permutations for 100 cities
    i = 0
    temp_list = list(range(0,len(cityNumber)))
    #print(temp_list)
    for i in range(0,max_city_amount):
        random.shuffle(temp_list)
        #print("Reshuffled list : ", i, temp_list)

        population.append(temp_list[:])
        #population.append(np.random.permutation(cityNumber))
    #print('This is poputation : ', population)

    return




def individual_distance(indexList):
    sum = 0.0
    for j in range(1, len(indexList)):
        sum += calDist(indexList[j], indexList[j - 1])
    return sum

# caclulate the individual fitting rate and total fitting rate
def gen_individual_fitting_rate(indexList):
    sum = 0.0
    ind_fitting_rate = []
    total_fitting_rate = 0.0
    min_ind = {}
    min_ind["mylist"] = []
    min_ind["distance"] = inf
    for i in range(0, len(indexList)):
        sum = individual_distance(indexList[i])

        if min_ind['distance'] > sum:
            min_ind['mylist'] = list(indexList[i])
            min_ind['distance'] = sum

        #print("this is sum: ", sum)
        total_fitting_rate += 1/sum
        ind_fitting_rate.append(1/sum)
        sum =0.0
        #print('This is individual fitting rate: ',i, "here",ind_fitting_rate[i])

    #print("total fitting rate is : ", total_fitting_rate)
    return total_fitting_rate,min_ind,ind_fitting_rate

#generate new population in which indidual fitting rate of all individual are greater than natural select rate
def selection_fitted_individuals(population,individual_fitting_rate):
    temp_popu = []
    for i in range(0,len(individual_fitting_rate)):
        if individual_fitting_rate[i] > natural_select_rate:
            temp_popu.append(population[i])

    return temp_popu


#select two random individuals for genrating children

def select_two_individuals(temp_population):

    index = random.sample(range(0, len(temp_population)), 2)
    if index[0]!= index[1]:
        ind1 = list(temp_population[index[0]])
        ind2 = list(temp_population[index[1]])
        #print('ind1:', ind1)
        #print('ind2:', ind2)
    return ind1,ind2


#check if two individuals same

def check_two_individuals(ind1,ind2):
    for i in range(0,len(ind1)):
        if ind1[i]!= ind2[i]:
            #print("They are not the same")
            return True
    return False


# final assign two individuals
def final_check_individuals(temp_population):
    ind1,ind2 = select_two_individuals(temp_population)

    if check_two_individuals(ind1, ind2):
        return ind1, ind2
    else:
        ind1,ind2 = select_two_individuals(temp_population)
        return ind1, ind2


def crossover(ind1,ind2,rate):
    #print('ind1 before cross over:', ind1)
    #print('ind2 before cross over:', ind2)
    inde1 = random.randint(0, len(cities)-rate)
    inde2 = inde1 + rate
    #print('index1 for crossover:', inde1)
    #print('index1 for crossover:', inde2)
    segment1 = list(ind1[inde1:inde2])
    segment2 = list(ind2[inde1:inde2])
    #print('segment1 : ',segment1)
    #print('segment2 : ',segment2)

    #find out same element in the two segement
    same_element = list(set(segment1).intersection(segment2))
    #print('Same segment : ', same_element)

    #remove same elements in two segment
    seg1 = [elem for elem in segment1 if elem not in same_element ]
    seg2 = [elem for elem in segment2 if elem not in same_element ]
    #print('seg1 : ', seg1)
    #print('seg2 : ', seg2)

    #print('ind1 s length is : ', len(ind1))
    #print('ind2 s length is : ', len(ind2))
    #print( 'ind1 is :',ind1)
    #print('ind2 is :', ind2)

    mapindex1 = [list(ind1).index(i) for i in seg2]

    mapindex2 = [list(ind2).index(j) for j in seg1]

    #print('Mapindex1: ',mapindex1)
    #print('mapindex1 length is :', len(mapindex1))
    #print("mapindex2: ", mapindex2)
    #print('mapindex2 length is :', len(mapindex2))
    #crossover two individuals

    ind1[inde1:inde2] = list(segment2)
    ind2[inde1:inde2] = list(segment1)
    #print("ind1 after crossover:", ind1)
    #print("ind2 after crossover:", ind2)

    #temp1 =[]
    #temp2 =[]
    #for i in range(0,len(mapindex1)):
    #    temp1.append(mapindex1[i])
     #   temp2.append(mapindex2[i])
    #print()
    for i in range(0, len(seg1)):
        ind1[mapindex1[i]] = seg1[i]
        ind2[mapindex2[i]] = seg2[i]

    #print("ind1 after remove dup:", ind1)
    #print("ind2 after remove dup:", ind2)

    return ind1,ind2

#crossover with rate 30
def crossover1(ind1,ind2):
    crossover_rate = 30
    ind1,ind2 = crossover(ind1,ind2,crossover_rate)
    return ind1,ind2

# crossover with rate 10
def crossover2(ind1, ind2):
    crossover_rate = 10
    ind1,ind2 = crossover(ind1,ind2,crossover_rate)
    return ind1,ind2



# reverse 3 numbers in individual
def mutation1(ind1,ind2):
    reverse_rate = 3
    inde1 = random.randint(0, len(cities) - reverse_rate)
    #print('index for mutation1 is : ', inde1)

    inde2 = inde1+ reverse_rate
    #print("original ind1:", ind1)
    #print("original ind2:", ind2)

    ind1[inde1:inde2] = ind1[inde1:inde2][::-1]
    ind2[inde1:inde2] = ind2[inde1:inde2][::-1]
    #print("new ind1:", ind1)
    #print("new ind2:", ind2)

    return ind1,ind2

#random select 4 numbers to mutate
def mutation2(ind1,ind2):
    mutation_rate = 4
    index = []
    index = random.sample(range(0, len(cities)), mutation_rate)
    #print(index)


    ind1[index[0]],ind1[index[1]] = ind1[index[1]],ind1[index[0]]
    ind1[index[2]],ind1[index[3]] = ind1[index[3]],ind1[index[2]]
    ind2[index[0]],ind2[index[1]] = ind2[index[1]],ind2[index[0]]
    ind2[index[2]],ind2[index[3]] = ind2[index[3]],ind2[index[2]]
    #print("new ind1:", ind1)
    #print("new ind2:", ind2)

    return ind1,ind2


# compare children's distance and his parents distance, parameter is from parents
# dictionary
def compare_fitting_rate(indexList1,distance1, distance2):
    sum = individual_distance(indexList1)
    if sum < distance1 and sum< distance2:

        #print('This children is a good kid.',sum)
        return True
    else:
        #print('this children is evil and naughty!')
        return False




population = [] # a list for individual
max_iteration = 80 # limit iteration number
iteration = 0 # initial interative number

max_city_amount = 50 # number of individual in population
selectRate = 1/max_city_amount # select rate for total fitting rate.

generate_population(cities) # generate a population

#individual_fitting_rate = [] # a list for individual fitting rate
#best_individual = {}# keep best individual

total_fitting_rate,best_individual,individual_fitting_rate = gen_individual_fitting_rate(population) #generate total fitting rate
natural_select_rate = total_fitting_rate*selectRate

# all selected parents
temp_population = selection_fitted_individuals(population,individual_fitting_rate)

#just for checking the temp population list
#for i in range(0,len(temp_population)):
    #print("This is the ", i, temp_population[i])



dataset1a = [] # keep children generated by crossover1 &mutation1
dataset1b = [] # keep children generated by crossover1 &mutation2
dataset2a = [] # keep children generated by crossover2 &mutation1
dataset2b = [] # keep children generated by crossover2 &mutation2

#generate dataset1a

def gen_dataset1a(population_list):


    while len(dataset1a)< max_city_amount:
        individual1, individual2 = final_check_individuals(population_list)

        #print("this is parent1:", individual1)
        #print("this is parent2:", individual2)
        parents1a = {}  # keep parent1, parent2 and the correspond distance
        parents1a['parent1'] = individual1
        parents1a['parent2'] = individual2
        parents1a['parent1_distance'] = individual_distance(individual1)
        parents1a['parent2_distance'] = individual_distance(individual2)
        children1a_1,children1a_2 = list(individual1), list(individual2)
        #children1a_1, children1a_2 = crossover1(children1a_1, children1a_2)
        #children1a_1, children1a_2 = mutation1(children1a_1, children1a_2)
        crossover1(children1a_1, children1a_2)
        mutation1(children1a_1, children1a_2)
        #print('this is children1a_1: ', children1a_1)

        #print('this is children1a_2: ', children1a_2)

        #print('I am best individual : ', best_individual)
        if compare_fitting_rate(children1a_1, parents1a['parent1_distance'], parents1a['parent2_distance']) and len(dataset1a) <max_city_amount:
            #print("children1a_1 is append:",children1a_1)
            dataset1a.append(children1a_1)
        if compare_fitting_rate(children1a_2, parents1a['parent1_distance'], parents1a['parent2_distance']) and len(dataset1a) <max_city_amount:
            #print("children1a_1 is append:", children1a_2)
            dataset1a.append(children1a_2)


        continue

    return

#generate dataset1b in process

def gen_dataset1b(population_list):

    while len(dataset1b)< max_city_amount:
        individual1, individual2 = final_check_individuals(population_list)

        #print("this is ind1 and 2", individual1, individual2)
        parents1b = {}  # keep parent1, parent2 and the correspond distance
        parents1b['parent1'] = individual1
        parents1b['parent2'] = individual2
        parents1b['parent1_distance'] = individual_distance(individual1)
        parents1b['parent2_distance'] = individual_distance(individual2)
        children1b_1,children1b_2 = list(individual1), list(individual2)
        crossover1(children1b_1, children1b_2)
        mutation2(children1b_1, children1b_2)

        #print('this is ind1: ', children1b_1)

        #print('this is ind2: ', children1b_2)

        #print('I am best individual : ', best_individual)
        if compare_fitting_rate(children1b_1, parents1b['parent1_distance'], parents1b['parent2_distance']) and len(dataset1b) <max_city_amount:
            dataset1b.append(children1b_1)
        if compare_fitting_rate(children1b_2, parents1b['parent1_distance'], parents1b['parent2_distance']) and len(dataset1b) <max_city_amount:
            dataset1b.append(children1b_2)

        continue

    return


#generate dataset2a in process

def gen_dataset2a(population_list):

    while len(dataset2a)< max_city_amount:
        individual1, individual2 = final_check_individuals(population_list)

        #print("this is ind1 and 2", individual1, individual2)
        parents2a = {}  # keep parent1, parent2 and the correspond distance
        parents2a['parent1'] = list(individual1)
        parents2a['parent2'] = list(individual2)
        parents2a['parent1_distance'] = individual_distance(individual1)
        parents2a['parent2_distance'] = individual_distance(individual2)
        children2a_1,children2a_2 = list(individual1), list(individual2)
        crossover2(children2a_1, children2a_2)
        mutation1(children2a_1, children2a_2)

        #print('this is ind1: ', children2a_1)

        #print('this is ind2: ', children2a_2)

        #print('I am best individual : ', best_individual)
        if compare_fitting_rate(children2a_1, parents2a['parent1_distance'], parents2a['parent2_distance']) and len(dataset2a) <max_city_amount:
            dataset2a.append(children2a_1)
        if compare_fitting_rate(children2a_2, parents2a['parent1_distance'], parents2a['parent2_distance']) and len(dataset2a) <max_city_amount:
            dataset2a.append(children2a_2)

        continue

    return
#generate dataset2b
def gen_dataset2b(population_list):

    while len(dataset2b)< max_city_amount:
        individual1, individual2 = final_check_individuals(population_list)

        #print("this is ind1 and 2", individual1, individual2)
        parents2b = {}  # keep parent1, parent2 and the correspond distance
        parents2b['parent1'] = list(individual1)
        parents2b['parent2'] = list(individual2)
        parents2b['parent1_distance'] = individual_distance(individual1)
        parents2b['parent2_distance'] = individual_distance(individual2)
        children2b_1,children2b_2 = list(individual1), list(individual2)
        crossover1(children2b_1, children2b_2)
        mutation2(children2b_1, children2b_2)

        #print('this is ind1: ', children2b_1)

        #print('this is ind2: ', children2b_2)

        #print('I am best individual : ', best_individual)
        if compare_fitting_rate(children2b_1, parents2b['parent1_distance'], parents2b['parent2_distance'])and len(dataset2b) <max_city_amount:
            dataset2b.append(children2b_1)
        if compare_fitting_rate(children2b_2, parents2b['parent1_distance'], parents2b['parent2_distance'])and len(dataset2b) <max_city_amount:
            dataset2b.append(children2b_2)

        continue

    return

# draw graph for one path
def draw(finalPath):
    axis = plt.subplot(111, aspect='equal')
    axis.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    for i, city in enumerate(cities):
        axis.text(city[0], city[1], str(i + 1))
        #axis.text(city[0], city[1], str(i)) # remove i +1
    #finalPath = np.array(finalPath) - 1
    finalPath = np.array(finalPath) #remove -1, because the path here is not with +1 in the function
    axis.plot(cities[finalPath, 0], cities[finalPath, 1], color='red')
    plt.suptitle('Route of GA search', fontsize=16)
    plt.show()
    plt.gcf().clear()


# draw paths of four datasets
def drawfull(finalPath1a,finalPath1b,finalPath2a,finalPath2b):
    axis1 = plt.subplot(221, aspect='equal')
    axis2 = plt.subplot(222, aspect='equal')
    axis3 = plt.subplot(223, aspect='equal')
    axis4 = plt.subplot(224, aspect='equal')
    axis1.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    axis2.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    axis3.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    axis4.plot(cities[:, 0], cities[:, 1], 'o', color='blue')
    finalPath1a = np.array(finalPath1a)
    finalPath1b = np.array(finalPath1b)
    finalPath2a = np.array(finalPath2a)
    finalPath2b = np.array(finalPath2b)
    for i,city in enumerate(cities):
        axis1.text(city[0], city[1], str(i+1))
    for i,city in enumerate(cities):
        axis2.text(city[0], city[1], str(i+1))
    for i,city in enumerate(cities):
        axis3.text(city[0], city[1], str(i+1))
    for i,city in enumerate(cities):
        axis4.text(city[0], city[1], str(i+1))
    plt.suptitle('Route of GA search,1a,1b,2a,2b', fontsize=16)
    axis1.plot(cities[finalPath1a, 0], cities[finalPath1a, 1], color='red')
    axis2.plot(cities[finalPath1b, 0], cities[finalPath1b, 1], color='green')
    axis3.plot(cities[finalPath2a, 0], cities[finalPath2a, 1], color='yellow')
    axis4.plot(cities[finalPath2b, 0], cities[finalPath2b, 1], color='purple')
    plt.show()
    plt.gcf().clear()



gen_dataset1a(temp_population)
gen_dataset1b(temp_population)
gen_dataset2a(temp_population)
gen_dataset2b(temp_population)


best_ind1a = {}
best_ind1a['sum'] = inf
best_ind1a['list'] = []
best_ind1b = {}
best_ind1b['sum'] = inf
best_ind1b['list'] = []
best_ind2a = {}
best_ind2a['sum'] = inf
best_ind2a['list'] = []
best_ind2b = {}
best_ind2b['sum'] = inf
best_ind2b['list'] = []
for i in range(0, max_iteration):

    #interative dataset1a
    print('this is i:', i)

    #best_individual1a = []
    total_fitting_rate1a, temp_best_individual1a, individual_fitting_rate1a = gen_individual_fitting_rate(
        dataset1a)
    natural_select_rate1a = total_fitting_rate1a * selectRate
    #best_individual1a.append(temp_best_individual1a['mylist'])
    #print('best individual 1a :', best_individual1a)
    temp_population1a = selection_fitted_individuals(dataset1a, individual_fitting_rate1a)
    del dataset1a[:]
    gen_dataset1a(temp_population1a)


    if best_ind1a['sum'] > temp_best_individual1a['distance']:
        best_ind1a['sum'] = temp_best_individual1a['distance']
        best_ind1a['list'] = temp_best_individual1a['mylist']
    total_fitting_rate1a = 0.0
    #draw(best_ind1a['list']) to see dynamic graph for dataset 1a, remove '#'
    del individual_fitting_rate1a[:]
    del temp_population1a[:]
    temp_best_individual1a.clear
    print("best individual's distance for dataset1a is :", best_ind1a)



    #interative dataset1b
    #print('this is i:', i)
    #best_individual1b = []
    total_fitting_rate1b, temp_best_individual1b, individual_fitting_rate1b = gen_individual_fitting_rate(
        dataset1b)
    natural_select_rate1b = total_fitting_rate1b * selectRate
    #best_individual1b = list(temp_best_individual1b['mylist'])
    #print('best individual 1b :', best_individual1b)
    temp_population1b = selection_fitted_individuals(dataset1b, individual_fitting_rate1b)
    del dataset1b[:]
    gen_dataset1b(temp_population1b)


    if best_ind1b['sum'] > temp_best_individual1b['distance']:
        best_ind1b['sum'] = temp_best_individual1b['distance']
        best_ind1b['list'] = temp_best_individual1b['mylist']
    total_fitting_rate1b = 0.0
    #draw(best_ind1b['list']) to see dynamic graph for dataset 1b, remove '#'
    del individual_fitting_rate1b[:]
    del temp_population1b[:]
    temp_best_individual1b.clear
    print("best individual's distance for dataset1b is :", best_ind1b)

    #interative dataset2a
    #print('this is i:', i)
    #best_individual2a = []
    total_fitting_rate2a, temp_best_individual2a, individual_fitting_rate2a = gen_individual_fitting_rate(
        dataset2a)
    natural_select_rate2a = total_fitting_rate2a * selectRate
    #best_individual2a = list(temp_best_individual2a['mylist'])
    #print('best individual 2a :', best_individual2a)
    temp_population2a = selection_fitted_individuals(dataset2a, individual_fitting_rate2a)
    del dataset2a[:]
    gen_dataset2a(temp_population2a)


    if best_ind2a['sum'] > temp_best_individual2a['distance']:
        best_ind2a['sum'] = temp_best_individual2a['distance']
        best_ind2a['list'] = temp_best_individual2a['mylist']
    total_fitting_rate2a = 0.0
    #draw(best_ind2a['list']) to see dynamic graph for dataset 2a, remove '#'
    del individual_fitting_rate2a[:]
    del temp_population2a[:]
    temp_best_individual2a.clear
    print("best individual's distance for dataset2a is :", best_ind2a)


    #interative dataset2b
    #print('this is i:', i)
    #best_individual2b = []
    total_fitting_rate2b, temp_best_individual2b, individual_fitting_rate2b = gen_individual_fitting_rate(
        dataset2b)
    natural_select_rate2b = total_fitting_rate2b * selectRate
    #best_individual2b = list(temp_best_individual2b['mylist'])
    #print('best individual 2b :', best_individual2b)
    temp_population2b = selection_fitted_individuals(dataset2b, individual_fitting_rate2b)
    del dataset2b[:]
    gen_dataset2b(temp_population2b)


    if best_ind2b['sum'] > temp_best_individual2b['distance']:
        best_ind2b['sum'] = temp_best_individual2b['distance']
        best_ind2b['list'] = temp_best_individual2b['mylist']
    total_fitting_rate2b = 0.0
    #draw(best_ind2b['list']) to see dynamic graph for dataset 2b, remove '#'
    del individual_fitting_rate2b[:]
    del temp_population2b[:]
    temp_best_individual2b.clear
    print("best individual's distance for dataset2b is :", best_ind2b)



print('I am best ind1a',best_ind1a)
print('I am best ind1b',best_ind1b)
print('I am best ind2a',best_ind2a)
print('I am best ind2b',best_ind2b)

print("\nRunning time is: " + str(time.process_time()) + " seconds." )

drawfull(best_ind1a['list'],best_ind1b['list'],best_ind2a['list'],best_ind2b['list'])





