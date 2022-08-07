import numpy as np
import random
import math


class MyMap:
    def __init__(self, start, end):
        self.matrix = np.array([['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
                                ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
                                ['#', '.', '#', '#', '.', '.', '#', '.', '.', '#'],
                                ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
                                ['#', '.', '.', '.', '#', '#', '.', '.', '.', '#'],
                                ['#', '.', '.', '.', '#', '#', '.', '.', '.', '#'],
                                ['#', '.', '#', '.', '.', '.', '.', '#', '.', '#'],
                                ['#', '.', '#', '.', '#', '#', '.', '#', '.', '#'],
                                ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
                                ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']])
        self.edge = 8
        self.start = start
        self.end = end
        if self.matrix[self.start[0], self.start[1]] == '.':
            self.matrix[self.start[0], self.start[1]] = 's'
        if self.matrix[self.end[0], self.end[1]] == '.':
            self.matrix[self.end[0], self.end[1]] = 'e'


class Genome:
    def __init__(self, len):
        self.bits = []
        self.fitness = 0
        self.stop = 0
        for g in range((int)(len / 2)):
            self.bits.append(random.randint(0, 1))
            self.bits.append(random.randint(0, 1))


def build_genList(num):
    genList = []
    for i in range(2 * num):
        gen = Genome(num)
        genList.append(gen)

    return genList


def detect(map, gen, start, end):
    position = []
    position.append(start[0])
    position.append(start[1])
    for j in range(int(len(gen.bits) / 2)):
        g = [gen.bits[j * 2], gen.bits[j * 2 + 1]]
        if g[0] == 0 and g[1] == 0:
            position[1] += 1
        elif g[0] == 0 and g[1] == 1:
            position[0] -= 1
        elif g[0] == 1 and g[1] == 0:
            position[1] -= 1
        else:
            position[0] += 1
        if map.matrix[position[0], position[1]] == '#':
            gen.fitness = 1 / (abs(end[0] - position[0]) + abs(end[1] - position[1]) + 1)
            gen.stop = j
            break
        elif position[0] == end[0] and position[1] == end[1]:
            gen.fitness = 1
            gen.stop = j
            return 1
    # if gen.stop == 0:
    #     gen.fitness = 1 / (abs(end[0] - position[0]) + abs(end[1] - position[1]) + 1)
    #     gen.stop = int(len(gen.bits) / 2)
    return 0


def select(genList):
    total_fit = 0
    for g in genList:
        total_fit += g.fitness
    piece = random.random() * total_fit
    temp = 0
    for g in genList:
        temp = temp + g.fitness
        if temp >= piece:
            return g


def crossover(dad, mum, rate):
    child1 = []
    child2 = []
    rn = random.random()
    if rn < rate:
        pt = random.randint(0, len(dad.bits))
        for i in range(pt):
            child1.append(dad.bits[i])
            child2.append(mum.bits[i])
        for i in range(pt, len(dad.bits)):
            child1.append(mum.bits[i])
            child2.append(dad.bits[i])
    else:
        child1 = dad.bits
        child2 = mum.bits
    return child1, child2


def mutate(gen, rate):
    for i in range(0, len(gen.bits)):
        if random.random() < rate:
            if gen.bits[i] == 1:
                gen.bits[i] = 0
            else:
                gen.bits[i] = 1


def showPath(map, bits, start, end):
    step = start
    map.matrix[start[0], start[1]] = 's'
    s = 0
    while not (step[0] == end[0] and step[1] == end[1]):
        if [bits[s], bits[s + 1]] == [0, 0]:
            step[1] = step[1] + 1
            map.matrix[step[0], step[1]] = '@'
        elif [bits[s], bits[s + 1]] == [0, 1]:
            step[0] = step[0] - 1
            map.matrix[step[0], step[1]] = '@'
        elif [bits[s], bits[s + 1]] == [1, 0]:
            step[1] = step[1] - 1
            map.matrix[step[0], step[1]] = '@'
        elif [bits[s], bits[s + 1]] == [1, 1]:
            step[0] = step[0] + 1
            map.matrix[step[0], step[1]] = '@'
        s = s + 2
    map.matrix[end[0], end[1]] = 'e'
    print(map.matrix)


if __name__ == '__main__':
    start = [6, 1]
    end = [2, 8]
    map = MyMap(start, end)
    gn = 30
    print(map.matrix)
    genList = build_genList(gn)
    flag = 0
    for g in genList:
        flag = detect(map, g, start, end)
        if flag == 1:
            print("Find It!")
            print(g.stop)
            print(g.bits)
            exit(0)
    newlist = []
    child_n = 0
    time = 0
    while flag == 0:
        time += 1
        child_n = 0
        while child_n < gn * 2:
            dad = select(genList)
            mum = select(genList)

            child1 = Genome(gn)
            child2 = Genome(gn)

            child1.bits, child2.bits = crossover(dad, mum, 0.5)
            detect(map, child1, start, end)
            detect(map, child2, start, end)

            mutate(child1, 0.3)
            mutate(child2, 0.3)

            flag = detect(map, child1, start, end)
            if flag == 1:
                print("Find It!")
                print(f'number of iteration is {time}')
                print(f'effective chromosome length {child1.stop + 1}')
                showPath(map, child1.bits, start, end)
                break

            flag = detect(map, child2, start, end)
            if flag == 1:
                print("Find It!")
                print(f'number of iteration is {time}')
                print(f'effective chromosome length {child2.stop + 1}')
                showPath(map, child2.bits, start, end)
                break

            newlist.append(child1)
            newlist.append(child2)

            child_n += 2
        genList = newlist
        newlist = []
