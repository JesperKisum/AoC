"""
--- Day 1: Sonar Sweep ---
You're minding your own business on a ship at sea when the overboard alarm goes off! You rush to see if you can help. Apparently, one of the Elves tripped and accidentally sent the sleigh keys flying into the ocean!

Before you know it, you're inside a submarine the Elves keep ready for situations like this. It's covered in Christmas lights (because of course it is), and it even has an experimental antenna that should be able to track the keys if you can boost its signal strength high enough; there's a little meter that indicates the antenna's signal strength by displaying 0-50 stars.

Your instincts tell you that in order to save Christmas, you'll need to get all fifty stars by December 25th.

Collect stars by solving puzzles. Two puzzles will be made available on each day in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

As the submarine drops below the surface of the ocean, it automatically performs a sonar sweep of the nearby sea floor. On a small screen, the sonar sweep report (your puzzle input) appears: each line is a measurement of the sea floor depth as the sweep looks further and further away from the submarine.

For example, suppose you had the following report:

199
200
208
210
200
207
240
269
260
263
This report indicates that, scanning outward from the submarine, the sonar sweep found depths of 199, 200, 208, 210, and so on.

The first order of business is to figure out how quickly the depth increases, just so you know what you're dealing with - you never know if the keys will get carried into deeper water by an ocean current or a fish or something.

To do this, count the number of times a depth measurement increases from the previous measurement. (There is no measurement before the first measurement.) In the example above, the changes are as follows:

199 (N/A - no previous measurement)
200 (increased)
208 (increased)
210 (increased)
200 (decreased)
207 (increased)
240 (increased)
269 (increased)
260 (decreased)
263 (increased)
In this example, there are 7 measurements that are larger than the previous measurement.

How many measurements are larger than the previous measurement?
"""


import os 
import numpy as np
import matplotlib.pyplot as plt

dat = np.asarray(np.loadtxt('01.txt'))

dat_dx = np.diff(dat)
np.sum(np.where(dat_dx>0,1,0))


"""
--- Part Two ---
Considering every single measurement isn't as useful as you expected: there's just too much noise in the data.

Instead, consider sums of a three-measurement sliding window. Again considering the above example:

199  A      
200  A B    
208  A B C  
210    B C D
200  E   C D
207  E F   D
240  E F G  
269    F G H
260      G H
263        H
Start by comparing the first and second three-measurement windows. The measurements in the first window are marked A (199, 200, 208); their sum is 199 + 200 + 208 = 607. The second window is marked B (200, 208, 210); its sum is 618. The sum of measurements in the second window is larger than the sum of the first, so this first comparison increased.

Your goal now is to count the number of times the sum of measurements in this sliding window increases from the previous sum. So, compare A with B, then compare B with C, then C with D, and so on. Stop when there aren't enough measurements left to create a new three-measurement sum.

In the above example, the sum of each three-measurement window is as follows:

A: 607 (N/A - no previous sum)
B: 618 (increased)
C: 618 (no change)
D: 617 (decreased)
E: 647 (increased)
F: 716 (increased)
G: 769 (increased)
H: 792 (increased)
In this example, there are 5 sums that are larger than the previous sum.

Consider sums of a three-measurement sliding window. How many sums are larger than the previous sum?

"""


# part 2
dat_3 = np.zeros_like(dat)
for i,_ in  enumerate(dat):
    dat_3[i] = np.sum(dat[i:i+3])
dat_3_dx = np.diff(dat_3[0:-2])

np.sum(np.where(dat_3_dx>0,1,0))


#oneliners 
part_1 = np.sum(np.where(np.diff(np.asarray(np.loadtxt('01.txt')))>0,1,0))

part_2 = np.sum(np.where(np.diff([np.sum(np.asarray(np.loadtxt('01.txt'))[i:i+3]) for i , _ in enumerate(np.asarray(np.loadtxt('01.txt'))) ][0:-2])>0,1,0))

print(part_1,part_2)


dat = np.loadtxt('02.txt',dtype=str)

dat = np.char.split(dat)

f = 0
u = 0
d = 0
for i, val in enumerate(dat):
    if dat[i][0] == ['forward']:
        f = f + int(dat[i][1][0])
    elif dat[i][0] == ['up']:
        u = u + int(dat[i][1][0])
    elif dat[i][0] == ['down']:
        d = d + int(dat[i][1][0])

f * (d-u)


#part 2
dat = np.loadtxt('02.txt',dtype=str)

dat = np.char.split(dat)

aim = 0
hoz = 0
depth = 0
for i, val in enumerate(dat):
    if dat[i][0] == ['forward']:
        hoz = hoz + int(dat[i][1][0])
        depth = depth + aim * int(dat[i][1][0])
    elif dat[i][0] == ['up']:
        aim = aim - int(dat[i][1][0])
    elif dat[i][0] == ['down']:
        aim = aim + int(dat[i][1][0])

hoz * depth


import os 
import numpy as np
import matplotlib.pyplot as plt

dat = np.asarray(np.loadtxt('03.txt'))
dat = [str(int(i)) for i in dat]

for idx , val in enumerate(dat):
    if len(val)<14:
        for n in range(13-len(val)):
            dat[idx] = '0' + dat[idx]

dat_T = np.zeros(13, dtype=int)
print(range(len(dat)))

for i in range(len(dat_T)):
    for n in range(len(dat)):
        if dat[n][i]=='1':
            dat_T[i] = dat_T[i] + 1

    
print(dat_T[1:])  


dat_res = np.zeros_like(dat_T)

dat_res = [1 if dat_T[i]-(len(dat)/2)>0 else 0 for i in range(len(dat_T))  ]
print(dat_res[1:])


lst = dat_res[1:]
res = int("".join(str(x) for x in lst), 2)
print(res)

lst_inv = [0 if i==1 else 1 for i in lst ]
res_inv = int("".join(str(x) for x in lst_inv), 2)
print(res_inv)
print(res_inv*res)


# part 2 
oxy = [dat[i][1:] for i in range(len(dat))]

for i in range(len(oxy[0])):
    if len(oxy)==1:
        break
    
    oxy_T = np.zeros(12, dtype=int)
    
    for ii in range(len(oxy_T)):
        for n in range(len(oxy)):
            if oxy[n][ii]=='1':
                oxy_T[ii] = oxy_T[ii] + 1

    oxy_res = np.zeros_like(oxy_T)
    oxy_res = [1 if oxy_T[ii]-(len(oxy)/2)>=0 else 0 for ii in range(len(oxy_T))  ]
    #print(oxy_res)
    for n in range(len(oxy)):
        if int(oxy[n][i])get_ipython().getoutput("=oxy_res[i]:")
            oxy[n]='0'
    oxy = [x for x in oxy if x get_ipython().getoutput("= '0']")
    print(len(oxy))


                


co2 = [dat[i][1:] for i in range(len(dat))]

for i in range(len(co2[0])):
    if len(co2)==1:
        break
    
    co2_T = np.zeros(12, dtype=int)
    
    for ii in range(len(co2_T)):
        for n in range(len(co2)):
            if co2[n][ii]=='1':
                co2_T[ii] = co2_T[ii] + 1

    co2_res = np.zeros_like(co2_T)
    co2_res = [1 if co2_T[ii]-(len(co2)/2)>=0 else 0 for ii in range(len(co2_T))  ]
    
    for n in range(len(co2)):
        if int(co2[n][i])==co2_res[i]:
            co2[n]='0'
    co2 = [x for x in co2 if x get_ipython().getoutput("= '0']")
    print(len(co2))




print(co2,oxy)
int(co2[0],2)*int(oxy[0],2)








import os 
import numpy as np
import matplotlib.pyplot as plt

with open('06.txt') as f:
    lines = f.readlines()
    
    
dat = np.asarray(lines[0][:].split(','))
dat = np.asarray([int(i) for i in dat])


def lantern(arr):
    for i,_ in enumerate(arr):
        arr[i]=arr[i]-1
    idx = np.where(arr==-1)
    arr[idx]=6
    
    if len(idx[0])>0:
        for n in range(0,len(idx[0])):
            arr = np.append(arr,8)
    return arr

def days(arr,n):
    for i in range(0,n):
        arr = lantern(arr)
        
    return arr
res = days(dat,1)


#actual solution to the problem
with open('06.txt') as f:
    dat = f.readlines()
dat = dat[:][0]

from collections import deque

def count_fish(data, days):
    life = [0]*9
    for i in data.split(','): life[int(i)] += 1
    fish = deque(life)
    
    for i in range(days):
        spawn = fish.popleft()
        fish[-2] += spawn
        fish.append(spawn)
    return sum(fish)

print('Part 1', count_fish(dat, 80))
print('Part 2', count_fish(dat, 256))

























































