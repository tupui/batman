import csv
import numpy as np

with open('My_BC.csv', newline = '' ) as csvfile:
    myreader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
    vect_time = np.zeros(10)
    mat_BC = np.zeros((10, 2))
    i = 0
    for row in myreader:
        print (row[0], row[1], row[2])
        vect_time[i] = row[0]
        mat_BC[i,0] = row[1]
        mat_BC[i,1] = row[2]
        i = i+1

print (vect_time)
print (mat_BC) 
