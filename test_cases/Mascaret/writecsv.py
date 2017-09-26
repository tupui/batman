import csv
import numpy as np

vect_time = np.array([0.,1800., 3600., 5400., 7200., 9000., 10800., 12600., 14400., 16200.])
BC_1 =  np.array([1000., 1000., 1000., 2000., 2000., 2000., 1000., 1000., 1000., 1000.])
BC_2 =  np.array([12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5])
with open('My_BC.csv', 'w', newline = '' ) as csvfile:
    mywriter = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
#    mywriter.writerow(["Time(s)", "BC_1(m3/s)", "BC_2(m)"])
    for i in range(10):
        mywriter.writerow([vect_time[i]] + [BC_1[i]] + [BC_2[i]] )


