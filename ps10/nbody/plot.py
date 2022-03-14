#                                                                                                                                   
#  This file is part of the course materials for AMATH483/583 at the University of Washington,                                      
#  Spring 2017                                                                                                                      
#                                                                                                                                   
#  Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License                                   
#  https://creativecommons.org/licenses/by-nc-sa/4.0/                                                                               
#                                                                                                                                   
#  Author: Andrew Lumsdaine                                                                                                         
#                                                                                                                                   

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import csv
from itertools import takewhile

import csv
import sys
import os
import io
import re


plt.figure(figsize=(9,4))

names = []

for name in sys.argv[1:]:

    infile = name

    names.append(os.path.basename(name).replace(".out.txt",""))

    legs = []
    with open(infile, 'r') as csvfile:
        bulk = csvfile.read()
        blocks = re.compile(r"\n{2,}").split(bulk)

        for b in blocks:
            title = b[0:4]
            legs.append(title);

            Ns = []
            Ps = []
            Ts = []
            Ss = []

            reader = csv.DictReader(io.StringIO(b[5:]), delimiter='\t')

            for row in reader:
                Ns.append(int(row['procs']))
                Ts.append(float(row['time']))
                Ss.append(Ts[0]/float(row['time']))


            plt.loglog(Ns, Ss)

pp = PdfPages('time.pdf')

#plt.axis([0, 2048, .01, 24])                                                                                                       

plt.legend(names, loc='upper left', fontsize='small')
plt.title('N-Body Simulation')
plt.xlabel('Processes')
plt.ylabel('Speedup')
plt.xticks([1, 2, 4, 8, 16, 32],[1, 2, 4, 8, 16, 32])

pp.savefig()
pp.close()
# plt.show()                                                                                                                       
