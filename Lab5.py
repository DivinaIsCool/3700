#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit


# In[1]:


# Define a function to read three columns of data.

def get_data(fname):
    """
    To use this function, type something like:
    
    list1, list2, list3 = get_data(filename) 
    
    Where filename is the name of a file with three columns of data. 
    This function will return these three columns of data as individual python lists.
    """

    # Create two blank lists.
    col1 = []
    col2 = []

    # Make a list of all the lines in the file.  Each entry in the list
    # is one line.
    with open(fname, 'r') as f:
        lines = f.readlines()         # Load all the lines

        # For each line in the file...
        for each_line in lines:
            # Split the lines into words.
            words = each_line.split()
            
            # Turn each word into floats, ignoring the case when
            # a line begins with the comment character '%'.  The
            # test for a comment is when the first character in
            # the first word is '#'.  We also test for the case of
            # blank lines (number of words = 0).
            if len(words) > 0:
                if words[0][0] != '#':        # First character of first word
                    val1 = float(words[0])
                    val2 = float(words[1])
                    col1.append(val1)
                    col2.append(val2)

    # All done
    return col1, col2

csvolt, cscount=get_data('CesiumDataZavier')
covolt, cocount=get_data('CobaltDataZavier')
navolt, nacount=get_data('SodiumDataZavier')


# In[110]:


plt.scatter(csvolt, cscount, label='Cesium Data', zorder=1, s=5)
plt.scatter(2.287, 7810, label='0.622MeV Peak', zorder=2, s=15, color='red')
plt.xlabel('Voltage (V)')
plt.ylabel('Occurances')
plt.legend()


# In[111]:


plt.scatter(covolt, cocount, label='Cobalt Data', color='lime', zorder=1, s=5)
plt.scatter(4.488, 300, label='1.333MeV Peak', zorder=2, s=15, color='red')
plt.scatter(3.987, 380, label='1.172MeV Peak', zorder=3, s=15, color='blue')
plt.xlabel('Voltage (V)')
plt.ylabel('Occurances')
plt.legend()


# In[113]:


plt.scatter(navolt, nacount, label='Sodium Data', color='orange', zorder=1, s=5)
plt.scatter(4.287, 1100, label='1.277MeV Peak', zorder=2, s=15, color='red')
plt.scatter(1.788, 9490, label='0.511MeV Peak', zorder=3, s=15, color='blue')
plt.xlabel('Voltage (V)')
plt.ylabel('Occurances')
plt.legend()


# In[20]:


csmaxind=np.argmax(cscount)
csmaxvolt=csvolt[csmaxind]
print('For Cesium, the occurances peak at:', round(csmaxvolt, 3), 'Volts')


# In[44]:


comaxind=np.argmax(cocount[175:200])
voltright=covolt[175:200]
comaxvolt=voltright[comaxind]
print('For Cobalt, the occurances peak at:', round(comaxvolt, 3), 'Volts')


# In[55]:


namaxind=np.argmax(nacount)
namaxvolt=navolt[namaxind]
print('For Sodium, the occurances peak at:', round(namaxvolt, 3), 'Volts')
print(namaxind)


# In[42]:


namaxind=np.argmax(nacount[100:200])
voltright=navolt[100:200]
namaxvolt=voltright[namaxind]
print('For Sodium, the occurances peak again at:', round(namaxvolt, 3), 'Volts')


# In[104]:


def mlm(x, A, B):
    return A + B*x
V = np.array([1.788,2.287,3.987,4.287,4.488])
E = np.array([0.511,0.622,1.172,1.277,1.333])
plt.scatter(V, E,zorder=2, label='Data')
popt, pcov = fit(mlm, V, E)
A = popt[0]
B = popt[1]
print('A is:', round(A,3), 'and B is:', round(B,3))
xdata=np.linspace(0,5.2,100)
ydata=A+B*xdata
plt.plot(xdata,ydata,color='orange',zorder=1,label='fit: E = -0.066 + 0.312*V')
plt.legend()
plt.xlabel('Voltage (V)')
plt.ylabel('Energy (MeV)')


# In[100]:


sumy = np.sum(E)
sumx = np.sum(V)
sumxy = np.sum(E*V)
sumyy = np.sum(E**2)
sumxx = np.sum(V**2)
n = 5
Amlm = (sumy*sumxx-sumxy*sumx)/(n*sumxx-sumx**2)
Bmlm = (n*sumxy - sumy*sumx)/(n*sumxx-sumx**2)
print('A is:', round(Amlm,3), 'and B is:', round(Bmlm,3))


# In[ ]:




