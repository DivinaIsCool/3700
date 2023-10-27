#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from scipy import stats 
from scipy.optimize import curve_fit 
from scipy.stats import norm


# In[2]:


# Small dictionary with our selected changes.
style_revisions = { 
    'font.size' : 10,
    'figure.figsize' : [7.5, 5.5],
    'axes.linewidth': 1.5,
    'lines.linewidth' : 1.5,
    'xtick.top' : True,
    'ytick.right' : True, 
    'xtick.direction' : 'in',
    'ytick.direction' : 'in', 
    'xtick.major.size' : 5,
    'ytick.major.size' : 11, 
    'xtick.minor.size' : 5,
    'ytick.minor.size' : 5.5
}


# In[3]:


# Update the matplotlib dictionary.
plt.rcParams.update(style_revisions)


# In[4]:


# Define a function to take Chi Squared

def chi(xi, xti, sigi):
    di = (xi-xti)
    chivals = di**2 / sigi
    return sum(chivals)


# In[5]:


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
    col3 = []

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
                    val3 = float(words[2])
                    col1.append(val1)
                    col2.append(val2)
                    col3.append(val3)

    # All done
    return col1, col2, col3

R1_14, R2_14, R12_14=get_data('Colton_Zavier PCB14')
R1_15, R2_15, R12_15=get_data('Colton_Zavier PCB15')
R1_16, R2_16, R12_16=get_data('Colton_Zavier PCB16')
R1_17, R2_17, R12_17=get_data('Colton_Zavier PCB17')
R1_18, R2_18, R12_18=get_data('Colton_Zavier PCB18')


# In[6]:


R1 = np.array(R1_14)
R1 = np.append(R1, R1_15)
R1 = np.append(R1, R1_16)
R1 = np.append(R1, R1_17)
R1 = np.append(R1, R1_18)
R2 = np.array(R2_14)
R2 = np.append(R2, R2_15)
R2 = np.append(R2, R2_16)
R2 = np.append(R2, R2_17)
R2 = np.append(R2, R2_18)
R12 = np.array(R12_14)
R12 = np.append(R12, R12_15)
R12 = np.append(R12, R12_16)
R12 = np.append(R12, R12_17)
R12 = np.append(R12, R12_18)


# In[65]:


hist1, bound1 = np.histogram(R1, bins=14)
center1=np.array([])
for i, o in enumerate(bound1):
    if i < (len(bound1)-1):
        center = (bound1[i] + bound1[i+1])/2
        center1=np.append(center1, center)
# Define the xdata and ydata
rxdata=center1
rydata=hist1
nbin=150
ahist1, abound1 = np.histogram(R1, bins=nbin)
acenter1=np.array([])
for i, o in enumerate(abound1):
    if i < (len(abound1)-1):
        center = (abound1[i] + abound1[i+1])/2
        acenter1=np.append(acenter1, center)
# Define the Gaussian function
def Gauss(x, A, B, C):
    return A * np.exp(-((x - B) ** 2) / (2 * C ** 2))
xdata=acenter1
ydata=ahist1
# Perform the Gaussian fit with initial parameter guesses
initial_guess = [10, 1075, 20]  # You can adjust these initial values
parameters, covariance = curve_fit(Gauss, xdata, ydata, p0=initial_guess)

fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

# Generate the fitted curve using the obtained parameters
fit_y = Gauss(xdata, fit_A, fit_B, fit_C)

# Plot the data and the fit
plt.scatter(rxdata, rydata, label='Data', color='black', zorder=3, s=20)
error1=hist1**.5
plt.errorbar(center1,hist1,yerr=error1, linestyle='', color='black', zorder=2, elinewidth=1)
plt.plot(xdata, fit_y*nbin/15, '-', label='Fit', color='orange', zorder=1)
plt.legend()
plt.xlabel('R1 (Ω)')
plt.ylabel('Occurances')
plt.show()
plt.savefig('r1')
print('The σ_1 value is:', round(fit_C,2))
rydatainterp = np.interp(rxdata, xdata, fit_y*nbin/15)
chi1=chi(rydata,rydatainterp,error1)
print('The Χ^2_1 value is:', round(chi1,2))
print('The Χ^2_1/dof value is:', round(chi1,2)/(len(center1)-3))


# In[64]:


rbin=35
hist2, bound2 = np.histogram(R2, bins=rbin)
center1=np.array([])
center2=np.array([])
for i, o in enumerate(bound2):
    if i < (len(bound2)-1):
        center = (bound2[i] + bound2[i+1])/2
        center2=np.append(center2, center)
# Define the xdata and ydata
rxdata=center2
rydata=hist2
nbin=150
ahist2, abound2 = np.histogram(R2, bins=nbin)
acenter1=np.array([])
acenter2=np.array([])
for i, o in enumerate(abound2):
    if i < (len(abound2)-1):
        center = (abound2[i] + abound2[i+1])/2
        acenter2=np.append(acenter2, center)
# Define the Gaussian function
def Gauss(x, A, B, C):
    return A * np.exp(-((x - B) ** 2) / (2 * C ** 2))
xdata=acenter2
ydata=ahist2
# Perform the Gaussian fit with initial parameter guesses
initial_guess = [1, 1270, 15]  # You can adjust these initial values
parameters, covariance = curve_fit(Gauss, xdata, ydata, p0=initial_guess)

fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

# Generate the fitted curve using the obtained parameters
fit_y = Gauss(xdata, fit_A, fit_B, fit_C)

# Plot the data and the fit
plt.scatter(rxdata, rydata, label='Data', color='black', zorder=3, s=13)
error2=hist2**.5
plt.errorbar(center2,hist2,yerr=error2, linestyle='', color='black', zorder=2, elinewidth=1)
plt.plot(xdata, fit_y*nbin/rbin, '-', label='Fit', color='orange', zorder=1)
plt.legend()
plt.xlabel('R2 (Ω)')
plt.ylabel('Occurances')
plt.show()
plt.savefig('r2')
print('The σ_2 value is:', round(fit_C,2))
rydatainterp = np.interp(rxdata, xdata, fit_y*nbin/rbin)
useerror2 = np.array([])
for i in error2:
    if i != 0:
        useerror2=np.append(useerror2,i)
userydata=np.array([])
for i, o in enumerate(rydata):
    if error2[i] != 0:
        userydata=np.append(userydata,o)
userydatainterp=np.array([])
for i, o in enumerate(rydatainterp):
    if error2[i] != 0:
        userydatainterp=np.append(userydatainterp,o)
chi2=chi(userydata,userydatainterp,useerror2)
print('The Χ^2_1 value is:', round(chi2,2))
print('The Χ^2_1/dof value is:', round(chi2,2)/(len(center2)-3))


# In[62]:


rbin=60
hist12, bound12 = np.histogram(R12, bins=rbin)
center12=np.array([])
for i, o in enumerate(bound12):
    if i < (len(bound12)-1):
        center = (bound12[i] + bound12[i+1])/2
        center12=np.append(center12, center)
# Define the xdata and ydata
rxdata=center12
rydata=hist12
nbin=150
ahist12, abound12 = np.histogram(R12, bins=nbin)
acenter12=np.array([])
for i, o in enumerate(abound12):
    if i < (len(abound12)-1):
        center = (abound12[i] + abound12[i+1])/2
        acenter12=np.append(acenter12, center)
# Define the Gaussian function
def Gauss(x, A, B, C):
    return A * np.exp(-((x - B) ** 2) / (2 * C ** 2))
xdata=acenter12
ydata=ahist12
# Perform the Gaussian fit with initial parameter guesses
initial_guess = [1, 2360, 10]  # You can adjust these initial values
parameters, covariance = curve_fit(Gauss, xdata, ydata, p0=initial_guess)

fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

# Generate the fitted curve using the obtained parameters
fit_y = Gauss(xdata, fit_A, fit_B, fit_C)

# Plot the data and the fit
plt.scatter(rxdata, rydata, label='Data', color='black', zorder=3, s=11)
error12=hist12**.5
plt.errorbar(center12,hist12,yerr=error12, linestyle='', color='black', zorder=2, elinewidth=1)
plt.plot(xdata, fit_y*nbin/rbin, '-', label='Fit', color='orange', zorder=1)
plt.legend()
plt.xlabel('R1 + R2 (Ω)')
plt.ylabel('Occurances')
plt.show()
plt.savefig('r12')
print('The σ_12 value is:', round(fit_C,2))
rydatainterp = np.interp(rxdata, xdata, fit_y*nbin/rbin)
useerror12 = np.array([])
for i in error12:
    if i != 0:
        useerror12=np.append(useerror12,i)
userydata=np.array([])
for i, o in enumerate(rydata):
    if error12[i] != 0:
        userydata=np.append(userydata,o)
userydatainterp=np.array([])
for i, o in enumerate(rydatainterp):
    if error12[i] != 0:
        userydatainterp=np.append(userydatainterp,o)
chi12=chi(userydata,userydatainterp,useerror12)
print('The Χ^2_1 value is:', round(chi12,2))
print('The Χ^2_1/dof value is:', round(chi12,2)/(len(center12)-3))


# In[ ]:




