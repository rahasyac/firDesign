# Name : FNU Rahasya Chandan
# UTA Id: 1000954962
#: Design of Lowpass and Highpass FIR Filters

import numpy as np 
import matplotlib.pyplot as plt # matlab plotting in python

#Load data from a file, with missing values handled using delimeter
# data is the y-axis
data = np.genfromtxt('data-filtering.csv', delimiter=',') 

# We have to create two different filters, one lowpass and one highpass

# (a) Low Pass Filter with specific conditions mentioned below 
x = np.arange(0, 2000, 1) #numpy.arange([start, ]stop, [step, ]dtype=None)
y = data

fc = 50     #cut-off frequency 
fs = 2000   #sampling rate for the data
L = 21      #filter length
M = L - 1   #filter length - 1
ft = fc/fs  #normalized cut-off frequency

# (b) Plot all of the values of the filtered signal
# Plot Original Signal
plt.figure(1)
plt.subplot(3,1,1) #subplot(nrows, ncols, index)
plt.title("Original Signal")
plt.plot(x,y)

# Plot 4Hz Signal
#y-axis
x1 = np.cos(2*np.pi*4*(x/fs)) #function to help calculate cosine for all x

plt.subplot(3,1,2)
plt.title("4 Hz Signal")
plt.plot(x, x1)

# Plot Application of Low Pass Filter
# Filter weight w[n]
w1 = np.zeros(L)  # returns a new array filled with zeros

for n in range(len(w1)):
	if n == (M/2):
		w1[n] = 2*ft
	else:
		s1 = np.sin(2*np.pi*ft*(n - (M/2))) # s1 = sin of filter weight 1
		w1[n] = s1/(np.pi*(n-(M/2)))

#we use convolve() function to apply the filter to the ORIGINAL signal.
LowFilter = np.convolve(data, w1)

plt.subplot(3,1,3)
plt.tight_layout() # To adjust subplot paramaters so that the subplot(s) fits in to the figure area. 
plt.title("Application of Lowpass Filter")
plt.plot(LowFilter)
plt.show()


# (C) Highpass filter with specific conditions mentioned below 
x = np.arange(0,100,1) #numpy.arange([start, ]stop, [step, ]dtype=None)
y = data[:100]

fc = 280     #cut-off frequency 
fs = 2000    #sampling rate for the data
L = 21       #filter length
M = L - 1    #filter length - 1
ft = fc/fs   #normalized cut-off frequency

# (d) Plot all of the values of the filtered signal
# Plot Original Signal
plt.figure(2)
plt.subplot(3,1,1) #subplot(nrows, ncols, index)
plt.title("Original Signal") 
plt.plot(x,y)

# Plot 330Hz Signal
# x-axis
z = np.arange(0,100,1) #numpy.arange([start, ]stop, [step, ]dtype=None)
# y-axis
x1 = np.cos(2*np.pi*330*(z/fs))  #function to help calculate cosine for all z

plt.subplot(3,1,2)
plt.title("330 Hz Signal")
plt.plot(x1)


# Plot Application of High Pass Filter
# Filter weight w[n]
w2 = np.zeros(L) # returns a new array filled with zeros

for n in range(len(w2)):
	if n == (M/2):
		w2[n] = (1 - 2*ft)
	else:
		s2 = np.sin(2*np.pi*ft*(n - (M/2))) # s2 = sin of filter weight 2
		w2[n] = -s2/(np.pi*(n-(M/2)))
        
#we use convolve() function to apply the filter to the ORIGINAL signal.
HighFilter = np.convolve(data, w2)
hfp = HighFilter[:100]

plt.subplot(3,1,3)
plt.tight_layout() # To adjust subplot paramaters so that the subplot(s) fits in to the figure area. 
plt.title("Application of Highpass Filter")
plt.plot(hfp)
plt.show()

