import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# https://github.com/Cbaerts/Lab12

# File Path
file = '/Users/cedricbaerts/Desktop/Physics3926/Week12/Lab12/co2_mm_mlo.csv'
# Open File and skip explaining rows and seperate by commas
csvFile = np.loadtxt(file, skiprows=52, delimiter=",")

# Part 1
# Set condition to grab data
condition = ((1981 <= csvFile[:, 0]) & (csvFile[:, 0] <= 1990))
# Find where condition = true 
index = np.where(condition)
# Pull Data
part1 = csvFile[index]

# Set Data to values
xData = part1[:,2]
yData = part1[:,3]

# Followed this tutorial to find the polynomial coefficients
# https://www.youtube.com/watch?v=dQw4w9WgXcQ
# An attempt but didnt work
# otherpoly = np.polynomial.polynomial.Polynomial(xData, yData, 4)
# Find Polynomial of the data w/ 2 degrees, between year boundaries
poly = Polynomial.fit(xData, yData, 2, domain=[xData[0], xData[-1]])
# coefs = poly.convert(domain=poly.domain).coef
# yFit = np.polyval(coefs[::-1], xData)
# Sub in year values into the polynomial
yFit = poly(xData)

# Create two plots one above another
fig, ax = plt.subplots(2, 1, figsize=[8,8])
# Find residual data
subtracted = yData - yFit
# Plot fitted line and data
ax[0].plot(xData, yData, label='Data', color='maroon')
ax[0].plot(xData, yFit, label="Fitted Polynomial", linestyle='--')
ax[0].set_title("CO2 Data From Mauna Loa, Hawai'i From 1981-1990")
ax[0].set_ylabel("CO2 Concentration (PPM)")
ax[0].set_xlabel('Year')
ax[0].legend()
ax[1].plot(xData, subtracted, label='Data', color='maroon')
ax[1].set_title("Residual Data")
ax[1].set_ylabel("CO2 Concentration (PPM)")
ax[1].set_xlabel('Year')
ax[1].legend()
# plt.savefig('BaertsCedric_Lab12_Fig1')
# plt.show()
plt.close()

# Part 2
# Trial and error sine parameters
amp = 3.5
period = 1
phase = 5.8
trialNerror = amp* np.sin(2*np.pi*(xData/period) + phase) - 0.3

# Plot trial and error sine function over the residual data
plt.subplot()
plt.plot(xData, subtracted, label='Residual Data', color='maroon')
plt.plot(xData, trialNerror, label='Simple Sinusoidal Function', color='navy')
plt.title("Subtracted Data")
plt.ylabel("Residual CO2 Concentration (PPM)")
plt.xlabel('Year')
plt.legend()
# plt.show()
plt.close()

# Fourier transfor of residuals
fourier = np.fft.fft(subtracted)
n = subtracted.size
timestep = xData[1] - xData[0]
freq = np.fft.fftfreq(n, d=timestep)

# Plot fourier transform
plt.plot(freq[:len(freq)//2], np.abs(fourier)[:len(freq)//2], label='All Fourier Amplitude', color='dodgerblue')
plt.title("Fourier Transform of Residuak Data")
plt.xlabel("Frequency (1/year)")
plt.ylabel("Fourier Amplitude")
plt.axis((0, 6, 0, 200))
# plt.show()
plt.close()
# These values do agree with my estimated period

# Part 3
# Take all data after 1981
condition = ((1981 <= csvFile[:, 0]))
index = np.where(condition)
part3 = csvFile[index]
xData = part3[:,2]
yData = part3[:,3]
yFit = poly(xData)

print(xData[np.where(yFit>400)[0][0]])
print(xData[np.where(yData>400)[0][0]])
# Predidcted date for 400 PPM is in 2011 (February) while the actual data was in 2014 (March)
# The Predicted polynomial diverges from the actual data around 1990-1995
# Slowed down the CO2 usage rate (yay world kinda)
plt.plot(xData, yData, label='Data', color='maroon')
plt.plot(xData, yFit, label="Fitted Polynomial", linestyle='--', color='dodgerblue')
plt.title("CO2 Data From Mauna Loa, Hawai'i From 1981-2020")
plt.ylabel("CO2 Concentration (PPM)")
plt.xlabel('Year')
plt.legend()
# plt.savefig('BaertsCedric_Lab12_Fig2')
plt.show()
plt.close()