import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

file = '/Users/cedricbaerts/Desktop/Physics3926/Week12/Lab12/co2_mm_mlo.csv'

csvFile = np.loadtxt(file, skiprows=52, delimiter=",")

condition = ((1981 <= csvFile[:, 0]) & (csvFile[:, 0] <= 1990))
index = np.where(condition)
part1 = csvFile[index]

xData = part1[:,2]
yData = part1[:,3]
# otherpoly = np.polynomial.polynomial.Polynomial(xData, yData, 4)
poly = Polynomial.fit(xData, yData, 2, domain=[xData[0], xData[-1]])
# coefs = poly.convert(domain=poly.domain).coef
# yFit = np.polyval(coefs[::-1], xData)
yFit = poly(xData)

fig, ax = plt.subplots(2, 1, figsize=[8,8])

subtracted = yData - yFit

ax[0].plot(xData, yData, label='Data', color='maroon')
ax[0].plot(xData, yFit, label="Fitted Polynomial", linestyle='--')
ax[0].set_title("CO2 Data With Fitted Trend Line")
ax[0].set_ylabel("CO2 Concentration (PPM)")
ax[0].set_xlabel('Year')
ax[0].legend()
ax[1].plot(xData, subtracted, label='Data', color='maroon')
ax[1].set_title("Subtracted Data")
ax[1].set_ylabel("CO2 Concentration (PPM)")
ax[1].set_xlabel('Year')
ax[1].legend()
plt.savefig('BaertsCedric_Lab12_Fig1')
plt.show()