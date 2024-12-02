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
p = Polynomial.fit(xData, yData, 1)
print(p)
coefs = p.convert(domain=p.domain).coef
yFit = np.polyval(coefs[::-1], xData)
print(yFit)
plt.plot(xData, yData, label='Data')
# plt.plot(xData, yFit, label="Fitted Polynomial", linestyle='--')
plt.legend()
plt.show()