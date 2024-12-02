import numpy as np

file = '/Users/cedricbaerts/Desktop/Physics3926/Week12/Lab12/co2_mm_mlo.csv'

csvFile = np.loadtxt(file, skiprows=52, delimiter=",")

condition = ((1981 <= csvFile[:, 0]) & (csvFile[:, 0] <= 1990))
index = np.where(condition)
part1 = csvFile[index]
print(part1)
