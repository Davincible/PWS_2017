import numpy as np

# number of exhibits between 50 and 70.000
# age of the oldest exhibit
# age of the newest exhibit
# entrance price
# number of stars the local restaurant has (between 1 and 5)
# average value of the exhibits

training_set = np.array(([50, 10, 5, 6.95, 1, 2000], [150, 12, 1, 9.95, 1, 2400],
                         [500, 34, 5, 12, 2, 3000], [1240, 80, 2, 13.65, 2, 4000],
                         [1500, 144, 6, 14.50, 3, 3900], [4000, 180, 2, 15.00, 3, 5000],
                         [5000, 360, 40, 13.78, 2, 3680], [9500, 250, 26, 16.90, 3, 5600],
                         [12000, 320, 32, 18.00, 3, 8000], [16000, 20, 20, 11.60, 4, 5000],
                         [24000, 500, 400, 19.70, 5, 6660], [25666, 666, 44, 16.66, 5, 8585]), dtype=float)

target_output = np.array(([4500000],[4660000],[5600000],[7000000],[7500000],[680000],[7300000],[6800000],[800000],[7800000],[8450000],[9999000]), dtype=float)

# Declare Variables
theta = np.array((0, 0, 0, 0, 0, 0), dtype=float)
temp0 = 0
temp1 = 0
temp2 = 0
temp3 = 0
temp4 = 0
temp5 = 0
alpha = 0.001 # Learning rate


h = lambda x0, x1, x2, x3, x4, x5 : theta[0]*x0 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3**2 + theta[4]*x4 + theta[5]*x5

def gradient_descent(hypothesis, in_put, out_put):
    temp_0 = 0
    temp_1 = 0
    temp_2 = 0
    temp_3 = 0
    temp_4 = 0
    temp_5 = 0

    for m in range(0, len(in_putput) - 1):
        temp_0 += hypothesis(in_put[m,0], in_put[m,1], in_put[m,2], in_put[m,3], in_put[m,4], in_put[m,5]) -
