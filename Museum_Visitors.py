#
# Predict the amount of visitors a museum attracts based on 6 input features
#   Written by David Brouwer
#   pws - 2017
#

import numpy as np

# x1 = number of exhibits
# x2 = age of the oldest exhibit
# x3 = age of the newest exhibit
# x4 = entrance price
# x5 = number of stars the local restaurant has (between 1 and 5)
# x6 = average value of the exhibits

training_set = np.array(([1, 50, 10, 5, 6.95, 1, 2000], [1, 150, 12, 1, 9.95, 1, 2400],
                         [1, 500, 34, 5, 12, 2, 3000], [1, 1240, 80, 2, 13.65, 2, 4000],
                         [1, 1500, 144, 6, 14.50, 3, 3900], [1, 4000, 180, 2, 15.00, 3, 5000],
                         [1, 5000, 360, 40, 13.78, 2, 3680], [1, 9500, 250, 26, 16.90, 3, 5600],
                         [1, 12000, 320, 32, 18.00, 3, 8000], [1, 16000, 20, 20, 11.60, 4, 5000],
                         [1, 24000, 500, 400, 19.70, 5, 6660], [1, 25666, 666, 44, 16.66, 5, 8585]), dtype=float)

target_output = np.array(([4500000], [4660000], [5600000], [7000000], [7500000], [680000], [7300000], [6800000],
                          [800000], [7800000], [8450000], [9999000]), dtype=float)

# Declare Variables
theta = np.array((0, 0, 0, 0, 0, 0, 0), dtype=float)
temp0 = 0
temp1 = 0
temp2 = 0
temp3 = 0
temp4 = 0
temp5 = 0
temp6 = 0
alpha = 8  # Learning rate
iterations = 10000  # number of iterations
gd_threshold = 0.00001  # when Gradient Descent passes this threshold, the algorithm will stop.

# defining the hypothesis function
h = lambda x0, x1, x2, x3, x4, x5, x6 : theta[0]*x0 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3 + theta[4]*x4 + \
                                        theta[5]*x5 + theta[6]*x6


# defining the function to scale our input features
def mean_normalization(in_put):
    mean_array = np.mean(in_put, axis=0)
    max_array = np.amax(in_put, axis=0)

    for m in range(0, len(in_put)):
        for n in range(0, len(in_put[m])):
            in_put[m, n] = float((in_put[m, n] - mean_array[n]) / max_array[n])


# defining the function to calculate the change of the cost function in per cent
def change(old, new):
    change_var = ((new - old)/ old) * 100.00
    return change_var


# defining the Gradient Descent function
def gradient_descent(hypothesis, in_put, out_put):
    temp_0 = 0
    temp_1 = 0
    temp_2 = 0
    temp_3 = 0
    temp_4 = 0
    temp_5 = 0
    temp_6 = 0

    for m in range(0, len(in_put) - 1):
        temp_0 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 0])
        temp_1 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 1])
        temp_2 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 2])
        temp_3 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 3])
        temp_4 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 4])
        temp_5 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 5])
        temp_6 += ((hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                               in_put[m, 6]) - target_output[m]) * in_put[m, 6])

    temp_0 /= len(in_put)
    temp_1 /= len(in_put)
    temp_2 /= len(in_put)
    temp_3 /= len(in_put)
    temp_4 /= len(in_put)
    temp_5 /= len(in_put)
    temp_6 /= len(in_put)

    return temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6


# define the cost function; sum of the squared errors
def cost_function(hypothesis, in_put, out_put):
    sum = 0

    for m in range(0, len(in_put) - 1):
        sum += (hypothesis(in_put[m, 0], in_put[m, 1], in_put[m, 2], in_put[m, 3], in_put[m, 4], in_put[m, 5],
                           in_put[m, 6]) - out_put[m])**2
    cost = sum / (2 * len(in_put))
    return cost

# calculate and print out the cost before applying Gradient Descent, with theta = 0
old_cost = cost_function(h, training_set, target_output)
print(old_cost)

# apply mean normalization to both the input and output data
mean_normalization(training_set)
mean_normalization(target_output)

sum = 1
numofit = 0  # a counter for the number of iterations

while sum > gd_threshold:
    temp0, temp1, temp2, temp3, temp4, temp5, temp6 = gradient_descent(h, training_set, target_output)
    sum = float(abs(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6))
    numofit += 1

    theta[0] = theta[0] - alpha * temp0
    theta[1] = theta[1] - alpha * temp1
    theta[2] = theta[2] - alpha * temp2
    theta[3] = theta[3] - alpha * temp3
    theta[4] = theta[4] - alpha * temp4
    theta[5] = theta[5] - alpha * temp5
    theta[6] = theta[6] - alpha * temp6

new_cost = cost_function(h, training_set, target_output)
print(new_cost)
change_in_cost = change(old_cost, new_cost)
print('The cost function decreased with: %.2f' % change_in_cost + '%')
print('To reach the threshold value, Gradient Descent has iterated %d times, \nwith a learning rate of 8' % numofit)
print('the values of theta are: ', theta)
