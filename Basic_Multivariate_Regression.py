#   Multivariate Regression
#   Code Written by David Brouwer
#   pws - 2017

import numpy as np

# Set a data set, both input and output. And declare the theta vector, by default we set theta to zero.
# Here we use a training set with 3 examples, of which each has 3 features
input_features = np.array(([1,6,6], [1,4,6], [1,2,6]), dtype=float)
output_goal = [1, 4, 7]
theta = np.array((0,0,0), dtype=float)
tempsum = 1
counter = 0


# hypothesis = theta0 * x0 + theta1 * x1 + theta2 * x2
# alpha = learning rate
alpha = 0.066
h = lambda x, x1, x2: theta[0] * x + theta[1] * x1 + theta[2] * x2

# define the cost function
def cost_function(hypothesis, input, output):
    sum = 0

    for m in range(0, 2):
        sum += ((hypothesis(input[m,0], input[m,1], input[m,2]) - output[m])**2)

    cost = sum / (2 * int(len(input)))
    return cost

# define Gradient Descent
def gradient_descent(hypothesis, input, output):
    temp0 = 0
    temp1 = 0
    temp2 = 0

    for i in range(1, len(input)):
        temp0 += (hypothesis(input[i,0], input[i,1], input[i,2]) - output[i]) / len(input)
        temp1 += ((hypothesis(input[i,0], input[i,1], input[i,2]) - output[i]) * input[i,1]) / len(input)
        temp2 += ((hypothesis(input[i, 0], input[i, 1], input[i, 2]) - output[i]) * input[i, 2]) / len(input)

    return temp0, temp1, temp2

# define percenage change function
def change(new, old):
    change_var = ((new - old)/ old) * 100.00

    return change_var

# Calculate the cost for theta = 0
cost_before = cost_function(h, input_features, output_goal)

# set the number of steps Gradient Descent has to take
iterations = 500000

# Let Gradient Descent work, and update theta
while(tempsum != 0.0):
    temp0, temp1, temp2 = gradient_descent(h, input_features, output_goal)
    tempsum = temp0 + temp1 + temp2
    counter += 1
    theta[0] = theta[0] - alpha * temp0
    theta[1] = theta[1] - alpha * temp1
    theta[2] = theta[2] - alpha * temp2

# Print the values of theta
print('theta zero = ', theta[0])
print('theta one = ', theta[1])
print('theta two = ', theta[2])
print()

# Calculate the cost after optimization of theta
cost_after = cost_function(h, input_features, output_goal)

# calculate the change in cost in %
cost_change = change(cost_after, cost_before)

print('cost before = %.2f' % cost_before + ', cost after = %.5f' % cost_after)
print('the cost function has changed with %.5f' % cost_change + '%')
print('the counter = ', counter)
print(tempsum)
