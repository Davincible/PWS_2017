#
#   Writen By David Brouwer
#   Pws - 2017

from matplotlib import pyplot as plt

input_features = [1, 2, 3, 4, 5, 6, 7, 8]
output_goal = [1, 4, 7, 6, 2, 8, 9, 4]

# hypothesis = theta0 + theta1 * x
# alpha = learning rate

theta0 = 0
theta1 = 0
alpha = 0.01
h = lambda x: theta0 + theta1 * x*x

# define the function to create the graph
def plot_line(y, data_points):
    graph_width = [i for i in range(int(min(data_points)) - 1, int(max(data_points)) + 2)]
    graph_height = [y(x) for x in graph_width]
    plt.plot(graph_width, graph_height, 'r')

# define Gradient Descent
def gradient_descent(hypothesis, input, output):
    temp0 = 0
    temp1 = 0

    for i in range(1, len(input)):
        temp0 += (hypothesis(input[i]) - output[i]) / len(input)
        temp1 += ((hypothesis(input[i]) - output[i]) * input[i]) / len(input)

    return temp0, temp1

# set the number of steps Gradient Descent has to take
iterations = 100

# Let Gradient Descent work, and update theta
for i in range(iterations):
    temp0, temp1 = gradient_descent(h, input_features, output_goal)
    theta0 = theta0 - alpha * temp0
    theta1 = theta1 - alpha * temp1

print('theta zero = ', theta0)
print('theta one = ', theta1)

#plot the graph
plt.plot(input_features, output_goal, 'bo')
plot_line(h, input_features)
plt.show()