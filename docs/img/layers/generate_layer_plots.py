import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

def plot_func(x, y):
    plt.figure(figsize=(5, 3))
    plt.plot(x, y, 'k', linewidth=2)
    plt.gca().spines['left'].set_position('zero')
    plt.gca().spines['bottom'].set_position('zero')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.xticks([-3, -2, -1, 1, 2, 3])
    plt.xlim([-3.05, 3.05])
    plt.tight_layout()


plt.rc('font', family='Franklin Gothic Book')
xx = np.linspace(-6, 6, 601)

# Neg
plot_func(xx, -xx)
plt.yticks([-3, -2, -1, 1, 2, 3])
plt.ylim([-3.05, 3.05])
plt.savefig("neg.svg")
plt.show()

# Abs
plot_func(xx, abs(xx))
plt.yticks([-3, -2, -1, 1, 2, 3])
plt.ylim([-3.05, 3.05])
plt.savefig("abs.svg")
plt.show()

# Exp
plot_func(xx, np.exp(xx))
plt.yticks([5, 10, 15, 20])
plt.ylim([0, 20])
plt.savefig("exp.svg")
plt.show()

# Log
plot_func(xx, np.log(xx))
plt.yticks([-3, -2, -1, 1, 2, 3])
plt.ylim([-3, 3])
plt.xticks([1, 2, 3, 4, 5])
plt.xlim([0, 5])
plt.savefig("log.svg")
plt.show()

# Reciprocal
plot_func(xx, 1.0/xx)
plt.yticks([-20, -10, 10, 20])
plt.ylim([-20, 20])
plt.savefig("reciprocal.svg")
plt.show()

# Sqrt
plot_func(xx, np.sqrt(xx))
plt.yticks([-3, -2, -1, 1, 2, 3])
plt.ylim([0, 3])
plt.xticks([1, 2, 3, 4, 5])
plt.xlim([-0.05, 5])
plt.savefig("sqrt.svg")
plt.show()

# Sigmoid
plot_func(xx, 1.0/(1+np.exp(-xx)))
plt.yticks([0.5, 1])
plt.ylim([0, 1.05])
plt.xticks([-6, -4, -2, 2, 4, 6])
plt.xlim([-6, 6])
plt.savefig("sigmoid.svg")
plt.show()

# Relu
plot_func(xx, np.maximum(xx, 0.0))
plt.yticks([1, 2, 3])
plt.ylim([-0.05, 3.05])
plt.savefig("relu.svg")
plt.show()