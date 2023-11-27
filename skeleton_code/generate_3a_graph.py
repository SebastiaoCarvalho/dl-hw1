import matplotlib.pyplot as plt
import numpy as np

points = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
labels = np.array([1 if np.sum(point) >= -1 and np.sum(point) <= 1 else -1 for point in points])

for i, point in enumerate(points):
    plt.plot(point[0], point[1], 'ro' if labels[i] == 1 else 'bo', markersize=7)

plt.axis([-2, 2, -2, 2])
plt.arrow(-2, 0, 3.9, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')  # x-axis arrow
plt.arrow(0, -2, 0, 3.88, head_width=0.1, head_length=0.1, fc='k', ec='k')  # y-axis arrow


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Non-linearly separable data example')

plt.savefig('3a.png')
plt.close()