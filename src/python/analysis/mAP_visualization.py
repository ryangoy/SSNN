import numpy as np
import matplotlib.pyplot as plt


N = 4
menMeans = [19.4, 0.5, 11.3, 21.1]
print sum(menMeans)/4
# womenMeans = [0.0999000999000999, 0.0, 2.3804543362757813, 0.1567398119122257]

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width)
# p2 = plt.bar(ind, womenMeans, width,
#              )

plt.ylabel('mAP')
plt.title('mAP by category for Stanford dataset single detecotor @ 0.25 IoU')
plt.xticks(ind, ('table', 'board', 'chair', 'sofa'))
plt.yticks(np.arange(0, 100, 10))
# plt.legend((p1[0], p2[0]), ('0.5 IoU', '0.25 IoU'))

plt.show()