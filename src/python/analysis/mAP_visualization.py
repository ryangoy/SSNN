# visualizes the distribution of mAP per category. hardcoded.

import numpy as np
import matplotlib.pyplot as plt


mAPs1 = [19.4, 0.5, 11.3, 21.1]
N = len(mAPs1)
mAPs2 = [0.0999000999000999, 0.0, 2.3804543362757813, 0.1567398119122257]
M = len(mAPs2)
assert N == M

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, mAPs1, width)
p2 = plt.bar(ind, mAPs2, width)

plt.ylabel('mAP')
plt.title('mAP by category for Stanford dataset single detector')
plt.xticks(ind, ('table', 'board', 'chair', 'sofa'))
plt.yticks(np.arange(0, 100, 10))
plt.legend((p1[0], p2[0]), ('0.25 IoU', '0.5 IoU'))

plt.show()