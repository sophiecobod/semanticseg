import numpy as np
pathFig = "./figures/"
listFig = "list.txt"
fig_list = np.genfromtxt(pathFig + listFig, delimiter='*', dtype=None)
print(fig_list)