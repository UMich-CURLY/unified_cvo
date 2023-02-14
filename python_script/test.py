import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# generate data for each of the three box plots in each subplot
np.random.seed(0)
data1_1 = np.random.normal(100, 10, 100)
data1_2 = np.random.normal(120, 20, 100)
data1_3 = np.random.normal(140, 30, 100)

data2_1 = np.random.normal(90, 10, 100)
data2_2 = np.random.normal(110, 20, 100)
data2_3 = np.random.normal(130, 30, 100)

data3_1 = np.random.normal(80, 10, 100)
data3_2 = np.random.normal(100, 20, 100)
data3_3 = np.random.normal(120, 30, 100)
print(data3_3.shape)
# create a figure and axis
fig, ax = plt.subplots(1, 5, figsize=(15, 5), sharey=True)

for i in range(5):
    # plot the first box plot in the first subplot
    bp = ax[i].boxplot([data1_1, data1_2, data1_3], vert=True, showfliers=False, patch_artist=True)
    ax[i].set_xlabel('0.0')
    ax[i].set_xticks([])
    ax[i].set_title('Box Plot 1 of Error')
    if (i == 0):
        ax[i].set_ylabel('Error')
    ax[i].set_yticks([])

    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
plt.subplots_adjust(wspace=0)
#plt.show()


fig, ax = plt.subplots(1, 3)

box1 = [1,2,3,4,5]
box2 = [5,4,3,2,1]

ax[0].boxplot(box1)
ax[0].set_title("Subplot 1")

ax[1].boxplot(box2)
ax[1].set_title("Subplot 2")

ax[2].boxplot([box1, box2])
ax[2].set_title("Subplot 3")

fig.legend(["Box 1", "Box 2"])
plt.show()