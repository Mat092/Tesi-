import numpy as np
from nottetempo import noises
import matplotlib.pyplot as plt

dataset = "moons"

linksFile = "/home/mattia/dati/" + dataset +"/links_data.npy"
minFile = "/home/mattia/dati/" + dataset + "/min_data.npy"
lenFile = "/home/mattia/dati/" + dataset + "/len_data.npy"

links_data = np.load(linksFile)
min_data = np.load(minFile)
len_data = np.load(lenFile)

mean_links = np.mean(links_data, axis = 0)
mean_min = np.mean(min_data, axis = 0)
mean_len= np.mean(len_data, axis = 0)

dev_links = np.std(links_data,axis = 0)
dev_min = np.std(min_data,axis = 0)
dev_len = np.std(len_data,axis = 0)

plt.figure(figsize=(5,5))

plt.plot(noises, mean_links)
plt.fill_between(noises,mean_links-dev_links,mean_links+dev_links,alpha = 0.5)
plt.xlabel("noise")
plt.ylabel("Links")

plt.figure(figsize=(5,5))

plt.plot(noises,mean_min)
plt.fill_between(noises,mean_min-dev_min,mean_min+dev_min,alpha = 0.5)
plt.xlabel("noise")
plt.ylabel("Layer più piccolo")

plt.figure(figsize=(5,5))

plt.plot(noises,mean_len)
plt.fill_between(noises,mean_len-dev_len, mean_len+dev_len, alpha = 0.5)
plt.xlabel("noise")
plt.ylabel("Profondità")


#figname = "/home/mattia/Scrivania/drift/gen" + str(j) + ".png"
#
#plt.savefig(fname = figname)
#plt.show()