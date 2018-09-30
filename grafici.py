import numpy as np
from nottetempo import noises
import matplotlib.pyplot as plt
import seaborn as sns

dataset = "circles+"

linksFile = "dati/" + dataset +"/links_data.npy"
minFile = "dati/" + dataset + "/min_data.npy"
lenFile = "dati/" + dataset + "/len_data.npy"

links_data = np.load(linksFile)
min_data = np.load(minFile)
len_data = np.load(lenFile)

print(len(links_data))

mean_links = np.mean(links_data, axis = 0)
mean_min = np.mean(min_data, axis = 0)
mean_len= np.mean(len_data, axis = 0)

dev_links = np.std(links_data,axis = 0)
dev_min = np.std(min_data,axis = 0)
dev_len = np.std(len_data,axis = 0)


plt.figure(figsize=(5,5))
plt.plot(noises, mean_links,color = "black",label = "Links/Noise graph")
plt.fill_between(noises,mean_links-dev_links,mean_links+dev_links,
                 alpha = 0.5,color = "grey")
plt.xlabel("Noise",fontsize = "x-large")
plt.ylabel("Links",fontsize = "x-large")
plt.grid(1,"both","both")
plt.legend(fontsize = "x-large")
sns.despine(plt.gcf(), trim=True, offset = 10)
plt.ylim(0)
#plt.xlim()
plt.tight_layout()
plt.savefig("/home/mattia/Scrivania/Tesi/Tesi_latex/images/links_noise_" + 
            dataset +".png")


plt.figure(figsize=(5,5))
plt.plot(noises,mean_min,color = "black",label = "Smaller Layer/Noise graph")
plt.fill_between(noises,mean_min-dev_min,mean_min+dev_min,
                 alpha = 0.5,color = "grey")
plt.xlabel("Noise",fontsize = "x-large")
plt.ylabel("Smaller Layer",fontsize = "x-large")
plt.grid(1,"both","both")
plt.legend(fontsize = "x-large",loc = 1)
plt.ylim(0)
sns.despine(plt.gcf(), trim=True, offset = 10)
plt.tight_layout()
plt.savefig("/home/mattia/Scrivania/Tesi/Tesi_latex/images/small_noise_" + 
            dataset+".png")


plt.figure(figsize=(5,5))
plt.plot(noises,mean_len,color = "black",label = "Depth/Noise graph")
plt.fill_between(noises,mean_len-dev_len, mean_len+dev_len, 
                 alpha = 0.5,color = "grey")
plt.xlabel("Noise",fontsize = "x-large")
plt.ylabel("Depth",fontsize = "x-large")
plt.grid(1,"both","both")
plt.legend(fontsize = "x-large")
sns.despine(plt.gcf(), trim=True, offset = 10)
plt.ylim(0)
plt.tight_layout()
plt.savefig("/home/mattia/Scrivania/Tesi/Tesi_latex/images/depth_noise_" + 
            dataset+".png")

plt.show()
