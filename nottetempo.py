import Algorithm as alg
import numpy as np

repetions = 50
final_noise = 1.0
starting_noise = 0.0

dataset = "circles+"

linksFile = "/home/mattia/dati/" + dataset +"/links_data.npy"
minFile = "/home/mattia/dati/" + dataset + "/min_data.npy"
lenFile = "/home/mattia/dati/" + dataset + "/len_data.npy"

best_links = np.zeros(repetions)
min_layers = np.zeros(repetions)
lenghts = np.zeros(repetions)
noises = np.linspace(starting_noise,final_noise,repetions)
    
def funz():
    for j in range(30):
        linksData = np.load(linksFile)
        minData = np.load(minFile)
        lenData = np.load(lenFile)
        for i in range(repetions):
            tpl = alg.algorithm(noises[i], dataset)
            best_links[i] = tpl[0]
            min_layers[i] = tpl[1]
            lenghts[i] = tpl[2]
        
        linksData = np.vstack((linksData, best_links))
        minData = np.vstack((minData, min_layers))
        lenData = np.vstack((lenData, lenghts))
        
        np.save(linksFile,linksData)
        np.save(minFile,minData)
        np.save(lenFile,lenData)

if __name__ == "__main__":
    funz()



