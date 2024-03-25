import torch, torchvision, os, PIL
from torch.utils.data import Dataset
import numpy as np






class DataSet(Dataset):
    """ Classe du Dataset """



    def __init__(self, path, size=128, lim=10000):
       """ Constructeur """
       self.sizes = [size, size]   # Initialise une liste size qui représente les dimensions souhaitées pour redimensionner les images (128x128).
       items, labels = [], []    # Initialise deux listes vides.
       # Chargement des données du fichier :
       for data in os.listdir(path)[:lim]: # Parcourt les fichiers du répertoire spécifié par path jusqu'à la limite lim.
            item = os.path.join(path, data) # Concatène le chemin du répertoire path + le nom du fichier data pour former le chemin complet de l'image.
            items.append(item) # Ajout du chemin complet de l'image à la liste items.
            labels.append(data) # Ajoute le nom du fichier (étiquette) à la liste labels.
       self.items = items
       self.labels = labels



    def __len__(self):
        """ Méthode qui renvoie la longueur de l'ensemble de données """
        return len(self.items)



    def __getitem__(self, idx):
        """ Méthode qui renvoie un élément de l'ensemble de données à l'index idx """
        data = PIL.Image.open(self.items[idx]).convert('RGB') # (178, 218)  /  Ouvre l'image à l'index spécifié avec PIL et la convertit en mode RGB.
        # Redimensionne l'image à l'aide de la transformation Resize de torchvision et la convertit en tableau NumPy pour faire des opérations dessus :
        data = np.asarray(torchvision.transforms.Resize(self.sizes)(data)) # 128 x 128 x 3
        # Transpose les dimensions de l'image pour qu'elles correspondent à l'ordre attendu par PyTorch (C x H x W).
        # Puis convertit le tableau en type de données float32 :
        data = np.transpose(data, (2,0,1)).astype(np.float32, copy=False) # 3 x 128 x 128 # from 0 to 255
        # Convertit le tableau NumPy en un tenseur PyTorch + Normalise les valeurs des pixels de l'intervalle [0, 255] à l'intervalle [0, 1] :
        data = torch.from_numpy(data).div(255) # from 0 to 1
        # Renvoie le couple (données, étiquette) correspondant à l'index spécifié.
        return data, self.labels[idx] # Les données sont les pixels de l'image, et l'étiquette est le nom du fichier.

