# *************************************** Librairies *************************************** #
import torch, torchvision, os, PIL
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image









# *************************************** Classe du Générateur *************************************** #
class Generator(nn.Module):
    # Constructeur : Méthode d'initialisation du Générateur :
    def __init__(self, z_dim=64, d_dim=16):
        super(Generator, self).__init__()

        # Vecteur de bruit :
        self.z_dim = z_dim

        # Couches de neurones :
        # Chaque couche est un enchainements d'opérations : Convolution transposée + Normalisation + Activation ReLU.
        self.gen = nn.Sequential( # Créer un Conteneur Séquentiel qui transforme un vecteur latent en une image de taille 128x128 pixels.

            ## ConvTranspose2d: In channels, out_channels, kernel_size, stride=1, padding=0
            ## Calculating new width and height: (n-1)*stride -2*padding +ks
            ## n = width or height
            ## ks = kernel size
            ## Nous commencons avec une 1x1 image avec z_dim number de channel (200)

            # La diminution du nombre de canaux au fur et à mesure des couches permet
            # de générer les caractéristiques de hauts niveau. Puis de les réduire pour
            # correspondre à l'image de sortie.

            # Couche de convolution 1 : Génère les caractéristiques de haut niveau.
            nn.ConvTranspose2d(z_dim, d_dim * 32, 4, 1, 0), # 4x4 (ch:200, 512) Inverse convolution classique (Arandit l'entrée au lieu de la réduire).
            nn.BatchNorm2d(d_dim * 32),  # Normalise les activations de chaque canal pour stabiliser l'entrainement.
            nn.ReLU(True),  # Introduit de la non Linéarité pour capturer des phénomènes complexes. Remplace toutes les valeurs négatives par 0.

            # La Réduction du nombre de dimensions (ou canaux) en sortie par rapport à l'entrée
            # permet au modèle  de capturer des caractéristiques de plus en plus complexes.

            # Couche de convolution 2 : Réduction du nombre de canaux, augmentation de la taille spatiale.
            nn.ConvTranspose2d(d_dim * 32, d_dim * 16, 4, 2, 1),  ## 8x8 (ch:512, 256)
            nn.BatchNorm2d(d_dim * 16),     # Normalisation des activations.
            nn.ReLU(True),  # Introduit de la non Linéarité pour capturer des phénomènes complexes. Remplace toutes les valeurs négatives par 0.

            # Couche de convolution 3 : Réduction du nombre de canaux, augmentation de la taille spatiale.
            nn.ConvTranspose2d(d_dim * 16, d_dim * 8, 4, 2, 1),  ## 16x16 (ch:256, 128)
            nn.BatchNorm2d(d_dim * 8),      # Normalisation des activations.
            nn.ReLU(True),  # Introduit de la non Linéarité pour capturer des phénomènes complexes. Remplace toutes les valeurs négatives par 0.

            # Couche de convolution 4 : Réduction du nombre de canaux, augmentation de la taille spatiale.
            nn.ConvTranspose2d(d_dim * 8, d_dim * 4, 4, 2, 1),  ## 32x32 (ch:128, 64)
            nn.BatchNorm2d(d_dim * 4),      # Normalisation des activations.
            nn.ReLU(True),  # Introduit de la non Linéarité pour capturer des phénomènes complexes. Remplace toutes les valeurs négatives par 0.

            # Couche de convolution 5 : Réduction du nombre de canaux, augmentation de la taille spatiale.
            nn.ConvTranspose2d(d_dim * 4, d_dim * 2, 4, 2, 1),  ## 64x64 (ch:64, 32)
            nn.BatchNorm2d(d_dim * 2),      # Normalisation des activations.
            nn.ReLU(True),  # Introduit de la non Linéarité pour capturer des phénomènes complexes. Remplace toutes les valeurs négatives par 0.

            # Couche de convolution 6 : Dernière couche, génère l'image finale avec 3 canaux
            nn.ConvTranspose2d(d_dim * 2, 3, 4, 2, 1),  ## 128x128 (ch:32, 3)
            nn.Tanh()  ### Produit un résultat entre -1 et 1 / Génère l'image.
        )

    # Définit le passage avant du Générateur :
    def forward(self, noise):  # self : Couche de Neurones / noise : Tenseur de bruits.
        # Remodèle le Tenseur de bruits en entrée et le passe à travers les couches définies (nn.Sequential()
        x = noise.view(len(noise), self.z_dim, 1, 1)  # 128 (taille dernière couche) x 200 (canaux première couche) x 1 x 1
        return self.gen(x)  # Renvoie l'image générées.









# *************************************** Classe du Critique *************************************** #
class Critic(nn.Module):
    # Constructeur :  Méthode d'initialisation du Générateur :
    def __init__(self, d_dim=16):
        super(Critic, self).__init__()

        # Couches de neurones :
        # Chaque couche est un enchainements d'opérations : Convolution + Normalisation + Activation LeakyReLU.
        self.crit = nn.Sequential(        # Créer un Conteneur Séquentiel qui identifie si l'image est vraie ou fausse.

            # Conv2d: in_channels, out_channels, kernel_size; stride=1, padding=0
            ## New width and height:           # (n+2*pad-ks)//stride +1
            # (n+2*pad-ks)//stride +1 = (128+2*1-4)//2+1=64x64 (ch: 3;16)

            # L'augmentation du nombre de dimensions (ou canaux) en sortie par rapport à l'entrée
            # permet au modèle  de capturer des caractéristiques de plus en plus complexes.

            # Couche de convolution 1 :
            nn.Conv2d(3, d_dim, 4, 2, 1), # 3 canaux en entrées (image RGB) / in_channels, out_channels, kernel_size, stride, padding
            nn.InstanceNorm2d(d_dim),    # Normalisation des activations.
            nn.LeakyReLU(0.2),            # Rétropropage les gradients même pour les valeurs de sortie négatives.

            # Couche de convolution 2 :
            nn.Conv2d(d_dim, d_dim * 2, 4, 2, 1), # 32x32 (ch: 16, 32) / in_channels, out_channels, kernel_size, stride, padding
            nn.InstanceNorm2d(d_dim * 2),  # Normalisation des activations.
            nn.LeakyReLU(0.2),              # Rétropropage les gradients même pour les valeurs de sortie négatives.

            # Couche de convolution 3 :
            nn.Conv2d(d_dim * 2, d_dim * 4, 4, 2, 1), # 16x16 (ch: 32, 64) / in_channels, out_channels, kernel_size, stride, padding
            nn.InstanceNorm2d(d_dim * 4),      # Normalisation des activations.
            nn.LeakyReLU(0.2),                  # Rétropropage les gradients même pour les valeurs de sortie négatives.

            # Couche de convolution 4 :
            nn.Conv2d(d_dim * 4, d_dim * 8, 4, 2, 1), # 8x8 (ch: 64, 128) / in_channels, out_channels, kernel_size, stride, padding
            nn.InstanceNorm2d(d_dim * 8),      # Normalisation des activations.
            nn.LeakyReLU(0.2),                  # Rétropropage les gradients même pour les valeurs de sortie négatives.

            # Couche de convolution 5 :
            nn.Conv2d(d_dim * 8, d_dim * 16, 4, 2, 1), # 4x4 (ch: 128, 256) / in_channels, out_channels, kernel_size, stride, padding
            nn.InstanceNorm2d(d_dim * 16),     # Normalisation des activations.
            nn.LeakyReLU(0.2),                  # Rétropropage les gradients même pour les valeurs de sortie négatives.

            # Le out_channels = 1 : Cela fait de cette couche une tâche de classification binaire :
            nn.Conv2d(d_dim * 16, 1, 4, 1, 0),  ## (n+2*pad-ks)//stride +1 = (4+2*0-4)// 1+1 = 1X1 (ch: 256, 1)
        )

    # Applique le modèle discriminatoire à une image en entrée :
    def forward(self, image):
        # image: 128 x 3 x 128 x 128
        crit_pred = self.crit(image)  # 128 x 1 x 1 x 1     /    Passe l'image dans la couche de neurones pour générer une prédiction.
        return crit_pred.view(len(crit_pred), -1)  ## 128 x 1   /    Redimensionne le tensor crit_pred en une taille : batch_size, 1.
        # batch_size, 1 est la taille utilisé pour les scores binaires.








# *************************************** Classe du DataSet *************************************** #
class DataSet(Dataset):                 # Déclare une classe DataSet qui hérite de la classe Dataset de Pytorch.

    # Constructeur de classe :
    def __init__(self, path, size=128, lim=10000):
       self.sizes = [size, size]   # Initialise une liste size qui représente les dimensions souhaitées pour redimensionner les images (128x128).
       items, labels = [], []    # Initialise deux listes vides.
       # Chargement des données du fichier :
       for data in os.listdir(path)[:lim]: # Parcourt les fichiers du répertoire spécifié par path jusqu'à la limite lim.
            item = os.path.join(path, data) # Concatène le chemin du répertoire path + le nom du fichier data pour former le chemin complet de l'image.
            items.append(item) # Ajout du chemin complet de l'image à la liste items.
            labels.append(data) # Ajoute le nom du fichier (étiquette) à la liste labels.
       self.items = items
       self.labels = labels

    # Méthode qui renvoie la longueur de l'ensemble de données :
    def __len__(self):
        return len(self.items)

    # Méthode qui renvoie un élément de l'ensemble de données à l'index idx.:
    def __getitem__(self, idx):
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








