# ********************* Librairies ********************* #
import torch, torchvision, os, PIL, pdb  # Librairies importées.
from torch import nn  # Module de la librairie Pytorch pour définir les réseaux de neurones.
from torch.utils.data import Dataset  # Créer des ensembles de données.
from torch.utils.data import DataLoader  # Charge les données.
from torchvision import transforms  # Fournit des transformations d'images (Normalisation, Redimensionnement).
from torchvision.utils import make_grid  # Créer une grille d'images à partir d'un lot d'images.
from tqdm.auto import tqdm  # Ajoute Barre de progression sur les boucles.
import numpy as np  # Bibliothèque de Calcul Numérique.
from PIL import Image  # Module de la librairie PIL pour manipuler des images.
import matplotlib.pyplot as plt  # Créer des graphiques et des visualisations.
import wandb  # Permet de suivre et Visualiser l'entrainement.







# ********************* Fonction de visualisation ********************* #
# Cette fonction extrait 25 éléments du Tensor et les affiche sous forme de Grille.
def show(tensor, num=25, wandbactive=0, name=''):  #
    data = tensor.detach().cpu()  # Détache le tensor du graphe de calcul
    grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)  # Créé une Grille d'images à partir des premiers éléments du Tensor.
    # optional :   # Permute(1, 2, 0) : Permute les dimensions rendre le Tensor compatible avec l'ordre des canaux.
    if (wandbactive==1):  # Vérifie si option de visualisation sont activées.
        wandb.log({name:wandb.Image(grid.numpy().clip(0, 1))})
    plt.imshow(grid.clip(0, 1))  # Affiche une grille via Matplotlib / clip(0, 1) : Intègre les valeurs du Tensor entre 0 et 1.
    plt.show()  # Affiche la grille d'images.






# ********************* Hyperparamètres et paramètres principaux ********************* #

n_epochs = 10000    # Nombre d'époques d'entraînement.
batch_size = 128    # Taille du lot (batch) d'échantillons utilisés lors de chaque itération d'entraînement.
lr = 1e-4           # Taux d'apprentissage : Contrôle la taille des pas lors de l'optimisation. Définit la vitesse de convergence du modèle.
z_dim = 200         # Dimension de l'espace latent.
device = 'cpu'  # device = 'cuda'    # GPU  # Dispositif sur lequel le modèle est entraîné
cur_step = 0        # Étape courante dans l'entraînement. --> Suivi du progrès de l'entrainement.
crit_cycles = 5     # Spécifie le nombre d'itérations d'entraînement du critique pour chaque itération d'entraînement du générateur.
gen_losses = []     # Stocke les pertes du générateur à chaque itération d'entraînement.
crit_losses = []    # Stocke les pertes du critique à chaque itération d'entraînement.
show_step = 35      # Fréquence à laquelle la méthode show doit être appelée pour afficher des échantillons générés.
save_step = 35      # Fréquence à laquelle les modèles ou les poids du modèle sont sauvegardés pendant l'entraînement.
wandbact = 1        # Track les statistiques des poids et biais (paramètre optionnel). # Paramètre binaire (1 ou 0)
# pour activer ou désactiver le suivi des statistiques des poids et des biais<;






# ********************* Optionnel ********************* #
"""
!pip install wandb - qqq
wandb.login(key='')
%%capture
"""
wandb.login(key='74a98cfd8dce6ac68b261d2789b794207700b868')
experiment_name = wandb.util.generate_id()
myrun = wandb.init(
    project="wgan",
    group=experiment_name,
    config={
        "optimizer": "adam",
        "model": "wgan gp",
        "epoch": "1000",
        "batch_size": 128
    })

config = wandb.config
print(experiment_name)






# ********************* Classe du Générateur ********************* #
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

# Génère du bruit aléatoire à partir d'une distribution normale :
def gen_noise(num, z_dim, device='cpu'):  # num : Nombre d'exemple à générer. / Dimension de l'espace latent / dispositif ou placer le Tenseur de bruit.
    return torch.randn(num, z_dim, device=device)  # 128 x 200     # torch.randn() : génère des nombres aléatoires .






# ********************* Classe du Critique ********************* #
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





# ********************* Optionnel : Initialiser le poids des différentes façon ********************* #
# Cette méthode initialiser les poids et les biais des couches de convolution :
def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal(m.weight, 0.0, 0.02) # Initialise les poids de la couche avec des valeurs aléatoires.
        torch.nn.init.constant(m.bias, 0)         # Initialise les biais de la couche avec des valeurs aléatoires.

    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal(m.weight, 0.0, 0.02) # Initialise les poids de la couche avec des valeurs aléatoires.
        torch.nn.init.constant(m.bias, 0)         # Initialise les biais de la couche avec des valeurs aléatoires.

# gen = gen.apply(init_weight)
# crit = crit.apply(init_weight)






# ********************* Chargement du DataSet ********************* #
import gdown, zipfile                                           # Les modules gdown et zipfile télécharge des fichiers zip dans Google Drive et les manipule.
# import gdown, zipfile
# Si le lien ne marche pas : Download le dataset sur : https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.
# Sinon voir les autres sources dans la vidéo 46 du cours : "Code Review" :
# url = 'https://drive.google.com/uc?id=icNIac61PSA_LqDFYFUeyaQYekYPc75NH'    # Url du fichier a télécharger dans GoogleDrive.
path= 'data/celeba'                                                         # Chemin local ou le fichier sera download et sauvegardé.
download_path=f'{path}/archive.zip'                                # Définit le chemin complet du fichier zip téléchargé avec le path définit.

if not os.path.exists(path):                                                # Si le répertoire local du path n'existe pas :
    os.makedirs(path)                                                       # Création du répertoire du path et des répertoires intermédiaires.

# gdown.download(url, download_path, quiet=False) # Télécharge le fichier depuis l'URL et le charge dans downloadpath.
                                                # quiet=False affiche la progression du téléchargement.
with zipfile.ZipFile(download_path, 'r') as ziphandler: # Ouvre le fichier zip téléchargé en mode lecture ('r').
    ziphandler.extractall(path)                         # Extrait tous les fichiers du fichier zip dans le répertoire local spécifié par path.






# ********************* Classe du DataSet ********************* #
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






# ********************* Manipulation des classes créées ********************* #
## DataSet :
data_path = 'data/celeba/img_align_celeba'         # Chemin du fichier.
# Création d'une instance de la classe DataSet :
ds = DataSet(data_path, size=128, lim=10000)
# size-128 : taille réduite de l'image. / lim=10000 : taille réduite de l'image.

## DataLoader:
# Instance de DataLoader : Taille du batch (128)    /   shuffle=True -> Les données sont mélangées
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

## Models :
gen = Generator(z_dim).to(device) # Instance du Generateur (avec les dimensions de l'espace latent) affecté au device().
crit = Critic().to(device) # Instance du Generateur affecté au device().

## Optimizers :
# Optimiseur Adam pour le générateur (gen), avec taux d'apprentissage lr et paramètres bêtas spécifiés.
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9)) # 0.5 : coeff 1er moment. / 0.9 coeff 2e moment
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))
"""
Des Betas plus proches de 1.0 donneront plus de poids aux mises à jour récentes par rapport aux anciennes.
Des plus petits donnent plus de stabilité. Trop petites peuvent ralentir l'adaptation du modèle aux nouveaux gradients.
"""
#Initializations :
##gen=gen.apply(init_weights)
##crit=crit.apply(init_weights)

#wandb optional :
if (wandbact==1): # Si la condition est vraie, cela signifie que WandB est activé
    wandb.watch(gen, log_freq=100) # Surveille le Generateur avec 100 journalisation.
    wandb.watch(crit, log_freq=100) # Surveille le Critic avec 100 journalisation.
# "wandbact" contrôle l'activation ou la désactivation de l'intégration avec la bibliothèque Weights and Biases (WandB).
x, y = next(iter(dataloader)) # Itérable du Dataloader qui parcourt les lots de données : X entrées / Y : étiquettes.
show(x) # Visualisation des données.






# ********************* Cacul de la Pénalité de Gradient ********************* #
def get_gp(real, fake, crit, alpha, gamma=10):
    # Cette ligne crée des images mixtes en interpolant linéairement entre les images réelles (real) et les images générées (fake) :
    mix_images = real * alpha + fake * (1-alpha)
    # Evalue le critique (crit) sur les images mixtes:
    mix_scores = crit(mix_images) # 128 x 1
    # calculer le gradient des scores :
    gradient = torch.autograd.grad(
        inputs=mix_images,                        # Images mixtes en entrées.
        outputs=mix_scores,                       # Scores en sorties.
        grad_outputs=torch.ones_like(mix_scores), # Gradients de sortie de même forme que mix_scores.
        retain_graph=True,                        # Indique à PyTorch de conserver le graph pour d'autres calculs.
        create_graph=True,
    )[0] # 128 x 3 x 128 x 128                    # Renvoi le premier élément du Tuple (Le Gradient lui-même).

    # Modifie la forme du tensor gradient en le redimensionnant.
    # La première dimension reste la même (c'est la longueur de gradient pour chaque exemple).
    # La deuxième dimension est redimensionnée pour être de taille -1.
    # Chaque ligne du Tenseur représente le gradient d'un exemple :
    gradient = gradient.view(len(gradient), -1) # 128 x 49152
    # Calcule la Norme L2 du gradient pour toutes les lignes du Tenseur (chaque exemple) :
    gradient_norm = gradient.norm(2, dim=1)
    gp = gamma * ((gradient_norm-1)**2).mean()      # Pénalité de gradient = Moyenne des carrés des différences entre la norme du gradient et 1.
    return gp






# ********************* Sauvegarde et Chargement de Checkpoints ********************* #
root_path= './data/'        # Chemin racine où les checkpoints seront sauvegardés.
# Sauvegarder un Checkpoint :
def save_checkpoint(name, epoch): # Fonction pour sauvegarder un checkpoint du modèle génératif et de son optimiseur.
    #  Sauvegarde du modèle génératif (gen) et de son optimiseur (gen_opt) :
    # **************************** LOGS **************************** #
    print("log de sauvegarde Generateur : ")
    print(epoch)
    print(gen.state_dict())
    print(gen_opt.state_dict())
    print("log de sauvegarde Discriminateur : ")
    print(crit.state_dict())
    print(crit.state_dict())
    # **************************** LOGS **************************** #
    torch.save({
        'epoch': epoch,                                 # Valeur de l'époque
        'model_state_dict': gen.state_dict(),           # État du modèle génératif.
        'optimizer_state_dict': gen_opt.state_dict()    # État de l'optimiseur du modèle génératif.
    }, f"{root_path}G-{name}-E{epoch}.plk")                      # Le nom du fichier est G-param entrée.
    # Sauvegarde est effectuée pour un modèle critique :
    torch.save({
        'epoch': epoch,                                 # Valeur de l'époque
        'model_state_dict': crit.state_dict(),          # État du modèle critique.
        'optimizer_state_dict': crit_opt.state_dict()   # État de l'optimiseur du modèle critique.
    }, f"{root_path}C-{name}-E{epoch}.plk")                      # Le nom du fichier est C-param entrée.
    print("Saved checkpoint")                           # Message d'information.

# Charger un Checkpoint :
def load_checkpoint(name, epoch):
    print(f"{root_path}C-{name}-E{epoch}.plk")
    print(f"{root_path}G-{name}-E{epoch}.plk")

    checkpoint = torch.load(f"{root_path}C-{name}-E{epoch}.plk") # Chargement du fichier checkpoint du générateur.
    gen.load_state_dict(checkpoint['model_state_dict']) # Chargement de l'état du modèle.
    gen_opt.load_state_dict(checkpoint['optimizer_state_dict']) # Chargement de l'état de l'optimiseur.

    checkpoint = torch.load(f"{root_path}G-{name}-E{epoch}.plk") # Chargement du fichier checkpoint du critique.
    crit.load_state_dict(checkpoint['model_state_dict']) # Chargement de l'état du modèle.
    crit_opt.load_state_dict(checkpoint['optimizer_state_dict']) # Chargement de l'état de l'optimiseur.

    print("Loaded_checkpoint")  # Message d'information.

# Test du Checkpoint :
# save_checkpoint("test")
# load_checkpoint("test")






# ********************* Boucle d'entrainement : Entrainement Critic & Generator ********************* #
# Chargement du checkpoint :
# Charger un Checkpoint (à placer avant la boucle d'entraînement) :
load_checkpoint("latest", 12)
epoch = 12


# Boucle d'entrainement :
for epoch in range(n_epochs):        # Boucle qui parcourt les époques d'entraînement.
    for real, _ in tqdm(dataloader): # Cette boucle parcourt chaque lot de données dans le DataLoader :
        cur_bs = len(real)           # Taille du lot des données d'entrainement réelles (batch size), par exemple, 128
        real = real.to(device)       # Les données réelles (real) sont déplacées sur le dispositif d'entraînement.
        ### 1- Critic :
        mean_crit_loss = 0
        for _ in range(crit_cycles):
            # Initialisation du Gradient à 0.
            crit_opt.zero_grad()
            # Générer des données fausses avec le générateur :
            noise = gen_noise(cur_bs, z_dim)     # Génération du bruit aléatoire.
            fake = gen(noise)                    # Image fausse.
            # Prédiction du critique pour les données fausses et réelles :
            crit_fake_pred = crit(fake.detach()) # Prédiction du critique pour les données générées / .detach() : Détache le Graphique.
            crit_real_pred = crit(real)          # Prédiction du critique pour les données réelles.
            # Calcul de la pénalité de gradient (gradient penalty) :
            # Génération d'un facteur d'interpolation :
            alpha = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True) # 128 x 1 x 1 x 1
            # Calcul de la pénalité de gradient :
            gp = get_gp(real, fake.detach(), crit, alpha) # fake.detach() pour ne pas maj paramètres générateur.
            # Calcul de la perte du critique :
            # Différence entre les prédictions des données fausses et réelles, ainsi que la pénalité de gradient.
            crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp
            # Mise à jour de la moyenne de la perte du critique :
            mean_crit_loss += crit_loss.item() / crit_cycles
            # Rétropropagation de la perte et mise à jour des poids du critique :
            crit_loss.backward(retain_graph=True) # Rétropropagation de la perte.
            crit_opt.step()                       # Mise à jour des poids du critique avec l'optimiseur du critique
        # Stockage de la perte moyenne du critique pour cette itération :
        crit_losses += [mean_crit_loss]

        ### 2- Generator :
        # Initialisation des gradients du générateur à zéro :
        gen_opt.zero_grad()
        # Génération d'une image et évaluation :
        noise = gen_noise(cur_bs, z_dim) # Génération d'un tensor de bruit : taille du lot + dimension du bruit.
        fake = gen(noise)                # Génération d'images fausses avec le générateur.
        crit_fake_pred = crit(fake)      # Prédiction du critique pour les données générées.
        # Mise à jour des paramètres :
        gen_loss = -crit_fake_pred.mean() # Calcul perte : Moyenne négatives des prédictions du critique pour les données générées.
        gen_loss.backward()              # Rétropropagation de la perte pour mettre à jour les poids du générateur.
        gen_opt.step()                   # Mise à jour des poids du générateur en utilisant l'optimiseur du générateur
        # Stockage de la valeur de la perte du générateur pour cette itération :
        gen_losses += [gen_loss.item()]

        ### 3- Statistiques
        if(wandbact==1):       # Vérifie si l'utilisation de Weights & Biases (wandb) est activée
            wandb.log({'Epoch':epoch, 'Step': cur_step, 'Critic loss':mean_crit_loss, 'Gen loss': gen_loss})# Si oui : Afficher les logs.

        # Sauvegarde du checkpoint :
        #if(cur_step % show_step == 0 and cur_step > 0): # A chaque étape qui correspond à ces conditions.
            #print("Saving checkpoint: ", cur_step, save_step)  # Affichage de l'étape sauvegardé.
            # save_checkpoint(f"latest-{epoch}") # Sauvegarde du checkpoint.
        print("Saving checkpoint: ", cur_step, save_step)  # Affichage de l'étape sauvegardé.
        save_checkpoint("latest", epoch)

        if(cur_step % show_step == 0 and cur_step > 0):
            show(fake, wandbactive=1, name='fake')       # Affiche les images générées
            show(real, wandbactive=1, name='real')       # Affiche les images réelles.

            # Calcul de la moyenne des pertes du générateur sur les dernières show_step itérations :
            gen_mean = sum(gen_losses[-show_step:]) / show_step
            # Calcul de la moyenne des pertes du critique sur les dernières show_step itérations :
            crit_mean = sum(crit_losses[-show_step:]) / show_step
            # Affichage des informations d'entraînement :
            print(f"Epoch: {epoch}: Step {cur_step}: Generator loss : {gen_mean}, critic loss : {crit_mean}") #
            # Trace la courbe de la perte du générateur au fil des itérations :
            plt.plot(
                range(len(gen_losses)),     # L'axe des x de la courbe.
                torch.Tensor(gen_losses),   # L'axe des y de la courbe.
                label="Generator Loss"      # Étiquette de la courbe.
            )
            # Trace la Courbe de la perte du critique au fil des itérations :
            plt.plot(
                range(len(gen_losses)),     # L'axe des x de la courbe.
                torch.Tensor(crit_losses),  # L'axe des y de la courbe.
                label="Critic Loss"         # Étiquette de la courbe.
            )
            # Paramétrage final et affichage du graphique :
            plt.ylim(-1000, 1000)   # Limite l'axe y du graphique entre -1000 et 1000.
            plt.legend()            # Ajoute une légende au graphique.
            plt.show()              # Affiche le graphique.
        cur_step += 1               # Incrémente le compteur d'étapes.
























# ********************* Morphing entre 2 points d'un Espace Latent ********************* #











































































































































