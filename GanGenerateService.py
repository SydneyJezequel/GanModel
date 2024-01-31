#  **************************** Bibliothèques et Imports *********************************** #
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from GanBO import Generator







# ************************************ Hyperparamètres ************************************ #
batch_size=128
z_dim=200
device= 'cpu'






# ************************************ Attributs ************************************ #

root_path='./config/'
gen = Generator(z_dim).to(device)
weights_path = 'config/G-latest.pkl'






# ************************************ Méthodes ************************************ #



# Méthode qui Génère le bruit aléatoire à partir d'une distribution normale :
def gen_noise(num, z_dim, device='cpu'):  # num : Nombre d'exemple à générer. / Dimension de l'espace latent / dispositif ou placer le Tenseur de bruit.
    return torch.randn(num, z_dim, device=device)  # 128 x 200     # torch.randn() : génère des nombres aléatoires .



# Méthode pour afficher une image :
def show(tensor, num=25, wandbactive=0, name=''):
  data = tensor.detach().cpu()
  grid = make_grid(data[:num], nrow=5).permute(1,2,0)
  plt.imshow(grid.clip(0,1))
  plt.show()



# Méthode qui Initialise le Générateur et charge les paramètres du Modèle :
def load_model(name):
    checkpoint = torch.load(f"{root_path}G-{name}.pkl")
    gen.load_state_dict(checkpoint['model_state_dict'])



# Méthode pour générer des images :
def generate_gan_pictures():
    noise = gen_noise(batch_size, z_dim)
    gen.eval()
    fake = gen(noise)
    show(fake)






# ************************************ Démarrage de l'application ************************************ #

# Chargement des poids du modèle à partir du fichier
load_model('latest')  # Assurez-vous d'utiliser le bon nom ici (dans votre exemple, vous avez utilisé 'latest')
print("Les poids du modèle ont été chargés avec succès.")