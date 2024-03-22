import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import zipfile
import wandb
from GanBO import Generator, Critic, DataSet
import os
from torch.utils.data import DataLoader
import shutil






class GrainTrainService :
    """ Classe d'entrainement du modèle Gan """





    """ ************************ Attributs ************************ """

    root_path='./data/'
    wandbact = 1
    lr = 1e-4
    z_dim = 100
    device = 'cpu'

    # Générateur et Critic :
    gen = Generator(z_dim).to(device)
    crit = Critic().to(device)

    # Optimiseur Adam du Générateur et du Critic :
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))

    # Chemins des fichiers de paramètres des Générateur et Critic :
    fichier_parametres_generateur = "./data/G-latest.pkl"
    fichier_parametres_critic = "./data/C-latest.pkl"
    dossier_destination = "/Users/sjezequel/Desktop"





    """ ************************ Constructeur ************************ """

    _instance = None

    def __new__(cls, z_dim=200, device='cpu', lr=0.0002, root_path='./data/', wandbact=1):
        """ Constructeur """
        if cls._instance is None:
            cls._instance = super(GrainTrainService, cls).__new__(cls)
            # Initialisation des attributs de classe ici
            cls._instance._z_dim = z_dim
            cls._instance._device = device
            cls._instance._lr = lr
            cls._instance._root_path = root_path
            cls._instance._wandbact = wandbact
            # Générateur et Critic :
            cls._instance._gen = Generator(cls._instance._z_dim).to(cls._instance._device)
            cls._instance._crit = Critic().to(cls._instance._device)
            # Optimiseur Adam du Générateur et du Critic :
            cls._instance._gen_opt = torch.optim.Adam(cls._instance._gen.parameters(), lr=cls._instance._lr, betas=(0.5, 0.9))
            cls._instance._crit_opt = torch.optim.Adam(cls._instance._crit.parameters(), lr=cls._instance._lr, betas=(0.5, 0.9))
        return cls._instance





    """ ************************ Getter / Setter ************************ """

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def device(self):
        return self._device

    @property
    def lr(self):
        return self._lr

    @property
    def root_path(self):
        return self._root_path

    @property
    def wandbact(self):
        return self._wandbact

    @property
    def gen(self):
        return self._gen

    @property
    def crit(self):
        return self._crit

    @property
    def gen_opt(self):
        return self._gen_opt

    @property
    def crit_opt(self):
        return self._crit_opt





    """ ************************ Méthodes ************************ """

    def perform_gan_model_training(self, n_epochs, batch_size, lr, z_dim, device, cur_step, crit_cycles, gen_losses, crit_losses, show_step, save_step):
        """ Méthode qui exécute l'entrainement """
        # Logs des paramètres :
        print("Paramètres du service : batch_size : ", n_epochs, " batch_size : ", batch_size, " lr: ", lr, " dimensions latentes : ", z_dim, " device : ", device, " show_step : ", show_step, " save_step : ", save_step)
        # Chargement du jeu de données :
        dataloader = self.dataset_init(batch_size)
        for epoch in range(n_epochs):  # Boucle qui parcourt les époques d'entraînement.
            for real, _ in tqdm(dataloader):  # Cette boucle parcourt chaque lot de données dans le DataLoader :
                cur_bs = len(real)  # Taille du lot des données d'entrainement réelles (batch size), par exemple, 128
                real = real.to(device)  # Les données réelles (real) sont déplacées sur le dispositif d'entraînement.

                """ 1- Critic """
                mean_crit_loss = 0
                for _ in range(crit_cycles):
                    # Initialisation du Gradient à 0. :
                    self.crit_opt.zero_grad()
                    # Générer des données fausses avec le générateur :
                    noise = self.gen_noise(cur_bs, z_dim)  # Génération du bruit aléatoire.
                    fake = self.gen(noise)  # Image fausse.
                    # Prédiction du critique pour les données fausses et réelles :
                    crit_fake_pred = self.crit(
                        fake.detach())  # Prédiction du critique pour les données générées / .detach() : Détache le Graphique.
                    crit_real_pred = self.crit(real)  # Prédiction du critique pour les données réelles.
                    # Calcul de la pénalité de gradient (gradient penalty) :
                    # Génération d'un facteur d'interpolation :
                    alpha = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)  # 128 x 1 x 1 x 1
                    # Calcul de la pénalité de gradient :
                    gp = self.get_gp(real, fake.detach(), self.crit, alpha)  # fake.detach() pour ne pas maj paramètres générateur.
                    # Calcul de la perte du critique :
                    # Différence entre les prédictions des données fausses et réelles, ainsi que la pénalité de gradient.
                    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp
                    # Mise à jour de la moyenne de la perte du critique :
                    mean_crit_loss += crit_loss.item() / crit_cycles
                    # Rétropropagation de la perte et mise à jour des poids du critique :
                    crit_loss.backward(retain_graph=True)  # Rétropropagation de la perte.
                    self.crit_opt.step()  # Mise à jour des poids du critique avec l'optimiseur du critique
                # Stockage de la perte moyenne du critique pour cette itération :
                crit_losses += [mean_crit_loss]

                """ 2- Generator """
                # Initialisation des gradients du générateur à zéro :
                self.gen_opt.zero_grad()
                # Génération d'une image et évaluation :
                noise = self.gen_noise(cur_bs, z_dim)  # Génération d'un tensor de bruit : taille du lot + dimension du bruit.
                fake = self.gen(noise)  # Génération d'images fausses avec le générateur.
                crit_fake_pred = self.crit(fake)  # Prédiction du critique pour les données générées.
                # Mise à jour des paramètres :
                gen_loss = -crit_fake_pred.mean()  # Calcul perte : Moyenne négatives des prédictions du critique pour les données générées.
                gen_loss.backward()  # Rétropropagation de la perte pour mettre à jour les poids du générateur.
                self.gen_opt.step()  # Mise à jour des poids du générateur en utilisant l'optimiseur du générateur
                # Stockage de la valeur de la perte du générateur pour cette itération :
                gen_losses += [gen_loss.item()]

                """ 3- Statistiques """
                wandb.login(key='74a98cfd8dce6ac68b261d2789b794207700b868')
                experiment_name = wandb.util.generate_id()
                wandb.init(
                    project="wgan",
                    group=experiment_name,
                    config={
                        "optimizer": "adam",
                        "model": "wgan gp",
                        "epoch": "1000",
                        "batch_size": 128
                    })
                wandb.config
                # Vérifie si l'utilisation de Weights & Biases (wandb) est activée :
                if (self.wandbact == 1):
                    wandb.log({'Epoch': epoch, 'Step': cur_step, 'Critic loss': mean_crit_loss,
                               'Gen loss': gen_loss})  # Si oui : Afficher les logs.

                # Sauvegarde des fichiers de paramètres :
                if(cur_step % show_step == 0 and cur_step > 0): # A chaque étape qui correspond à ces conditions.
                    print("Saving checkpoint: ", cur_step, save_step)  # Affichage de l'étape sauvegardé.
                    self.save_checkpoint("latest", epoch)
                    self.deplacer_fichier(self.fichier_parametres_generateur, self.dossier_destination)
                    self.deplacer_fichier(self.fichier_parametres_critic, self.dossier_destination)

                if (cur_step % show_step == 0 and cur_step > 0):
                    self.show(fake, wandbactive=1, name='fake')  # Affiche les images générées
                    self.show(real, wandbactive=1, name='real')  # Affiche les images réelles.

                    # Calcul de la moyenne des pertes du générateur sur les dernières show_step itérations :
                    gen_mean = sum(gen_losses[-show_step:]) / show_step
                    # Calcul de la moyenne des pertes du critique sur les dernières show_step itérations :
                    crit_mean = sum(crit_losses[-show_step:]) / show_step
                    # Affichage des informations d'entraînement :
                    print(f"Epoch: {epoch}: Step {cur_step}: Generator loss : {gen_mean}, critic loss : {crit_mean}")  #
                    # Trace la courbe de la perte du générateur au fil des itérations :
                    plt.plot(
                        range(len(gen_losses)),  # L'axe des x de la courbe.
                        torch.Tensor(gen_losses),  # L'axe des y de la courbe.
                        label="Generator Loss"  # Étiquette de la courbe.
                    )
                    # Trace la Courbe de la perte du critique au fil des itérations :
                    plt.plot(
                        range(len(gen_losses)),  # L'axe des x de la courbe.
                        torch.Tensor(crit_losses),  # L'axe des y de la courbe.
                        label="Critic Loss"  # Étiquette de la courbe.
                    )
                    # Paramétrage final et affichage du graphique :
                    plt.ylim(-1000, 1000)  # Limite l'axe y du graphique entre -1000 et 1000.
                    plt.legend()  # Ajoute une légende au graphique.
                    plt.show()  # Affiche le graphique.
                cur_step += 1  # Incrémente le compteur d'étapes.



    def show(self, tensor, num=25, wandbactive=0, name=''):
        """ Fonction de visualisation """
        data = tensor.detach().cpu()  # Détache le tensor du graphe de calcul
        grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)  # Créé une Grille d'images à partir des premiers éléments du Tensor.
        # optional :   # Permute(1, 2, 0) : Permute les dimensions rendre le Tensor compatible avec l'ordre des canaux.
        if (wandbactive==1):  # Vérifie si option de visualisation sont activées.
            wandb.log({name:wandb.Image(grid.numpy().clip(0, 1))})
        plt.imshow(grid.clip(0, 1))  # Affiche une grille via Matplotlib / clip(0, 1) : Intègre les valeurs du Tensor entre 0 et 1.
        plt.show()  # Affiche la grille d'images.



    def gen_noise(self, num, z_dim, device='cpu'):  # num : Nombre d'exemple à générer. / Dimension de l'espace latent / dispositif ou placer le Tenseur de bruit.
        """ Fonction qui Génère du bruit aléatoire à partir d'une distribution normale """
        return torch.randn(num, z_dim, device=device)  # 128 x 200     # torch.randn() : génère des nombres aléatoires .



    def get_gp(self, real, fake, crit, alpha, gamma=10):
        """ Méthode qui calcule la Pénalité de Gradient """
        # Cette ligne crée des images mixtes en interpolant linéairement entre les images réelles (real) et les images générées (fake) :
        mix_images = real * alpha + fake * (1-alpha)
        # Evalue le critique (crit) sur les images mixtes :
        mix_scores = crit(mix_images) # 128 x 1
        # Calculer le gradient des scores :
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
        gp = gamma * ((gradient_norm-1)**2).mean() # Pénalité de gradient = Moyenne des carrés des différences entre la norme du gradient et 1.
        return gp



    def save_checkpoint(self, name, epoch):
      """ Méthode qui Sauvegarde l'entrainement """
      # Enregistrement de l'état du Générateur :
      torch.save({
          'epoch': epoch,
          'model_state_dict': self.gen.state_dict(),
          'optimizer_state_dict': self.gen_opt.state_dict()
      }, f"{self.root_path}G-{name}.pkl")
      # Enregistrement de l'état du Critique :
      torch.save({
          'epoch': epoch,
          'model_state_dict': self.crit.state_dict(),
          'optimizer_state_dict': self.crit_opt.state_dict()
      }, f"{self.root_path}C-{name}.pkl")
      # Log de bonne exécution :
      print("Saved checkpoint")



    def load_checkpoint(self, name):
      """ Méthode qui charge une sauvegarde de l'entrainement """
      # Chargement de l'état du Générateur :
      checkpoint = torch.load(f"{self.root_path}G-{name}.pkl")
      self.gen.load_state_dict(checkpoint['model_state_dict'])
      self.gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])
      # Chargement de l'état du Critique :
      checkpoint = torch.load(f"{self.root_path}C-{name}.pkl")
      self.crit.load_state_dict(checkpoint['model_state_dict'])
      self.crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])
      print("Loaded checkpoint")



    def dataset_init(self, batch_size):
        """ Méthode qui initialise le dataset """
        path = 'data/celeba'
        download_path = f'{path}/archive.zip'
        if not os.path.exists(path):
            os.makedirs(path)
        with zipfile.ZipFile(download_path, 'r') as ziphandler:
            ziphandler.extractall(path)
        # DataSet :
        data_path = 'data/celeba/img_align_celeba'  # Chemin du fichier.
        # Création d'une instance de la classe DataSet :
        ds = DataSet(data_path, size=128, lim=10000)
        # size-128 : taille réduite de l'image. / lim=10000 : taille réduite de l'image.
        # DataLoader :
        # Instance de DataLoader : Taille du batch (128)    /   shuffle=True -> Les données sont mélangées
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return dataloader



    def deplacer_fichier(self, source, destination):
        """ Méthode qui déplace les fichiers de paramètres des Critic et Généerateur """
        try:
            if os.path.exists(source):
                shutil.move(source, destination)
                print(f"Le fichier a été déplacé avec succès de {source} vers {destination}")
            else:
                print(f"Le fichier {source} n'existe pas.")
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")

