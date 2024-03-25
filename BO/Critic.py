from torch import nn






class Critic(nn.Module):
    """ Classe du Critique """



    def __init__(self, d_dim=16):
        """ Constructeur """
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



    def forward(self, image):
        """ Méthode qui envoie des données aléatoires dans le Critic """
        # image: 128 x 3 x 128 x 128
        crit_pred = self.crit(image)  # 128 x 1 x 1 x 1     /    Passe l'image dans la couche de neurones pour générer une prédiction.
        return crit_pred.view(len(crit_pred), -1)  ## 128 x 1   /    Redimensionne le tensor crit_pred en une taille : batch_size, 1.
        # batch_size, 1 est la taille utilisé pour les scores binaires.

