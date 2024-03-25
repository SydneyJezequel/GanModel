from torch import nn






class Generator(nn.Module):
    """ Classe du Générateur """



    def __init__(self, z_dim, d_dim=16):
        """ Constructeur """
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



    def forward(self, noise):  # self : Couche de Neurones / noise : Tenseur de bruits.
        """ Méthode qui envoie des données aléatoires dans le Générateur """
        print("self.z_dim : ", self.z_dim)
        # Remodèle le Tenseur de bruits en entrée et le passe à travers les couches définies (nn.Sequential()
        x = noise.view(len(noise), self.z_dim, 1, 1)  # 128 (taille dernière couche) x 200 (canaux première couche) x 1 x 1
        return self.gen(x)  # Renvoie l'image générées.

