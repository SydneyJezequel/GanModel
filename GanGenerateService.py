import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from GanBO import Generator






class GenerateService:
    """ Service qui gènère des images """





    """ ************************ Attributs ************************ """

    batch_size = 128
    z_dim = 200
    device = 'cpu'
    root_path = './config/'
    weights_path = 'config/G-latest.pkl'
    gen = Generator(z_dim).to(device)





    """ ************************ Constructeur ************************ """

    _instance = None

    def __new__(cls):
        """ Constructeur """
        if cls._instance is None:
            cls._instance = super(GenerateService, cls).__new__(cls)
            # Initialisation des attributs de classe ici
            cls._instance._batch_size = 128
            cls._instance._z_dim = 200
            cls._instance._device = 'cpu'
            cls._instance._root_path = './config/'
            cls._instance._weights_path = 'config/G-latest.pkl'
            cls._instance._gen = Generator(cls._instance._z_dim).to(cls._instance._device)
        return cls._instance





    """ ************************ Getter / Setter ************************ """

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    # Getter et Setter pour z_dim
    @property
    def z_dim(self):
        return self._z_dim

    @z_dim.setter
    def z_dim(self, value):
        self._z_dim = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_device):
        self._device = new_device

    @property
    def root_path(self):
        return self._root_path

    @property
    def weights_path(self):
        return self._weights_path

    @weights_path.setter
    def weights_path(self, new_weights_path):
        self._weights_path = new_weights_path

    @property
    def gen(self):
        return self._gen

    @gen.setter
    def gen(self, new_gen):
        self._gen = new_gen





    """ ************************ Méthodes ************************ """

    def generate_gan_pictures(self):
        """ Méthode pour générer des images """
        noise = self.gen_noise(self.batch_size, self.z_dim)
        self.gen.eval()
        fake = self.gen(noise)
        self.show(fake)



    def gen_noise(self, num, z_dim, device='cpu'):  # num : Nombre d'exemple à générer. / Dimension de l'espace latent / dispositif ou placer le Tenseur de bruit.
        """ Méthode qui Génère le bruit aléatoire à partir d'une distribution normale """
        return torch.randn(num, z_dim, device=device)  # 128 x 200     # torch.randn() : génère des nombres aléatoires .



    def show(self, tensor, num=25, wandbactive=0, name=''):
        """ Méthode pour afficher une image """
        data = tensor.detach().cpu()
        grid = make_grid(data[:num], nrow=5).permute(1,2,0)
        plt.imshow(grid.clip(0,1))
        plt.show()



    def load_model(self, name):
        """ Méthode qui initialise le Générateur et charge les paramètres du Modèle """
        checkpoint = torch.load(f"{self.root_path}G-{name}.pkl")
        self.gen.load_state_dict(checkpoint['model_state_dict'])

