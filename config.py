# Chemins et variables communs aux fichiers GanGenerateService et GanTrainService :
default_device = 'cpu'


# Chemins et variables du fichier GanGenerateService :
root_path = '../config/'
weights_path = '../config/G-latest.pkl'
model_file = './config/G-latest.pkl'


# Chemins et variables du fichier GanTrainService :
generator_parameters_file = "./data/G-latest.pkl"
critic_parameters_file = "./data/C-latest.pkl"
destination_folder = "/Users/sjezequel/Desktop"
wandb_key = '74a98cfd8dce6ac68b261d2789b794207700b868'
save_checkpoint_name = 'latest'
data_root_path = '../data/'
dataset_celeba_path = './data/celeba'
dataset_download_path = './data/celeba/archive.zip'
dataset_celeba_img_path = '../data/celeba/img_align_celeba'
