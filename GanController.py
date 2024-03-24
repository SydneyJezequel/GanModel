from fastapi import FastAPI
from service import GanGenerateService, GanTrainService
from BO.Hyperparameters import Hyperparameters






""" **************************************** Commande pour démarrer l'application **************************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn GanController:app --reload --workers 1 --host 0.0.0.0 --port 8010






""" **************************************** Chargement de l'Api **************************************** """

app = FastAPI()






""" **************************************** Api de test **************************************** """

@app.get("/ping")
async def pong():
    """ Api de test """
    return {"ping": "pong!"}






""" **************************************** Controllers **************************************** """

@app.get("/generate-faces")
async def generate_gan_pictures_controller():
    """ Exécution du modèle GAN """
    generate_service_instance = GanGenerateService.GenerateService()
    generate_service_instance.load_model('latest')
    generate_service_instance.generate_gan_pictures()






""" **************************************** Entrainement du modèle GAN **************************************** """

@app.post("/train-gan-model", status_code=200)
def train_gan_model(payload: Hyperparameters):
    """ Controller qui lance l'entrainement du modèle Gan """
    print("Logs Hyperparamètres : ")
    print(payload.n_epochs)
    print(payload.batch_size)
    print(payload.lr)
    print(payload.z_dim)
    print(payload.device)
    print(payload.show_step)
    print(payload.save_step)
    # Récupération des valeurs envoyées par l'utilisateur :
    n_epochs = payload.n_epochs
    batch_size = payload.batch_size
    lr = payload.lr
    z_dim = payload.z_dim
    device = payload.device
    show_step = payload.show_step
    save_step = payload.save_step
    # Valeur définies par défaut :
    cur_step = 0
    crit_cycles = 5
    gen_losses = []
    crit_losses = []
    # Chargement du service d'entrainement :
    gan_train_service_instance = GanTrainService.GrainTrainService()
    gan_train_service_instance.perform_gan_model_training(n_epochs, batch_size, lr, z_dim, device, cur_step, crit_cycles, gen_losses, crit_losses, show_step, save_step)

