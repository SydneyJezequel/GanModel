from pydantic import BaseModel






class Hyperparameters(BaseModel):
    """ Classe des Hyperamètres """
    n_epochs : int
    batch_size : int
    lr : float
    z_dim : int
    device : str
    show_step : int
    save_step : int

