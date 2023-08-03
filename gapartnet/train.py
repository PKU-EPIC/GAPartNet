from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
import torch
import wandb
torch.set_float32_matmul_precision('medium')
def log_name(config):
    # model
    model_str = ""
    if config["model"]["init_args"]["backbone_type"] == "SparseUNet":
        model_str += "SU"
    elif config["model"]["init_args"]["backbone_type"] == "PointNet":
        model_str += "PN"
    else:
        raise NotImplementedError(f"backbone type {config['model']['init_args']['backbone_type']} not implemented")
    
    model_str += "_"
    
    if config["model"]["init_args"]["use_sem_focal_loss"]:
        model_str += "T"
    else:
        model_str += "F"
    if config["model"]["init_args"]["use_sem_dice_loss"]:
        model_str += "T"
    else:
        model_str += "F"
    
    # data
    data_str = ""
    data_str += "BS" + str(config["data"]["init_args"]["train_batch_size"]) + "_"
    data_str += "Aug" + \
        ""+str(config["data"]["init_args"]["pos_jitter"]) +\
        "-"+str(config["data"]["init_args"]["color_jitter"]) +\
        "-"+str(config["data"]["init_args"]["flip_prob"]) +\
        "-"+str(config["data"]["init_args"]["rotate_prob"])
    
    # time
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%m-%d-%H-%M")
    return model_str, data_str, time_str

class CustomCLI(LightningCLI):
    def before_fit(self):
        # Use the parsed arguments to create a name
        if self.config["fit"]["model"]["init_args"]["debug"] == False:
            model_str, data_str, time_str = log_name(self.config["fit"])
            self.trainer.logger = WandbLogger(
                save_dir = "wandb",
                project = "perception",
                entity = "haoran-geng",
                group = "train_new",
                name = model_str + "_" + data_str + "_" + time_str,
                notes = "GAPartNet",
                tags = ["GAPartNet", "score", "npcs"],
                save_code = True,
                mode = "online",
            )
        else:
            print("Debugging, not using wandb logger")

def main():
    _ = CustomCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=233,
        save_config_kwargs={"overwrite": True},
    )
    
if __name__ == "__main__":
    main()
