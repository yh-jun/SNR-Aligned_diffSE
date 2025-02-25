import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")    
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
          parser_.add_argument("--modeltype", type=str, choices=["bbed", "sebridge", 'sebridge_v2', 'sebridge_v3'], default="bbed")
          parser_.add_argument("--snr_conditioned", type=str, choices=["false", "true", "fixed"], default="false")
          parser_.add_argument("--fixed_snr", type=float, default=1.0)

          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     model_type = temp_args.modeltype
     snr_conditioned = temp_args.snr_conditioned
     fixed_snr = temp_args.fixed_snr
     parser = pl.Trainer.add_argparse_args(parser)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     sigma_max = args.sigma_max
     transform_type = args.transform_type

     # Initialize logger, trainer, model, datamodule

     model = ScoreModel(
     backbone=args.backbone, sde=args.sde, model_type=model_type, snr_conditioned=snr_conditioned, data_module_cls=data_module_cls, fixed_snr=fixed_snr,
     **{
          **vars(arg_groups['ScoreModel']),
          **vars(arg_groups['SDE']),
          **vars(arg_groups['Backbone']),
          **vars(arg_groups['DataModule'])
     }
     )
     
     if snr_conditioned == 'fixed' or snr_conditioned == 'true':
          experiment_name = '{0}_{1}{2}_{3}'.format(model_type, snr_conditioned, fixed_snr, sigma_max)
     else:
          experiment_name = '{0}_{1}_{2}_{3}'.format(model_type, snr_conditioned, sigma_max, transform_type)

     if not args.nolog:
          #this needs to be changed accordingly to your wandb settings
          logger = WandbLogger(project="my project", entity = 'my name', log_model=True, save_dir="logs")
          logger.experiment.log_code(".")
          savedir_ck = f'./savedir/{experiment_name}' #change your folder, where to save files
          if not os.path.isdir(savedir_ck):
               os.makedirs(os.path.join(savedir_ck))
     else:
          logger = None




     # Set up callbacks for logger
     if args.num_eval_files and logger != None:
          callbacks = [ModelCheckpoint(dirpath=savedir_ck, save_last=True, filename='{epoch}-last')]
          checkpoint_callback_last = ModelCheckpoint(dirpath=savedir_ck,
               save_last=True, filename='{epoch}-last')
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=savedir_ck,
               save_top_k=10, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=savedir_ck,
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks = [checkpoint_callback_last, checkpoint_callback_pesq,
               checkpoint_callback_si_sdr]
     # Initialize the Trainer and the DataModule
     if logger != None:
          trainer = pl.Trainer.from_argparse_args(
               arg_groups['pl.Trainer'],
               strategy=DDPPlugin(find_unused_parameters=False), logger=logger,
               log_every_n_steps=10, num_sanity_val_steps=0,
               callbacks=callbacks
          )
     else:
          trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPPlugin(find_unused_parameters=False),
          log_every_n_steps=10, num_sanity_val_steps=0
     )

     # Train model
     checkpoint_file = None # If you want to resume training, input the path of checkpoint file here.

     if checkpoint_file != None:
          trainer.fit(model, ckpt_path=checkpoint_file)
     else:
          trainer.fit(model)
