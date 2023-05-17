#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model training script."""

import os
from typing import Any, Dict, Tuple, Type, cast

import hydra
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from torchgeo.datamodules import (
    BigEarthNetDataModule,
    ChesapeakeCVPRDataModule,
    CCMEODataModule,
    COWCCountingDataModule,
    ETCI2021DataModule,
    EuroSATDataModule,
    InriaAerialImageLabelingDataModule,
    LandCoverAIDataModule,
    NAIPChesapeakeDataModule,
    NASAMarineDebrisDataModule,
    OSCDDataModule,
    RESISC45DataModule,
    SEN12MSDataModule,
    So2SatDataModule,
    Spacenet1DataModule,
    TropicalCycloneDataModule,
    UCMercedDataModule,
)
from torchgeo.trainers import (
    BYOLTask,
    ClassificationTask,
    MultiLabelClassificationTask,
    ObjectDetectionTask,
    RegressionTask,
    SemanticSegmentationTask,
)
from torchgeo.trainers.segmentation import BinarySemanticSegmentationTask, MultiClassTransformer

TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[pl.LightningModule], Type[pl.LightningDataModule]]
] = {
    "bigearthnet": (MultiLabelClassificationTask, BigEarthNetDataModule),
    "byol": (BYOLTask, ChesapeakeCVPRDataModule),
    "ccmeo": (BinarySemanticSegmentationTask, CCMEODataModule),
    "ccmeo_multi": (SemanticSegmentationTask, CCMEODataModule),
    "ccmeo_multi_transformer": (MultiClassTransformer, CCMEODataModule),
    "chesapeake_cvpr": (SemanticSegmentationTask, ChesapeakeCVPRDataModule),
    "cowc_counting": (RegressionTask, COWCCountingDataModule),
    "cyclone": (RegressionTask, TropicalCycloneDataModule),
    "eurosat": (ClassificationTask, EuroSATDataModule),
    "etci2021": (SemanticSegmentationTask, ETCI2021DataModule),
    "inria": (SemanticSegmentationTask, InriaAerialImageLabelingDataModule),
    "landcoverai": (SemanticSegmentationTask, LandCoverAIDataModule),
    "naipchesapeake": (SemanticSegmentationTask, NAIPChesapeakeDataModule),
    "nasa_marine_debris": (ObjectDetectionTask, NASAMarineDebrisDataModule),
    "oscd": (SemanticSegmentationTask, OSCDDataModule),
    "resisc45": (ClassificationTask, RESISC45DataModule),
    "sen12ms": (SemanticSegmentationTask, SEN12MSDataModule),
    "so2sat": (ClassificationTask, So2SatDataModule),
    "spacenet1": (BinarySemanticSegmentationTask, Spacenet1DataModule),
    "ucmerced": (ClassificationTask, UCMercedDataModule),
}


@hydra.main(config_path="conf", config_name="ccmeo")
def main(conf: DictConfig) -> None:
    """Main training loop."""
    ######################################
    # Setup output directory
    ######################################
    conf = OmegaConf.create(conf)

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.program.seed)
    
    # checkpoint_path = conf.experiment.model_state_dict
    experiment_name = conf.experiment.name
    task_name = conf.experiment.task
    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError("`program.output_dir` must be a directory")
    os.makedirs(conf.program.output_dir, exist_ok=True)

    experiment_dir = os.path.join(conf.program.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if len(os.listdir(experiment_dir)) > 0:
        if conf.program.overwrite:
            print(
                f"WARNING! The experiment directory, {experiment_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            raise FileExistsError(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, exiting..."
            )

    with open(os.path.join(experiment_dir, "experiment_config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    ######################################
    # Choose task to run based on arguments or configuration
    ######################################
    # Convert the DictConfig into a dictionary so that we can pass as kwargs.
    task_args = cast(Dict[str, Any], OmegaConf.to_object(conf.experiment.module))
    datamodule_args = cast(
        Dict[str, Any], OmegaConf.to_object(conf.experiment.datamodule)
    )

    datamodule: pl.LightningDataModule
    task: pl.LightningModule
    if task_name in TASK_TO_MODULES_MAPPING:
        task_class, datamodule_class = TASK_TO_MODULES_MAPPING[task_name]
        task = task_class(**task_args)
        # if checkpoint_path is not None:
        #     print(f'Transfer learning from: {checkpoint_path}')
        #     task = task.load_model(path=checkpoint_path)
        datamodule = datamodule_class(**datamodule_args)
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    ######################################
    # Setup trainer
    ######################################
    #tb_logger = pl_loggers.TensorBoardLogger(conf.program.log_dir, name=experiment_name)
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, save_dir=conf.program.log_dir)

    if isinstance(task, ObjectDetectionTask):
        monitor_metric = "val_map"
        mode = "max"
    else:
        monitor_metric = "val_loss"
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric, dirpath=experiment_dir,  filename='{epoch}_{val_loss:.2f}',
        save_top_k=1, save_last=False
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, min_delta=0.00, patience=18, mode=mode
    )

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))

    trainer_args["callbacks"] = [checkpoint_callback, early_stopping_callback]
    trainer_args["logger"] = mlf_logger
    trainer_args["default_root_dir"] = experiment_dir
    trainer = pl.Trainer(**trainer_args)

    if trainer_args.get("auto_lr_find") or trainer_args.get("auto_scale_batch_size"):
        trainer.tune(model=task, datamodule=datamodule)

    ######################################
    # Run experiment
    ######################################
    trainer.fit(model=task, datamodule=datamodule)
    test_metrics = trainer.test(model=task, datamodule=datamodule)
    if trainer.global_rank == 0:
        best_model_path = Path(checkpoint_callback.best_model_path)
        gdl_model_path = best_model_path.with_suffix(".pth.tar") 
        trainer.model.save_model(best_model_path, gdl_model_path)
    return test_metrics


if __name__ == "__main__":
    # Taken from https://github.com/pangeo-data/cog-best-practices
    _rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(_rasterio_best_practices)

    _hydra_stacktrace = {
        "OC_CAUSE": "1",
    }

    os.environ.update(_hydra_stacktrace)

    # Main training procedure
    main()
