# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation tasks."""

import warnings
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (  # type: ignore[attr-defined]
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    BinaryJaccardIndex,
)

from ..datasets.utils import unbind_samples

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"

from ..utils import get_logger

logging = get_logger(__name__)


class MultiClassTransformer(pl.LightningModule):
    """LightningModule for semantic segmentation of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM encoders <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.model = instantiate(self.hparams["model"])
        checkpoint_path = self.hparams["model_state_dict"]
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        self.loss = instantiate(self.hparams["loss"])

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function
            ignore_index: Optional integer class index to ignore in the loss and metrics
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.3
           The *ignore_zeros* parameter was renamed to *ignore_index*.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["loss"]["ignore_index"], (int, type(None))):
           raise ValueError("ignore_index must be an int or None")
        if (kwargs["loss"]["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
           warnings.warn(
               "ignore_index has no effect on training when loss='jaccard'",
               UserWarning,
           )
        self.ignore_index = kwargs["loss"]["ignore_index"]
        self.config_task()

        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.hparams["model"]["classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.hparams["model"]["classes"],
                    ignore_index=self.ignore_index,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
    
    def load_model(self, path):
        checkpoint = torch.load(f=path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    def save_model(self, model_path, out_path):
        print(f"Updating PL to GDL checkpoint format")
        
        checkpoint = torch.load(f=model_path, map_location='cpu')
        # original dictionary is kept intact
        checkpoint['model_state_dict'] = checkpoint['state_dict']

        # removes ".model" prefix for each key in weights dict
        if list(checkpoint['model_state_dict'].keys())[0].startswith('model'):
            new_state_dict = {}
            new_state_dict['model_state_dict'] = checkpoint['model_state_dict'].copy()
            new_state_dict['model_state_dict'] = {k.split("model.")[-1]: v for k, v in
                                                checkpoint['model_state_dict'].items()}
            checkpoint['model_state_dict'] = new_state_dict['model_state_dict']
        
        torch.save(checkpoint, out_path)
        print(f"Saved GDL checkpoint format: {out_path}")
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(# type: ignore[override] 
                      self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(# type: ignore[override] 
                        self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment  # type: ignore[union-attr]
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override] 
                  self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]        
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams["optimizer"], params=self.model.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }


class SemanticSegmentationTask(pl.LightningModule):
    """LightningModule for semantic segmentation of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM encoders <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.model = instantiate(self.hparams["model"])
        checkpoint_path = self.hparams["model_state_dict"]
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        self.loss = instantiate(self.hparams["loss"])

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function
            ignore_index: Optional integer class index to ignore in the loss and metrics
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.3
           The *ignore_zeros* parameter was renamed to *ignore_index*.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["loss"]["ignore_index"], (int, type(None))):
           raise ValueError("ignore_index must be an int or None")
        if (kwargs["loss"]["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
           warnings.warn(
               "ignore_index has no effect on training when loss='jaccard'",
               UserWarning,
           )
        self.ignore_index = kwargs["loss"]["ignore_index"]
        self.config_task()

        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.hparams["model"]["classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.hparams["model"]["classes"],
                    ignore_index=self.ignore_index,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
    
    def load_model(self, path):
        checkpoint = torch.load(f=path, map_location='cpu')
        if "model_state_dict" in checkpoint.keys():
            state_key = "model_state_dict"
        else:
            state_key = "state_dict"
            
        self.model.load_state_dict(checkpoint[state_key], strict=True)
    
    def save_model(self, model_path, out_path):
        print(f"Updating PL to GDL checkpoint format")
        
        checkpoint = torch.load(f=model_path, map_location='cpu')
        # original dictionary is kept intact
        checkpoint['model_state_dict'] = checkpoint['state_dict']

        # removes ".model" prefix for each key in weights dict
        if list(checkpoint['model_state_dict'].keys())[0].startswith('model'):
            new_state_dict = {}
            new_state_dict['model_state_dict'] = checkpoint['model_state_dict'].copy()
            new_state_dict['model_state_dict'] = {k.split("model.")[-1]: v for k, v in
                                                checkpoint['model_state_dict'].items()}
            checkpoint['model_state_dict'] = new_state_dict['model_state_dict']
        
        torch.save(checkpoint, out_path)
        print(f"Saved GDL checkpoint format: {out_path}")
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(# type: ignore[override] 
                      self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(# type: ignore[override] 
                        self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment  # type: ignore[union-attr]
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override] 
                  self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]        
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams["optimizer"], params=self.model.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(
                    optimizer, max_lr=1e-3, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                    epochs=self.trainer.max_epochs,
                ),
                "monitor": "val_loss",
                "interval": "step",
            },
        }


class BinarySemanticSegmentationTask(pl.LightningModule):
    """
    LightningModule for semantic segmentation of images.
    FIXME: not all metrics from torchmetrics work. For JaccardIndex, manually set multilabel=True and under
    torchmetrics.functional.classification.jaccard._jaccard_from_confmat, add confmat = confmat.squeeze() at beginning
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.model = instantiate(self.hparams["model"])
        self.loss = instantiate(self.hparams["loss"])

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function
            ignore_zeros: Whether to ignore the "0" class value in the loss and metrics

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["loss"]["ignore_index"], (int, type(None))):
           raise ValueError("ignore_index must be an int or None")
        if (kwargs["loss"]["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
           warnings.warn(
               "ignore_index has no effect on training when loss='jaccard'",
               UserWarning,
           )
        self.ignore_index = kwargs["loss"]["ignore_index"]
        self.config_task()

        self.train_metrics = MetricCollection(
            [
                BinaryJaccardIndex(
                    num_classes=self.hparams["model"]["classes"],
                    ignore_index=self.hparams["loss"]["ignore_index"],
                    multilabel=True,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average JaccardIndex.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x).squeeze(dim=1)
        y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.float())

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_sigmoid, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average JaccardIndex.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions. TODO

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x).squeeze(dim=1)
        y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_sigmoid, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat_sigmoid
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x).squeeze(dim=1)
        y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.float())

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_sigmoid, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams["optimizer"], params=self.model.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # "scheduler": ReduceLROnPlateau(
                #     optimizer, patience=self.hparams["learning_rate_schedule_patience"]
                "scheduler": OneCycleLR(
                    optimizer, max_lr=1e-3, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                    epochs=self.trainer.max_epochs,
                ),
                "monitor": "val_loss",
                "interval": "step",
            },
        }
