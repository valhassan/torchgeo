# Licensed under the MIT License.

"""CCMEO datamodule."""

from pathlib import Path
from typing import Union, Sequence, Any, Callable, Dict, Optional, List

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from skimage import exposure
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import Units, GridGeoSampler
from torchvision.transforms import Compose

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from ..datasets.ccmeo import DigitalGlobe, InferenceDataset

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"

from ..samplers.single import GridGeoSamplerPlus


class CCMEODataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the CCMEO dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        train_splits: List[str],
        val_splits: List[str],
        test_splits: List[str],
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: int = 256,

        **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for CCMEO based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the CCMEO Dataset classes
            train_splits: The splits used to train the model, e.g. ["trn"]
            val_splits: The splits used to validate the model, e.g. ["val"]
            test_splits: The splits used to test the model, e.g. ["test"]
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)

        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.patch_size = patch_size
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = int(patch_size * 2.0)

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Apply batch augmentations after batch is transferred to the device.

        Args:
            batch: mini-batch of data
            batch_idx: batch index

        Returns:
            augmented mini-batch
        """
        if (
            hasattr(self, "trainer")
            and hasattr(self.trainer, "training")
            and self.trainer.training  # type: ignore[union-attr]
        ):
            # Kornia expects masks to be floats with a channel dimension
            x = batch["image"]
            y = batch["mask"].float().unsqueeze(1)

            train_augmentations = K.AugmentationSequential(
                K.RandomCrop((self.patch_size, self.patch_size)),
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomSharpness(p=0.5),
                K.ColorJitter(
                    p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                data_keys=["input", "mask"],
            )
            x, y = train_augmentations(x, y)

            # torchmetrics expects masks to be longs without a channel dimension
            batch["image"] = x
            batch["mask"] = y.squeeze(1).long()

        return batch

    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.

        Args:
            size: output image size
            image_value: value to pad image with
            mask_value: value to pad mask with

        Returns:
            function to perform padding
        """

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample.

        Args:
            size: output image size

        Returns:
            function to perform center crop
        """

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"] / 255.0

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        DigitalGlobe(self.root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = Compose(
            [
                self.preprocess,
            ]
        )
        val_transforms = Compose(
            [
                self.preprocess,
            ]
        )
        test_transforms = Compose(
            [
                self.preprocess,
            ]
        )

        self.train_dataset = DigitalGlobe(
            self.root_dir,
            splits=self.train_splits,
            transforms=train_transforms
        )

        self.val_dataset = DigitalGlobe(
            self.root_dir,
            splits=self.val_splits,
            transforms=val_transforms
        )

        self.test_dataset = DigitalGlobe(
            self.root_dir,
            splits=self.test_splits,
            transforms=test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size//4,  # TODO: softcode
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size//4,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
    #     """Run :meth:`torchgeo.datasets.>CCMEODataset.plot`.
    #
    #     .. versionadded:: 0.2
    #     """
    #     return self.val_dataset.plot(*args, **kwargs)


# adapted from: https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datamodules/chesapeake.py#L100
def pad(
        size: int = 512, mode='constant'
) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Returns a function to perform a padding transform on a single sample.
    Args:
        size: size of padding to apply
        image_value: value to pad image with
        mask_value: value to pad mask with
    Returns:
        function to perform padding
    """
    # use _pad from utils
    def _pad(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """ Pads img_arr """
        sample["image"] = F.pad(sample["image"], (size, size, size, size), mode=mode)
        return sample

    return _pad


def enhance(
        clip_limit=0.1
) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """
    Returns a function to perform a histogram stretching (aka enhancement) on a single sample.
    """
    def _enhance(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Clip histogram by clip_limit # FIXME apply to entire image before inference? apply according to class?
        @param sample: sample dictionary containing image
        @param clip_limit:
        @return:
        """
        if clip_limit is None:
            return sample
        sample['image'] = np.moveaxis(sample["image"].numpy().astype(np.uint8), 0, -1)  # send channels last
        img_adapteq = []
        for band in range(sample['image'].shape[-1]):
            out_band = exposure.equalize_adapthist(sample["image"][..., band], clip_limit=clip_limit)
            out_band = (out_band*255).astype(np.uint8)
            img_adapteq.append(out_band)
        out_stacked = np.stack(img_adapteq, axis=-1)
        sample["image"] = torch.from_numpy(np.moveaxis(out_stacked, -1, 0))
        return sample
    return _enhance


def preprocess(
) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    """
    Returns a function to preprocess a single sample.
    @param sample:
    @return:
    """
    def _preprocess(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocesses a single sample.
        Args:
            sample: sample dictionary containing image
        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()

        return sample
    return _preprocess


# adapted from https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datamodules/chesapeake.py
class InferenceDataModule(LightningDataModule):
    """LightningDataModule implementation for the InferenceDataset.
    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """
    def __init__(
        self,
        item_path: Union[str, Path],
        root_dir: Union[str, Path],
        outpath: Union[str, Path],
        bands: Sequence = ('red', 'green', 'blue'),
        patch_size: int = 256,
        stride: int = 256,
        pad: int = 256,
        batch_size: int = 1,
        num_workers: int = 0,
        download: bool = False,
        use_projection_units: bool = False,
        save_heatmap: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for InferenceDataset based Dataloader.
        @param item_path:
            path to stac item containing imagery assets to infer on
        @param root_dir:
            The ``root`` argugment to pass to the InferenceDataset class
        @param outpath:
            path to desired output
        @param patch_size:
            The size of each patch in pixels
        @param stride:
            stride between each chip
        @param pad:
            padding to apply to each chip
        @param batch_size:
            The batch size to use in all created DataLoaders
        @param num_workers:
            The number of workers to use in all created DataLoaders
        @param download:
            if True, download dataset and store it in the root directory.
        @param use_projection_units : bool, optional
            Is `patch_size` in pixel units (default) or distance units?
        @param save_heatmap: bool, optional
            if True, saves heatmap from raw inference, after merging and smoothing chips
        Raises:
            ValueError: if ``use_prior_labels`` is used with ``class_set==7``
        """
        super().__init__()  # type: ignore[no-untyped-call]

        self.item_path = item_path
        self.root_dir = root_dir
        self.outpath = outpath
        self.patch_size = patch_size
        self.stride = stride
        self.pad_size = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.use_projection_units = use_projection_units
        self.bands = bands
        self.save_heatmap = save_heatmap

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.
        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        InferenceDataset(
            item_path=self.item_path,
            root=self.root_dir,
            outpath=self.outpath,
            bands=self.bands,
            transforms=None,
            download=self.download,
            pad=self.pad_size,
        )

    def setup(self, stage: Optional[str] = None,
              test_transforms: Compose = Compose([pad(16, mode='reflect'), enhance, preprocess])):
        """Instantiate the InferenceDataset.
        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        @param test_transforms:
        @param stage: stage to set up
        """
        self.inference_dataset = InferenceDataset(
            item_path=self.item_path,
            root=self.root_dir,
            outpath=self.outpath,
            bands=self.bands,
            transforms=test_transforms,
            download=self.download,
            pad=self.pad_size,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for inference.
        Returns:
            inference data loader
        """
        units = Units.PIXELS if not self.use_projection_units else Units.CRS
        self.sampler = GridGeoSamplerPlus(
            self.inference_dataset,
            size=self.patch_size,
            stride=self.stride,
            units=units,
        )
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
            shuffle=False,
        )

    def write_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for inference.
        Returns:
            inference data loader
        """
        self.write_dataset = self.inference_dataset.copy()

        sampler = GridGeoSampler(
            self.inference_dataset,
            size=self.patch_size,
            stride=self.patch_size,
        )
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
            shuffle=False,
        )

    def postprocess(self):
        pass  # TODO: move some/all post-processing operations to this method
