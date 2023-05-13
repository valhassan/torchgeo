# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CCMEO datasets."""

import abc
import os
from pathlib import Path
from typing import Union, Optional, Sequence, Callable, Dict, Any, cast, List, Tuple

import pandas
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.common import is_url
import pystac
from pystac.extensions.eo import ItemEOExtension, Band
import rasterio as rio
from rtree import Index
from rtree.index import Property
import torch
from torchgeo.datasets import RasterDataset, BoundingBox
from torchvision.datasets.utils import download_url

from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import Affine
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    check_integrity,
    download_radiant_mlhub_collection,
    extract_archive,
    percentile_normalization,
)
from ..utils import get_logger

logging = get_logger(__name__)


class SingleBandItemEO(ItemEOExtension):
    def __init__(self, item: pystac.Item):
        super().__init__(item)
        self._assets_by_common_name = None

    @property
    def asset_by_common_name(self) -> Dict:
        """
        Adapted from: https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        Get assets by common band name (only works for assets containing 1 band
        @param common_name:
        @return:
        """
        if self._assets_by_common_name is None:
            self._assets_by_common_name = {}
            for name, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0]
                    if 'common_name' in eo_band.keys():
                        common_name = eo_band['common_name']
                        if not Band.band_range(common_name):  # Hacky but easiest way to validate common names
                            raise ValueError(f'Must be one of the accepted common names. Got "{common_name}".')
                        else:
                            self._assets_by_common_name[common_name] = {'href': a_meta.href, 'name': name}
        if not self._assets_by_common_name:
            raise ValueError(f"Common names for assets cannot be retrieved")
        return self._assets_by_common_name


class CCMEO(NonGeoDataset, abc.ABC):
    """Abstract base class for the SpaceNet datasets.

    The `CCMEO`_ datasets are a set of datasets that contain mostly 4-class labels
    (forest, waterbody, road, building) mapped over high-resolution satellite imagery
    obtained from a variety of sensors such as Worldview-2, Worldview-3, Worldview-4,
    GeoEye, Quickbird and aerial imagery.
    """

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

    @property
    @abc.abstractmethod
    def imagery(self) -> Dict[str, str]:
        """Mapping of image identifier and filename."""

    # TODO
    # @property
    # @abc.abstractmethod
    # def label_glob(self) -> str:
    #     """Label filename."""

    # @property
    # @abc.abstractmethod
    # def collection_md5_dict(self) -> Dict[str, str]:
    #     """Mapping of collection id and md5 checksum."""

    def __init__(
        self,
        root: str,
        image: str,
        collections: List[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
        splits=["trn"]
    ) -> None:
        """Initialize a new CCMEO Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection
            collections: collection selection
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        self.root = Path(root)
        self.image = image  # For testing
        self.splits = splits

        # TODO
        # if collections:
        #     for collection in collections:
        #         assert collection in self.collection_md5_dict

        self.collections = collections #or list(self.collection_md5_dict.keys())
        # self.filename = self.imagery[image]  # TODO
        self.transforms = transforms
        self.checksum = checksum

        # TODO
        #to_be_downloaded = self._check_integrity()

        # if to_be_downloaded:
        #     if not download:
        #         raise RuntimeError(
        #             f"Dataset not found in `root={self.root}` and `download=False`, "
        #             "either specify a different `root` directory or use "
        #             "`download=True` to automaticaly download the dataset."
        #         )
        #     else:
        #         self._download(to_be_downloaded)

        self.files = self._load_files(root)

    def _load_files(self, root: str) -> List[Dict[str, Path]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        # TODO: glob or list?
        files = []
        # for collection in self.collections:
        for split in self.splits:
            rows = pandas.read_csv(split, sep=';', header=None)
            for row in rows.values:
                imgpath, lbl_path = row[:2]
                imgpath, lbl_path = self.root / imgpath, self.root / lbl_path
                if not imgpath.is_file():
                    raise FileNotFoundError(imgpath)
                if not lbl_path.is_file():
                    raise FileNotFoundError(lbl_path)
                files.append({"image_path": imgpath, "label_path": lbl_path})

                # images = (Path(root) / collection).glob(f"**/{split}/*/images/*.tif")
                # images = sorted(images)
                # for imgpath in images:
                #     lbl_path = list((imgpath.parent.parent/"labels_burned").glob(f"*{imgpath.name[-16:]}"))[0]  # FIXME: add robustness
                #     files.append({"image_path": imgpath, "label_path": lbl_path})
        return files

    def _load_image(self, path: Union[str, Path]) -> Tuple[Tensor, Affine, CRS]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor, img.transform, img.crs

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files["image_path"])
        # h, w = img.shape[1:]
        mask, *_ = self._load_image(files["label_path"])
        mask[mask == 255] = 0  # TODO: make ignore_index work
        mask = mask.squeeze()

        if not img.shape[-2:] == mask.shape[-2:]:
            raise ValueError(f"Mismatch between image chip shape ({img.shape}) and mask chip shape ({mask.shape})")
        sample = {"image": img,
                  "mask": mask,
                  "image_path": str(files["image_path"]),
                  "label_path": str(files["label_path"]),
                  "aoi_id": files["image_path"].parent.parent.stem}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> List[str]:
        """Checks the integrity of the dataset structure.

        Returns:
            List of collections to be downloaded
        """
        # Check if collections exist
        missing_collections = []
        for collection in self.collections:
            stacpath = os.path.join(self.root, collection, "collection.json")

            if not os.path.exists(stacpath):
                missing_collections.append(collection)

        if not missing_collections:
            return []

        to_be_downloaded = []
        for collection in missing_collections:
            archive_path = os.path.join(self.root, collection + ".tar.gz")
            if os.path.exists(archive_path):
                print(f"Found {collection} archive")
                if (
                    self.checksum
                    and check_integrity(
                        archive_path, self.collection_md5_dict[collection]
                    )
                    or not self.checksum
                ):
                    print("Extracting...")
                    extract_archive(archive_path)
                else:
                    print(f"Collection {collection} is corrupted")
                    to_be_downloaded.append(collection)
            else:
                print(f"{collection} not found")
                to_be_downloaded.append(collection)

        return to_be_downloaded

    def _download(self, collections: List[str], api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            collections: Collections to be downloaded
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        for collection in collections:
            download_radiant_mlhub_collection(collection, self.root, api_key)
            archive_path = os.path.join(self.root, collection + ".tar.gz")
            if (
                not self.checksum
                or not check_integrity(
                    archive_path, self.collection_md5_dict[collection]
                )
            ) and self.checksum:
                raise RuntimeError(f"Collection {collection} corrupted")

            print("Extracting...")
            extract_archive(archive_path)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        # image can be 1 channel or >3 channels
        if sample["image"].shape[0] == 1:
            image = np.rollaxis(sample["image"].numpy(), 0, 3)
        else:
            image = np.rollaxis(sample["image"][:3].numpy(), 0, 3)
        image = percentile_normalization(image, axis=(0, 1))

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = sample["mask"].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class DigitalGlobe(CCMEO):
    """SpaceNet 1: Building Detection v1 Dataset.

    `SpaceNet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    Dataset features:

    * No. of images: 6940 (8 Band) + 6940 (RGB)
    * No. of polygons: 382,534 building labels
    * Area Coverage: 2544 sq km
    * GSD: 1 m (8 band),  50 cm (rgb)
    * Chip size: 101 x 110 (8 band), 406 x 438 (rgb)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * 8Band.tif (Multispectral)
        * RGB.tif (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    """

    dataset_id = "ccmeo-digitalglobe"  # TODO
    imagery = {"rgb": "*.tif", "rgbn": "*RGBN.tif"}  # TODO
    chip_size = {"rgb": (512, 512), "8band": (101, 110)}
    # label_glob = "labels.geojson"
    #collection_md5_dict = {"sn1_AOI_1_RIO": "e6ea35331636fa0c036c04b3d1cbf226"}  # TODO

    def __init__(
        self,
        root: str,
        splits: Sequence[str] = ["trn"],
        image: str = "rgb",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CCMEO Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be "rgb" or "rgbn"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        collections = ["gdl_buildings_lxastro"]  # TODO reimplement?
        assert image in {"rgb", "8band"}
        super().__init__(
            root, image, collections, transforms, download, checksum, splits
        )


class InferenceDataset(RasterDataset):
    def __init__(
            self,
            item_path: str,
            root: str = "data",
            outpath: Union[Path, str] = "pred.tif",
            crs: Optional[CRS] = None,
            res: Optional[float] = None,
            bands: Sequence[str] = [],
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            download: bool = False,
            singleband_files: bool = True,
            pad: int = 256,
    ) -> None:
        """Initialize a new CCCOT Dataset instance.

        @param item_path:
            path to stac item containing imagery assets to infer on
        @param root:
            root directory where dataset can be found
        @param outpath:
            path to desired output
        @param crs:
            Coordinate reference system of Dataset
        @param res:
            Resolution (GSD) of Dataset
        @param bands:
            band selection which must be a list of STAC Item common names from eo extension.
            See: https://github.com/stac-extensions/eo/#common-band-names
        @param transforms:
            Tranforms to apply to raw chip before feeding it to model
        @param download:
            if True, download dataset and store it in the root directory.
        @param singleband_files:
            if True, this class will expect assets from Stac Item to contain only one band  # TODO: implement multiband
        @param pad:
            padding to apply to each chip
        """
        self.item_url = item_path
        self.bands = bands
        if len(self.bands) == 0:
            logging.warning(f"At least one band should be chosen if assets need to be reached")
        self.root = Path(root)
        self.transforms = transforms
        self.separate_files = singleband_files
        self.download = download
        self.pad = pad
        self.outpath = outpath
        self.outpath_vec = self.root / f"{outpath.stem}.gpkg"
        self.outpath_heat = self.root / f"{outpath.stem}_heatmap.tif"
        self.cache = download

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.item_url = self.item_url if is_url(self.item_url) else to_absolute_path(self.item_url)
        # Read Stac item from url
        if self.separate_files:
            self.item = SingleBandItemEO(pystac.Item.from_file(str(self.item_url)))
        else:
            raise NotImplementedError(f"Currently only support single-band Stac Items")  # TODO

        # Create band inventory (all available bands)
        self.all_bands = [band for band in self.item.asset_by_common_name.keys()]

        # Filter only desired bands
        self.bands_dict = {k: v for k, v in self.item.asset_by_common_name.items() if k in self.bands}

        # Make sure desired bands are subset of inventory
        if not set(self.bands).issubset(set(self.all_bands)):
            raise ValueError(f"Selected bands ({self.bands}) should be a subset of available bands ({self.all_bands})")

        # Download assets if desired
        if self.download:
            for cname in self.bands:
                out_name = self.root / Path(self.bands_dict[cname]['href']).name
                download_url(self.bands_dict[cname]['href'], root=str(self.root), filename=str(out_name))
                self.bands_dict[cname]['href'] = out_name

        # Open first asset with rasterio (for metadata: colormap, crs, resolution, etc.)
        if self.bands:
            self.first_asset = self.bands_dict[self.bands[0]]['href']
            self.first_asset = self.first_asset if is_url(self.first_asset) else to_absolute_path(self.first_asset)

            self.src = rio.open(self.first_asset)

            # See if file has a color map
            try:
                self.cmap = self.src.colormap(1)
            except ValueError:
                pass

            if crs is None:
                crs = self.src.crs
            if res is None:
                res = self.src.res[0]

            # to implement reprojection, see:
            # https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datasets/geo.py#L361
            minx, miny, maxx, maxy = self.src.bounds

            # Get temporal information from STAC item
            self.date = self.item.item.datetime
            mint = maxt = self.date.timestamp()

            # Add paths to Rtree index
            coords = (minx, maxx, miny, maxy, mint, maxt)

            self.index.insert(0, coords, self.first_asset)
            self._crs = cast(CRS, crs)
            self.res = cast(float, res)

    def create_empty_outraster(self):
        """
        Writes an empty output raster to disk
        @return:
        """
        pred = np.zeros(self.src.shape, dtype=np.uint8)
        pred = pred[np.newaxis, :, :].astype(np.uint8)
        out_meta = self.src.profile
        out_meta.update({"driver": "GTiff",
                         "height": pred.shape[1],
                         "width": pred.shape[2],
                         "count": pred.shape[0],
                         "dtype": 'uint8',
                         'tiled': True,
                         'blockxsize': 256,
                         'blockysize': 256,
                         "compress": 'lzw'})
        with rio.open(self.outpath, 'w+', **out_meta) as dest:
            dest.write(pred)

    def create_empty_outraster_heatmap(self, num_classes: int):
        """
        Writes an empty output raster for heatmap to disk
        @param num_classes:
        @return:
        """
        pred = np.zeros((num_classes, self.src.shape[0], self.src.shape[1]), dtype=np.uint8)
        out_meta = self.src.profile
        out_meta.update({"driver": "GTiff",
                         "height": pred.shape[1],
                         "width": pred.shape[2],
                         "count": pred.shape[0],
                         "dtype": 'uint8',
                         'tiled': True,
                         'blockxsize': 256,
                         'blockysize': 256,
                         "compress": 'lzw'})
        with rio.open(self.outpath_heat, 'w+', **out_meta) as dest:
            dest.write(pred)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # TODO: turn off external logs (ex.: rasterio._env)
        # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
        with rio.Env(CPL_CURL_VERBOSE=False):
            if self.separate_files:
                data_list: List[Tensor] = []
                for band in getattr(self, "bands", self.all_bands):
                    band_filepaths = []
                    filepath = self.bands_dict[band]['href']  # hardcoded: stac item reader needs asset_by_common_name()
                    filepath = filepath if is_url(filepath) else to_absolute_path(filepath)
                    band_filepaths.append(filepath)
                    data_list.append(self._merge_files(band_filepaths, query))
                data = torch.cat(data_list)  # type: ignore[attr-defined]
            else:
                # FIXME: implement multi-band Stac item: https://github.com/stac-extensions/eo/blob/main/examples/item.json
                data = self._merge_files(filepaths, query)
        data = data.float()

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query, "files": filepaths}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample