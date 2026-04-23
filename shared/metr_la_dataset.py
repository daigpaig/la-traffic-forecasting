"""Project-local METR-LA dataset compatible with :class:`torch_geometric.data.InMemoryDataset`.

The upstream METR-LA pull request was never merged into PyTorch Geometric; this class
mirrors the intended layout (raw download + single :class:`~torch_geometric.data.Data`
holding a ``traffic`` tensor of shape ``[T, 207, 1]``).
"""

from __future__ import annotations

import os
import os.path as osp
import zipfile
from typing import Callable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, InMemoryDataset


class METR_LA(InMemoryDataset):
    r"""METR-LA traffic speeds (5-minute resolution) as a single temporal tensor.

    **STATS:**

    .. list-table::
        :widths: 10 10 10
        :header-rows: 1

        * - #timesteps
          - #sensors
          - #channels
        * - 34,272
          - 207
          - 1
    """

    # Same archive used by common open-source METR-LA loaders (e.g. Torch Spatiotemporal).
    url = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "metr_la.h5",
            "distances_la.csv",
            "sensor_locations_la.csv",
            "sensor_ids_la.txt",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        zip_path = osp.join(self.raw_dir, "metr_la_download.zip")
        urlretrieve(self.url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)
        os.remove(zip_path)

    def process(self) -> None:
        path = osp.join(self.raw_dir, "metr_la.h5")
        df = pd.read_hdf(path)
        values = np.ascontiguousarray(
            np.expand_dims(df.to_numpy(dtype=np.float32), axis=-1)
        )
        traffic = torch.from_numpy(values)
        data = Data(traffic=traffic)
        if self.pre_filter is not None and not self.pre_filter(data):
            raise RuntimeError("pre_filter rejected METR-LA data.")
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"
