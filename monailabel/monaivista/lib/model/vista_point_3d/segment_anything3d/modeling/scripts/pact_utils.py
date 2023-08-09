from copy import copy, deepcopy
import h5py
import os
from nibabel.imageglobals import LoggingOutputSuppressor
import numpy as np
import torch
from monai.config import DtypeLike, KeysCollection
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class LoadImageh5d(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            reader: Optional[Union[ImageReader, str]] = None,
            dtype: DtypeLike = np.float32,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = DEFAULT_POST_FIX,
            overwriting: bool = False,
            image_only: bool = False,
            ensure_channel_first: bool = False,
            simple_keys: bool = False,
            allow_missing_keys: bool = False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        # CHANGED to LABEL
        post_label_pth = d['label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['label'] = data[0]
        return d

def remove255(x):
    x[x==255] = 0
    return x