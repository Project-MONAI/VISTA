# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import Optional, Sequence, Union
import numpy as np
import torch
import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch, list_data_collate
from .monai_utils import sliding_window_inference
from segment_anything3d import build_3d_samm_swin_b
from .utils import point_based_window_inferer, pad_previous_mask
from functools import partial
from .utils import convert_points_to_disc 
import pdb
from segment_anything3d import sam_model_registry
from vista3d import vista_model_registry
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
rearrange, _ = optional_import("einops", name="rearrange")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if __package__ in (None, ""):
    from train import pre_operation, CONFIG
else:
    from .train import pre_operation, CONFIG

def infer_wrapper(inputs, model, **kwargs):
    outputs = model(input_images=inputs,**kwargs)
    return outputs.transpose(1,0)

class InferClass:
    def __init__(self,
                 config_file: Optional[Union[str,
                                             Sequence[str]]] = None,
                 **override):
        pre_operation(config_file, **override)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        _args = _update_args(config_file=config_file, **override)
        config_file_ = _pop_args(_args, "config_file")[0]

        parser = ConfigParser()
        parser.read_config(config_file_)
        parser.update(pairs=_args)
        
        self.amp = parser.get_parsed_content("amp")
        input_channels = parser.get_parsed_content("input_channels")
        patch_size = parser.get_parsed_content("patch_size")
        self.num_sw_batch_size = parser.get_parsed_content(
            "num_sw_batch_size")
        self.overlap_ratio_final = parser.get_parsed_content(
            "overlap_ratio_final")
        self.patch_size_valid = parser.get_parsed_content(
            "patch_size_valid")
        softmax = parser.get_parsed_content("softmax")

        ckpt_name = parser.get_parsed_content("infer")["ckpt_name"]
        output_path = parser.get_parsed_content("infer")["output_path"]
        self.sliding_window_inferer = point_based_window_inferer if parser.get_parsed_content("infer")["inferer"] == 'point' else sliding_window_inference

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
	    
        CONFIG["handlers"]["file"]["filename"] =parser.get_parsed_content("infer")["log_output_file"]	
        logging.config.dictConfig(CONFIG)
        self.infer_transforms = parser.get_parsed_content("transforms_infer")

        self.device = torch.device("cuda:0")
        model_registry = parser.get_parsed_content("model")
        try:
            model = vista_model_registry[model_registry](in_channels=input_channels, image_size=patch_size)
        except:
            model = sam_model_registry[model_registry](in_channels=input_channels, image_size=patch_size)
        self.model = model.to(self.device)

        pretrained_ckpt = torch.load(ckpt_name, map_location=self.device)
        self.model.load_state_dict(pretrained_ckpt)
        logger.debug(f"[debug] checkpoint {ckpt_name:s} loaded")
        
        post_transforms = [	
            transforms.Invertd(	
                keys="pred",	
                transform=self.infer_transforms,	
                orig_keys="image",	
                meta_keys="pred_meta_dict",	
                orig_meta_keys="image_meta_dict",	
                meta_key_postfix="meta_dict",	
                nearest_interp=False,	
                to_tensor=True),	
            transforms.Activationsd(	
                keys="pred",	
                softmax=softmax,	
                sigmoid=not softmax)]        	
        # return pred probs
        self.post_transforms_prob = transforms.Compose(post_transforms)        
        if softmax:
            post_transforms += [
                transforms.AsDiscreted(
                    keys="pred",
                    argmax=True)]
        else:
            post_transforms += [
                transforms.AsDiscreted(
                    keys="pred",
                    threshold=0.5)]
        self.post_transforms = transforms.Compose(post_transforms)
        self.prev_mask = None
        self.batch_data = None
        return

    def clear_cache(self):
        self.prev_mask = None
        self.batch_data = None

    def transform_points(self, point, affine):
        """ transform point to the coordinates of the transformed image
        point: numpy array [bs, N, 3]
        """
        bs, N = point.shape[:2]
        point = np.concatenate((point,np.ones((bs, N,1))), axis=-1)
        point = rearrange(point, 'b n d -> d (b n)')
        point = affine @ point
        point = rearrange(point, 'd (b n)-> b n d', b=bs)[:,:,:3]
        return point

    @torch.no_grad()
    def infer(self, image_file, point, point_label, label_prompt, save_mask=False):
        """ Infer a single image_file. If save_mask is true, save the argmax prediction to disk. If false,
        do not save and return the probability maps (usually used by autorunner emsembler).
        """
        self.model.eval()
        if self.batch_data is not None:
            batch_data = self.batch_data
        else:
            batch_data = self.infer_transforms(image_file)
            batch_data = list_data_collate([batch_data])
            self.batch_data = batch_data
        if point is not None:
            point = self.transform_points(point, np.linalg.inv(batch_data['image'].affine[0]) @ batch_data['image'].meta['original_affine'][0].numpy())
            self.sliding_window_inferer = point_based_window_inferer
        else:
            self.sliding_window_inferer = sliding_window_inference
        device_list_input = [self.device, self.device, "cpu"]	
        device_list_output = [self.device, "cpu", "cpu"]	
        for _device_in, _device_out in zip(	
                device_list_input, device_list_output):	
            try:	
                with torch.cuda.amp.autocast(enabled=self.amp):	
                    batch_data["pred"] = self.sliding_window_inferer(	
                        inputs=batch_data["image"].to(_device_in),	
                        roi_size=self.patch_size_valid,	
                        sw_batch_size=self.num_sw_batch_size,	
                        predictor=partial(infer_wrapper, model=self.model),
                        mode="gaussian",	
                        overlap=self.overlap_ratio_final,	
                        sw_device=self.device,	
                        device=_device_out,
                        point_coords=torch.tensor(point).to(_device_in) if point is not None else None,
                        point_labels=torch.tensor(point_label).to(_device_in) if point_label is not None else None,
                        class_vector=torch.tensor(label_prompt).to(_device_in) if label_prompt is not None else None,
                        masks=torch.tensor(self.prev_mask).to(_device_in) if self.prev_mask is not None else None,
                        point_mask=None)
                    
                    if not hasattr(batch_data["pred"],'meta'):
                        batch_data["pred"] = monai.data.MetaTensor(batch_data["pred"], affine=batch_data["image"].meta["affine"], meta=batch_data["image"].meta)

                    
                # this pad the prev mask, copied from sliding window inferer.
                self.prev_mask = batch_data["pred"]
                # self.prev_mask, _ = pad_previous_mask(batch_data["pred"].sigmoid(), self.patch_size_valid)
                # self.prev_mask = None
                if save_mask:
                    batch_data = [self.post_transforms(i)
                                for i in decollate_batch(batch_data)]
                else:
                    batch_data = [self.post_transforms_prob(i)
                                for i in decollate_batch(batch_data)]                    
                finished = True	
            except RuntimeError as e:
                if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                    raise e
                finished = False	
            if finished:	
                break
        if not finished:
            raise RuntimeError('Infer not finished due to OOM.')
        return batch_data[0]["pred"]
    

def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    infer_instance = InferClass(config_file, **override)
    if infer_instance.fast:
        logger.debug("[debug] fast mode")
        infer_instance.batch_infer()
    else:
        logger.debug("[debug] slow mode")
        infer_instance.infer_all()
    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
