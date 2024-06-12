import torch
import torch.nn.functional as F

import random
import pdb
import copy
import numpy as np
import time
from skimage import measure
from matplotlib import pyplot as plt
import monai
from monai.utils import ensure_tuple_rep
from monai.metrics import compute_dice
from .sliding_window import sliding_window_inference

ENABLE_SPECIAL=True
SPECIAL_INDEX = [23, 24, 25, 26, 27, 57, 128]
MERGE_LIST= {
    1: [25,26], # hepatic tumor and vessel merge into liver
    4: [24], # pancreatic tumor merge into pancreas
    132: [57], # overlap with trachea merge into airway
}

def get_point_label(id):
    # [B, N]
    if id in SPECIAL_INDEX and ENABLE_SPECIAL:
        return 2, 3
    else:
        return 0, 1
    
def convert_point_label(point_label, label_set=None):
    if label_set is None or not ENABLE_SPECIAL:
        return point_label
    assert point_label.shape[0] == len(label_set)
    for i in range(len(label_set)):
        if label_set[i] in SPECIAL_INDEX:
            for j in range(len(point_label[i])):
                point_label[i,j] = point_label[i,j] + 2 if point_label[i,j] > -1 else point_label[i,j]
    return point_label


def none_cat(point, point_pseudo):
    _point = None
    if point is not None:
        if point_pseudo is not None:
            if len(point.shape) == 3:
                pad_n = max(point.shape[1], point_pseudo.shape[1])
                point = F.pad(point,(0,0,0,pad_n-point.shape[1],0,0))
                point_pseudo = F.pad(point_pseudo, (0,0,0,pad_n-point_pseudo.shape[1],0,0))
            elif len(point.shape) == 2:
                pad_n = max(point.shape[1], point_pseudo.shape[1])
                point = F.pad(point,(0,pad_n-point.shape[1],0,0),value=-1)
                point_pseudo = F.pad(point_pseudo,(0,pad_n-point_pseudo.shape[1],0,0),value=-1)
            elif len(point.shape) == 1:
                pad_n = max(point.shape[0], point_pseudo.shape[0])
                point = F.pad(point,(0,pad_n-point.shape[1]),value=-1)
                point_pseudo = F.pad(point_pseudo,(0,pad_n-point_pseudo.shape[1]),value=-1)
            _point = torch.cat([point, point_pseudo], dim=0)
        else:
            _point = point
    elif point_pseudo is not None:
        _point = point_pseudo
    return _point

def sample_points_patch_val(labels, patch_coords, label_set, prev_mask, class_vector, use_center=True, mapped_label_set=None, max_ppoint=1, max_npoint=0, **kwargs):
    """ Sample points for patch during sliding window validation. The prev_mask is only used for auto + interactive. This function is called within 
    vista3d.py and will use largested cc combine, do not use for iterative point evaluation.
    label_set must match the value in labels. Mapped_label_set is the global mapped index. label_set must has the same length with mapped_label_set.
    """
    # only in validation when labels of the whole image is provided, sample points for every position
    if prev_mask is None or class_vector is None:
        _, point_coords, point_labels, _ = \
            generate_prompt_pairs_val(labels[patch_coords], label_set, 
                                    max_ppoint=max_ppoint,
                                    max_npoint=max_npoint,
                                    device=labels.device,
                                    use_center=use_center)
        point_labels = convert_point_label(point_labels, mapped_label_set)
    else:
        point_coords, point_labels = get_next_points(prev_mask[patch_coords].transpose(1,0).to(labels.device), 
                                                     labels[patch_coords], class_vector, None, None) 
    return point_coords, point_labels, torch.tensor(mapped_label_set).to(point_coords.device).unsqueeze(-1)

def erode3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(input_tensor.float().unsqueeze(0).unsqueeze(0), (erosion[2]//2, erosion[2]//2, erosion[1]//2, erosion[1]//2, erosion[0]//2, erosion[0]//2), mode='constant', value=1.0)

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)

def erode2d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 2)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(input_tensor.float().unsqueeze(0).unsqueeze(0), (erosion[1]//2, erosion[1]//2, erosion[0]//2, erosion[0]//2), mode='constant', value=1.0)

    # Apply erosion operation
    output = F.conv2d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)

def dilate3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(input_tensor.float().unsqueeze(0).unsqueeze(0), (erosion[2]//2, erosion[2]//2, erosion[1]//2, erosion[1]//2, erosion[0]//2, erosion[0]//2), mode='constant', value=0.0)

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output > 0, 1.0, 0.0)

    return output.squeeze(0).squeeze(0)

def get_center_point(plabels, Np):
    """ Get the center Np points of plabels
    """
    if Np == 1:
        max_slice = plabels.sum(0).sum(0).argmax().item()
        max_y = plabels[:,:,max_slice].sum(0).argmax().item()
        f = plabels[:,max_y,max_slice]
        lstart, lend, start, end = None, None, None, None
        max_interval = -1
        for x in range(len(f)):
            if f[x] == 1:
                if start is None:
                    start = x
                    end = x
                else:
                    end = x
            if f[x] == 0 and start is not None:
                interval = end - start
                if interval > max_interval:
                    max_interval = interval
                    lstart, lend = start, end 
                start, end = None, None
        if start is not None and end is not None:
            interval = end - start
            if interval > max_interval:
                lstart, lend, max_interval = start, end, interval  
        point = [lstart + (lend-lstart)//2, max_y, max_slice]
        if plabels[point[0],point[1],point[2]] == 0:
            print('Unexpected error occur in center point sample')
            
#         plt.imshow(plabels[:,:,max_slice].cpu().numpy());plt.scatter(point[1],point[0]);plt.show();plt.savefig('debug_center.png');plt.close()
        return torch.tensor(point).unsqueeze(0)
    else:
        raise NotImplementedError
    
def generate_prompt_pairs_auto_point(pred, gt, prompt_class, 
                                     pred_thresh=0.5, mapped=True,
                                     include_background=False, **kwargs):
    """ sample 2 points from foreground, 2 points from error region. This function is only used during auto + point evaluation. 
    """
    num_f = 2
    num_e = 2
    new_points = []
    new_points_label = []
    for id in range(len(prompt_class)):
        _new_points, _new_points_label = [], []
        _gt = (gt == prompt_class[id])[0,0]
        if (prompt_class[id] == 0 and not include_background) or _gt.sum() < num_f:
            new_points.append(torch.stack([torch.tensor([0,0,0], device=pred.device)] * (num_e + num_f)))
            new_points_label.append(torch.tensor([-1] * (num_e + num_f), device=pred.device))
            continue
        neg_id, pos_id = get_point_label(-1)
        if mapped:
            # if in the global index, some supported classes need modification.
            neg_id, pos_id = get_point_label(prompt_class[id])
            # DO NOT MERGE HERE. The merge classs evaluation will not perform merge.
            # if prompt_class[id].item() in MERGE_LIST.keys():
            #     for m in MERGE_LIST[prompt_class[id].item()]:
            #         _gt = torch.logical_or(_gt, (gt==m)[0,0])   
        
        plabelpoints = torch.nonzero(_gt)
        _new_points.extend(random.choices(plabelpoints, k=num_f))
        _new_points_label.extend([pos_id] * num_f)    
                          
        fn_mask = torch.logical_and(_gt, pred[id][0] < pred_thresh)
        fn_mask = erode3d(fn_mask)
        fn_mask = monai.transforms.utils.get_largest_connected_component_mask(fn_mask)
        fp_mask = torch.logical_and(torch.logical_not(_gt), pred[id][0] > pred_thresh)
        fp_mask = erode3d(fp_mask)
        fp_mask = monai.transforms.utils.get_largest_connected_component_mask(fp_mask)
        if (fn_mask.sum() >= fp_mask.sum()):
            plabelpoints = torch.nonzero(fn_mask)
            if len(plabelpoints) > num_e:
                _new_points.extend(random.choices(plabelpoints,k=num_e))
                _new_points_label.extend([pos_id] * num_e)                    
            else:
                _new_points.extend([torch.tensor([0,0,0], device=pred.device)] * num_e)
                _new_points_label.extend([-1] * num_e)           
        else:
            plabelpoints = torch.nonzero(fp_mask)
            if len(plabelpoints) > num_e:
                _new_points.extend(torch.stack(random.choices(plabelpoints,k=num_e)))
                _new_points_label.extend([neg_id] * num_e) 
            else:
                _new_points.extend([torch.tensor([0,0,0], device=pred.device)] * num_e)
                _new_points_label.extend([-1] * num_e)
        new_points.append(torch.stack(_new_points))
        new_points_label.append(torch.tensor(_new_points_label, device=pred.device))
    new_points = torch.stack(new_points)
    new_points_label = torch.stack(new_points_label)  
    return new_points, new_points_label

def generate_prompt_pairs_val(labels, label_set=None, max_ppoint=1, max_npoint=0, device='cpu', use_center=False):
    """ This function did not map point label to 2, 3. Only used in sample_points_patch_val
    Args:
        labels: torch.tensor from dataload, [1,1,H,W,D]
        label_set: the label list for the specific dataset
    Returns:
        label_prompt: [b, 1]
        point: [b, N, 3]
        point_label: [b, N]
        prompt_class: [b, 1], exactly the same with label_prompt for label indexing for training lloss. 

    """
    # class label number
    assert labels.shape[0] == 1, 'only support batch size 1'
    labels = labels[0,0]
    label_prompt = torch.tensor(label_set).to(device).unsqueeze(-1)
    unique_labels = labels.unique().cpu().numpy().tolist()
    _point = []
    _point_label = []
    Nn = max_npoint 
    Np = max_ppoint
    for id in label_set:
        if id in unique_labels:
            plabels = labels == int(id)
            nlabels = ~plabels
            _plabels = erode3d(plabels)
            _plabels = monai.transforms.utils.get_largest_connected_component_mask(_plabels)
            plabelpoints = torch.nonzero(_plabels).to(device)
            if len(plabelpoints) == 0:
                plabelpoints = torch.nonzero(plabels).to(device)
            nlabelpoints = torch.nonzero(nlabels).to(device)
            if use_center:
                pmean = plabelpoints.float().mean(0)
                pdis = ((plabelpoints - pmean)**2).sum(-1)
                _, sorted_indices = torch.sort(pdis)
                _point.append(torch.stack([plabelpoints[sorted_indices[i]] for i in range(min(len(plabelpoints), Np))] + 
                                        random.choices(nlabelpoints, k=min(len(nlabelpoints), Nn)) + 
                                        [torch.tensor([0,0,0],device=device)] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))))
                _point_label.append(torch.tensor([1] * min(len(plabelpoints), Np) + [0.] * min(len(nlabelpoints), Nn) + \
                                                [-1] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))).to(device))                
 
            else:
                _point.append(torch.stack(random.choices(plabelpoints, k=min(len(plabelpoints), Np)) + 
                                        random.choices(nlabelpoints, k=min(len(nlabelpoints), Nn)) + 
                                        [torch.tensor([0,0,0],device=device)] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))))
                _point_label.append(torch.tensor([1] * min(len(plabelpoints), Np) + [0.] * min(len(nlabelpoints), Nn) + \
                                                [-1] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))).to(device))
        else:
        # pad the background labels
            _point.append(torch.zeros(Np+Nn, 3).to(device)) # all 0
            _point_label.append(torch.zeros(Np+Nn).to(device) - 1) # -1 not a point
    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    prompt_class = copy.deepcopy(label_prompt)
    return label_prompt, point, point_label, prompt_class

def generate_prompt_pairs(labels, label_set=None, image_size=None, max_prompt=None, max_foreprompt=None, max_backprompt=1, max_point=20, 
                          include_background=False, drop_label_prob=0.2, drop_point_prob=0.2, 
                          convert_to_disc=False, radius=2, metric_class=None, ignore_labelset=False, point_sampler=None):
    """ 
    Args:
        labels: torch.tensor from dataload, [1,1,H,W,D]
        label_set: the label list for the specific dataset
        total_prompt: int, number of total prompt
        max_point: maximum number of points for each object
        include_background: if include label=0 into training prompt. May casue issue in partial label
                            trainig.
        metric_class: validation dice of each class. Must be the same dim with label_set
    Returns:
        label_prompt: [b, 1]
        point: [b, N, 3]
        point_label: [b, N]
        prompt_class: [b, 1], exactly the same with label_prompt for label indexing for training lloss. 

    """
    # class label number
    assert labels.shape[0] == 1, 'only support batch size 1'
    labels = labels[0,0]
    point_mask = None
    device = labels.device
    unique_labels = labels.unique().cpu().numpy().tolist()
    if ignore_labelset:
        if not include_background:
            unique_labels = list(set(unique_labels) - set([0]))
        background_labels = []
    else:
        if include_background:
            unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)))
        else:
            unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)) - set([0]))
        background_labels = list(set(label_set) - set(unique_labels))
    # during training, balance background and foreground prompts
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            background_labels = background_labels[:max_backprompt]

    if max_foreprompt is not None:
        if len(unique_labels) > max_foreprompt:
            random.shuffle(unique_labels)
            unique_labels = unique_labels[:max_foreprompt]  

    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                # unique_labels = random.sample(unique_labels, max_prompt)
                if metric_class is None:
                    prob = np.ones(len(unique_labels))
                else:
                    prob = 1 - metric_class[np.array(unique_labels).astype(int)] if len(label_set) == len(metric_class) else 1 - metric_class[np.array(unique_labels).astype(int)-1]
                prob = [w/sum(prob) for w in prob]
                unique_labels = np.random.choice(unique_labels, size=max_prompt, replace=False, p=prob).tolist()
                background_labels = []
            else:
                background_labels = random.sample(background_labels, max_prompt - len(unique_labels))
    _point = []
    _point_label = []     
    if point_sampler is None:
        Np = min(max_point, int(np.abs(random.gauss(mu=0,sigma=max_point//2)))+1)
        Nn = min(max_point, int(np.abs(random.gauss(mu=0,sigma=max_point//2))))
        for id in unique_labels:
            neg_id, pos_id = get_point_label(id)
            plabels = labels == int(id)
            nlabels = ~plabels
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
            _point.append(torch.stack(random.choices(plabelpoints, k=min(len(plabelpoints), Np)) + 
                                    random.choices(nlabelpoints, k=min(len(nlabelpoints), Nn)) + 
                                    [torch.tensor([0,0,0],device=device)] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))))
            _point_label.append(torch.tensor([pos_id] * min(len(plabelpoints), Np) + [neg_id] * min(len(nlabelpoints), Nn) + \
                                            [-1] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))).to(device))
        for id in background_labels:
            # pad the background labels
            _point.append(torch.zeros(Np+Nn, 3).to(device)) # all 0
            _point_label.append(torch.zeros(Np+Nn).to(device) - 1) # -1 not a point
    else:
        Np = max_point
        Nn = 0
        _point, _point_label = point_sampler(unique_labels, Np=Np, Nn=Nn)
        for id in background_labels:
            # pad the background labels
            _point.append(torch.zeros(len(_point_label[0]), 3).to(device)) # all 0
            _point_label.append(torch.zeros(len(_point_label[0])).to(device) - 1) # -1 not a point


    if len(unique_labels) == 0 and len(background_labels) == 0:
        # the iteration should be skipped
        label_prompt, point, point_label, prompt_class = None, None, None, None
    else:
        label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
        point = torch.stack(_point)
        point_label = torch.stack(_point_label)
        prompt_class = copy.deepcopy(label_prompt)
    if random.uniform(0,1) < drop_label_prob and len(unique_labels) > 0:
        label_prompt = None
        #If label prompt is dropped, there is no need to pad with points with label -1.
        pad = len(background_labels)
        point = point[:len(point)-pad]
        point_label = point_label[:len(point_label)-pad]
        prompt_class = prompt_class[:len(prompt_class)-pad]
    else:
        if random.uniform(0,1) < drop_point_prob:
            point = None
            point_label = None
    if point is not None and convert_to_disc:
        point_mask = convert_points_to_disc(image_size, point, point_label, radius=radius)
    return label_prompt, point, point_label, prompt_class, point_mask

def find_lcc_label(plabels, region):
    plabels_org = plabels.clone()
    plabels = erode3d(plabels, erosion=3)
    label = measure.label
    features, num_features = label(plabels.cpu().numpy(), connectivity=3, return_num=True)
    features = torch.from_numpy(features).to(region.device)
    max_cc = torch.zeros_like(region)
    max_cc_count = 0
    region = dilate3d(region, erosion=5).to(torch.bool)
    for i in range(1, num_features):
        cc = features==i
        if torch.logical_and(cc, region).any():
            cc_sum = cc.sum()
            if cc_sum > max_cc_count:
                max_cc = cc
                max_cc_count = cc_sum
    return dilate3d(max_cc, erosion=5).to(torch.bool) * plabels_org
                        
def open_lcc(plabels):
    plabels_org = plabels.clone()
    plabels = erode3d(plabels, erosion=3)
    plabels = monai.transforms.utils.get_largest_connected_component_mask(plabels)
    return dilate3d(plabels, erosion=3).to(torch.bool) * plabels_org

class Point_sampler():
    def __init__(self, label, label_sv, map_shift=512, offset=10, min_size=125, vrange=[0.6, 0.7, 0.8, 0.9],):
        self.label = label
        self.label_sv = label_sv
        self.map_shift = map_shift
        self.shifted = {}
        self.device = self.label.device
        self.label_ = label.clone()
        self.use_debug_plot = False
        self.window = torch.ones_like(label, dtype=torch.bool)
        self.window[offset:-offset, offset:-offset, offset:-offset] = False
        self.min_size = min_size
        self.vrange = vrange
        self.zshot_rate = 0.5

    def reset(self):
        self.label = self.label_.clone()
    
    def debug_plot(self, region, region_sv, point, point_label, name='regular'):
        if not self.use_debug_plot:
            return
        marker_size=1
        fig, axs = plt.subplots(max(2, len(point)), 2)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        for b in range(len(point)):
            color = 'r' if point_label[b] == 1 else 'b'
            axs[b, 0].imshow(region.data.cpu().numpy()[:,:,point[b][-1].cpu()])
            axs[b, 0].plot(point[b][1].cpu(), point[b][0].cpu(), marker='*', color=color, markersize=marker_size)
            axs[b, 0].get_xaxis().set_ticks([])
            axs[b, 0].get_yaxis().set_ticks([])
            axs[b, 1].imshow(region_sv.data.cpu().numpy()[:,:,point[b][-1].cpu()])
            axs[b, 1].plot(point[b][1].cpu(), point[b][0].cpu(), marker='*', color=color, markersize=marker_size)
            axs[b, 1].get_xaxis().set_ticks([])
            axs[b, 1].get_yaxis().set_ticks([])
        fig.savefig(f"{name}-{np.random.randint(10000)}.png", dpi = 600)
        plt.close()

    def skip_aug(self, id):
        if id in SPECIAL_INDEX and ENABLE_SPECIAL:
            return True
        return False
    
    def __call__(self, unique_labels, Np=1, Nn=0):
        _point = []
        _point_label = []
        vrange = self.vrange
        self.read_only_id = copy.deepcopy(unique_labels)
        for id in unique_labels:
            if self.skip_aug(id):
                v = 0
                print(f'{id} will use regular')
            else:
                v = np.random.rand()
            if v < vrange[0] or self.label_sv is None:
                _p, _pl = self.regular(id, Np, Nn)
            if v >= vrange[0] and v < vrange[1]:
                _p, _pl = self.organ_sub(id, Np, Nn)
            if v >= vrange[1] and v < vrange[2]:
                _p, _pl = self.organ_add(id, Np, Nn)
            if v >= vrange[2] and v < vrange[3]:
                _p, _pl = self.zeroshot_unseen(id, Np, Nn)
            if v >= vrange[3]:
                _p, _pl = self.zeroshot_random(id, Np, Nn)
            _point.append(_p)
            _point_label.append(_pl)
        return self._padding(_point, _point_label)
    
    def _padding(self, point, point_label):
        if len(point) > 0:
            max_len = max([len(_) for _ in point])
            point = [torch.stack(_ + [torch.tensor([0,0,0]).to(self.device)] * (max_len - len(_))) for _ in point]
            point_label = [torch.tensor(_ + [-1] * (max_len - len(_))).to(self.device) for _ in point_label]
            # print(point, point_label)
        return point, point_label
    
    def regular(self, id, Np=1, Nn=0):
        print('regular')
        plabels = self.label == int(id)
        _plabels = erode3d(plabels)
        _plabels[self.window] = 0
        plabelpoints = torch.nonzero(_plabels)
        kp = min(len(plabelpoints), Np)
        if kp == 0:
            plabelpoints = torch.nonzero(plabels)
            kp = min(len(plabelpoints), Np)
        _point = random.choices(plabelpoints, k=kp)
        neg_id, pos_id = get_point_label(id)
        _point_label= [pos_id] * kp
        if Nn > 0:
            nlabels = ~plabels
            nlabels = erode3d(nlabels)
            nlabelpoints = torch.nonzero(nlabels)
            kn = min(len(nlabelpoints), Nn)
            if kn > 0:
                _point += random.choices(nlabelpoints, k=kn)
                _point_label += [neg_id] * kn
        # self.debug_plot(plabels, self.label_, _point, _point_label, name='regular')
        return _point, _point_label  
    
    def remove_half(self, plabels):
        centroids = torch.mean(torch.stack(torch.nonzero(plabels, as_tuple=True)).float(), dim=1).to(torch.int)
        axis = random.choices([0,1,2], k=1)[0]
        parts = random.choices([0,1], k=1)[0]
        if axis == 0:
            if parts == 0:
                plabels[:centroids[0]] = 0
            else:
                plabels[centroids[0]:] = 0
        if axis == 1:
            if parts == 0:
                plabels[:,:centroids[1]] = 0
            else:
                plabels[:,centroids[1]:] = 0
        if axis == 2:
            if parts == 0:
                plabels[:,:,:centroids[2]] = 0
            else:
                plabels[:,:,centroids[2]:] = 0 
        nlabels = erode3d(plabels)
        nlabelpoints = torch.nonzero(nlabels) 
        npoint = None
        if len(nlabelpoints) > 0:
            npoint = random.choices(nlabelpoints, k=1)               
        return plabels, npoint     
    
    def zeroshot_random(self, id, Np=1, Nn=0):
        min_size = 20*20*20 
        region = self.label == 0
        sregion = self.label_sv[region]
        sid = torch.unique(sregion).cpu().numpy()
        random.shuffle(sid)
        for sid_p in sid:
            plabels = self.label_sv == sid_p
            plabels = open_lcc(plabels)
            ids = self.label[plabels].unique().cpu().numpy()
            if np.array([i in ids for i in self.read_only_id]).any():
                continue
            if plabels.sum() < min_size:
                continue
            _point = []
            _point_label= []
            random.shuffle(ids)
            count = 0
            max_merge = len(ids)
            for i in ids:
                if i == 0:
                    continue
                if count >= max_merge:
                    break
                fg = self.label == i
                overlap = torch.logical_and(plabels, fg)
                if 0.1 * fg.sum() <  overlap.sum(): 
                    plabels = torch.logical_or(plabels, fg)
                    _plabels = erode3d(overlap)
                    _plabels[self.window] = 0            
                    plabelpoints = torch.nonzero(_plabels)
                    kp = min(len(plabelpoints), 1)
                    if kp == 0:
                        continue        
                    _point += random.choices(plabelpoints, k=kp)
                    _point_label += [1] * kp      
                    count += 1            
            _plabels = torch.logical_and(plabels, region)
            _plabels = erode3d(_plabels)
            _plabels[self.window] = 0      
            plabelpoints = torch.nonzero(_plabels)
            kp = min(len(plabelpoints), Np)
            if kp == 0:
                continue        
            _point += random.choices(plabelpoints, k=kp)
            _point_label += [1] * kp     
            self.label[plabels.to(torch.bool)] = id + self.map_shift
            self.shifted[id] = id + self.map_shift     
            self.read_only_id.append(id + self.map_shift) 
            print('zeroshot_random')
            self.debug_plot(plabels, self.label_, _point, _point_label, name='zeroshot_random') 
            return _point, _point_label       
        return self.regular(id, Np, Nn)  
    
    def zeroshot_unseen(self, id, Np=1, Nn=0):
        # PASSED
        min_size = 20*20*20 
        region = self.label == 0
        sregion = self.label_sv[region]
        sid = torch.unique(sregion).cpu().numpy()
        random.shuffle(sid)
        npoint = None
        for sid_p in sid:
            plabels = torch.logical_and(self.label_sv == sid_p, region)
            plabels = open_lcc(plabels)
            if plabels.sum() < min_size:
                continue
            ids = self.label[plabels].unique().cpu().numpy().tolist()
            if np.array([i in ids for i in self.read_only_id]).any():
                continue
            _plabels = erode3d(plabels)
            _plabels[self.window] = 0
            plabelpoints = torch.nonzero(_plabels)
            kp = min(len(plabelpoints), Np)
            if kp == 0:
                continue
            _point = random.choices(plabelpoints, k=kp)
            _point_label= [1] * kp
            if Nn > 0:
                nlabels = torch.logical_and(~plabels, region)
                nlabels = erode3d(nlabels)
                nlabelpoints = torch.nonzero(nlabels)
                kn = min(len(nlabelpoints), Nn)
                if kn > 0:
                    _point += random.choices(nlabelpoints, k=kn)
                    _point_label += [0] * kn
            if npoint is not None:
                _point += npoint
                _point_label += [0]
            self.label[plabels.to(torch.bool)] = id + self.map_shift
            self.shifted[id] = id + self.map_shift
            self.read_only_id.append(id + self.map_shift) 
            print('zeroshot_unseen')
            self.debug_plot(plabels, self.label_, _point, _point_label, name='zs_unseen')
            return _point, _point_label
        return self.regular(id, Np, Nn)  
     
    def remove_outside(self, plabels, region):
        """ Remove the points that does not lay within region slices (3 views). Avoid sample far away points
        """
        ps = torch.stack(torch.nonzero(region, as_tuple=True)).transpose(1,0)
        index = torch.ones_like(plabels).to(torch.bool)
        index[ps[:,0].min():ps[:,0].max(), ps[:,1].min():ps[:,1].max(), ps[:,2].min():ps[:,2].max()] = False
        plabels[index] = 0
        return plabels

     
    def organ_add(self, id, Np=1, Nn=0):
        """ For class id, find a supvervoxel index sid that mostly inside id.
        """
        lower_size = 0.1 # times
        region = self.label == id
        region_size = region.sum()
        all_region = self.label == 0
        sid = torch.unique(self.label_sv[region]).cpu().numpy()
        random.shuffle(sid)
        for sid_p in sid:
            plabels = torch.logical_and(self.label_sv == int(sid_p), all_region)
            if not plabels.any():
                continue
            plabels = find_lcc_label(plabels, region)
            ids = self.label[plabels].unique().cpu().numpy().tolist()
            if np.array([i in ids for i in self.read_only_id]).any():
                continue
            psize = plabels.sum()
            if psize < lower_size * region_size:
                continue
            _plabels = erode3d(plabels)
            _plabels[self.window] = 0
            _plabels = self.remove_outside(_plabels, region)
            # only pick points at the slice with the organ
            plabelpoints = torch.nonzero(_plabels)
            kp = min(len(plabelpoints), Np)
            if kp == 0:
                continue
            _point = random.choices(plabelpoints, k=kp)
            plabels_ = erode3d(region)
            plabels_[self.window] = 0
            plabelpoints = torch.nonzero(plabels_)
            kp2 = min(len(plabelpoints), Np)
            if kp2 == 0:
                plabelpoints = torch.nonzero(region) 
            kp2 = min(len(plabelpoints), Np)
            _point += random.choices(plabelpoints, k=kp2)
            self.label[plabels] = id
            _point_label = [1] * (kp + kp2)
            print('organ_add')
            self.debug_plot(self.label==id, self.label_==id, _point, _point_label, name='add')
            if np.random.rand() < self.zshot_rate:
                self.label[self.label==id] = id + self.map_shift
                self.shifted[id] = id + self.map_shift  
                self.read_only_id.append(id + self.map_shift) 
            return _point, _point_label
        return self.regular(id, Np, Nn)
        
   
    def organ_sub(self, id, Np=1, Nn=0):
        """ Substract a sid that is not too big or too small. 10%-50% of the id area size. At least 1 pos in remaining region in id and 1 neg 
            in subtracted region.
        """
        upper_size = 0.9 # times
        lower_size = 0.1 # times        
        region = self.label == id
        region_size = region.sum()
        sregion = self.label_sv[region]
        sid = torch.unique(sregion).cpu().numpy()
        random.shuffle(sid)
        use_zs = np.random.rand() < 0.5
        for sid_p in sid:
            # must contain 2 points within id and 1 point within sid_p
            if use_zs:
                plabels = torch.logical_and(self.label_sv == int(sid_p), region)
                plabels = open_lcc(plabels)
                psize = plabels.sum()
                if psize < lower_size * region_size or psize > upper_size * region_size:
                    continue
                nlabels = torch.logical_and(~plabels, region)
            else:
                nlabels = torch.logical_and(self.label_sv == int(sid_p), region)
                nlabels = open_lcc(nlabels)
                nsize = nlabels.sum()
                if nsize < lower_size * region_size or nsize > upper_size * region_size:
                    continue
                plabels = torch.logical_and(~nlabels, region)
            
            _plabels = erode3d(plabels)
            _plabels[self.window] = 0
            plabelpoints = torch.nonzero(_plabels)
            _nlabels = erode3d(nlabels)
            _nlabels[self.window] = 0
            nlabelpoints = torch.nonzero(_nlabels) 
            kp = min(len(plabelpoints), Np)
            kn = min(len(nlabelpoints), max(1, Nn)) 
            if kp == 0 or kn == 0:
                continue
            _point = random.choices(plabelpoints, k=kp) + random.choices(nlabelpoints, k=kn) 
            _point_label= [1] * kp + [0] * kn
            self.debug_plot(plabels, self.label_ == id, _point, _point_label, name='sub')
            if use_zs or np.random.rand() < self.zshot_rate:
                self.label[plabels.to(torch.bool)] = id + self.map_shift
                self.shifted[id] = id + self.map_shift
                self.read_only_id.append(id + self.map_shift) 
            else:
                self.label[nlabels.to(torch.bool)] = 0
            print('organ_sub')
            return _point, _point_label 
        _point, _point_label = self.regular(id, Np, Nn)
        return _point, _point_label 



def get_gaussian_ball(image_size, radius=None):
    if radius is None:
        radius = image_size[0] // 3
    row_array = torch.arange(start=0, end=image_size[0], step=1, dtype=torch.float32)
    col_array = torch.arange(start=0, end=image_size[1], step=1, dtype=torch.float32)
    z_array = torch.arange(start=0, end=image_size[2], step=1, dtype=torch.float32)
    coord_rows, coord_cols, coord_z = torch.meshgrid(z_array, col_array, row_array)
    coords = torch.stack((coord_rows, coord_cols, coord_z), dim=0)
    center = torch.tensor([image_size[0]//2, image_size[1]//2, image_size[2]//2]).to(coords.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ball =  torch.exp(-(((coords - center)**2).sum(0)/ (2 *radius ** 2))**2)
    return ball
    
def convert_points_to_disc(image_size, point, point_label, radius=2, disc=False):
    # [b, N, 3], [b, N]
    # generate masks [b,2,h,w,d]
    if not torch.is_tensor(point):
        point = torch.from_numpy(point)
    masks = torch.zeros([point.shape[0], 2, image_size[0], image_size[1],image_size[2]],device=point.device)
    row_array = torch.arange(start=0, end=image_size[0], step=1, dtype=torch.float32, device=point.device)
    col_array = torch.arange(start=0, end=image_size[1], step=1, dtype=torch.float32, device=point.device)
    z_array = torch.arange(start=0, end=image_size[2], step=1, dtype=torch.float32, device=point.device)
    coord_rows, coord_cols, coord_z = torch.meshgrid(z_array, col_array, row_array)
    # [1,3,h,w,d] -> [b, 2, 3, h,w,d]
    coords = torch.stack((coord_rows, coord_cols, coord_z), dim=0).unsqueeze(0).unsqueeze(0).repeat(point.shape[0], 2, 1, 1, 1, 1)
    for b in range(point.shape[0]):
        for n in range(point.shape[1]):
            if point_label[b,n] > -1:
                channel = 0 if (point_label[b,n] == 0 or point_label[b,n]==2) else 1
                if disc:
                    masks[b,channel] += torch.pow(coords[b,channel] - point[b,n].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),2).sum(0) < radius ** 2
                else:
                    masks[b,channel] += torch.exp(-torch.pow(coords[b,channel] - point[b,n].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),2).sum(0)/ (2 *radius ** 2)) 
    # masks[masks>1] = 1
    return masks

def get_next_points_val(pred, gt, prompt_class, point, point_label, pred_thresh=0.5, mapped=True, include_background=False, use_center=False, erosion2d=False, **kwargs):
    """ This function is used to sample points for iterative point evaluation. Each time only 1 point
    is sampled. background index will be ignored. 
    mapped: If the input prompt_class are mapped to the global index, we will use special index. If not mapped (zero-shot),
    the special index will not be enabled.
    """
    new_points = []
    new_points_label = []
    for id in range(len(prompt_class)):
        if prompt_class[id] == 0 and not include_background:
            new_points.append(torch.tensor([0,0,0], device=pred.device))
            new_points_label.append(torch.tensor(-1, device=pred.device))
            continue
        neg_id, pos_id = get_point_label(-1)
        _gt = (gt == prompt_class[id])[0,0]
        if mapped:
            # if in the global index, some supported classes need modification.
            neg_id, pos_id = get_point_label(prompt_class[id])
            if prompt_class[id].item() in MERGE_LIST.keys():
                for m in MERGE_LIST[prompt_class[id].item()]:
                    _gt = torch.logical_or(_gt, (gt==m)[0,0])    
        fn_mask = torch.logical_and(_gt, pred[id][0] < pred_thresh)
        if erosion2d:
            fn_mask = erode3d(fn_mask, erosion=(3,3,1))
        else:
            fn_mask = erode3d(fn_mask, erosion=(3,3,3))
        fn_mask = monai.transforms.utils.get_largest_connected_component_mask(fn_mask)
        fp_mask = torch.logical_and(torch.logical_not(_gt), pred[id][0] > pred_thresh)
        if erosion2d:
            fp_mask = erode3d(fp_mask, erosion=(3,3,1))
        else:
            fp_mask = erode3d(fp_mask, erosion=(3,3,3))
        fp_mask = monai.transforms.utils.get_largest_connected_component_mask(fp_mask)
        if (fn_mask.sum() >= fp_mask.sum()):
            plabelpoints = torch.nonzero(fn_mask)
            if len(plabelpoints) > 0:
                if use_center:
                    pdis = ((plabelpoints - plabelpoints.float().mean(0))**2).sum(-1)
                    _, sorted_indices = torch.sort(pdis)
                    new_points.append(plabelpoints[sorted_indices[0]])
                    new_points_label.append(torch.tensor(pos_id, device=pred.device))
                else:
                    new_points.append(random.choices(plabelpoints,k=1)[0])
                    new_points_label.append(torch.tensor(pos_id, device=pred.device))                    
                print('sampled pos')
            else:
                new_points.append(torch.tensor([0,0,0], device=pred.device))
                new_points_label.append(torch.tensor(-1, device=pred.device))           
        else:
            plabelpoints = torch.nonzero(fp_mask)
            if len(plabelpoints) > 0:
                if use_center:
                    pdis = ((plabelpoints - plabelpoints.float().mean(0))**2).sum(-1)
                    _, sorted_indices = torch.sort(pdis)
                    new_points.append(plabelpoints[sorted_indices[0]])
                    new_points_label.append(torch.tensor(neg_id, device=pred.device))
                else:
                    new_points.append(random.choices(plabelpoints,k=1)[0])
                    new_points_label.append(torch.tensor(neg_id, device=pred.device)) 
                print('sampled neg')
            else:
                new_points.append(torch.tensor([0,0,0], device=pred.device))
                new_points_label.append(torch.tensor(-1, device=pred.device))
    new_points = torch.stack(new_points).unsqueeze(1)
    new_points_label = torch.stack(new_points_label).unsqueeze(1)   
    point = torch.cat([point, new_points], dim=1)
    point_label = torch.cat([point_label, new_points_label], dim=1)
    return point, point_label

def get_next_points_auto_point(pred, gt, prompt_class, class_vector=None, pred_thresh=0.5, mapped=True, include_background=False, use_fg=False, **kwargs):
    """ sample points from foreground or error region. This function is only used during patch based auto + point evaluation. mapped is always true if
    evaluate dataset with automatic, which requires global index.
    """
    new_points = []
    new_points_label = []
    for id in range(len(prompt_class)):
        neg_id, pos_id = get_point_label(-1)
        if mapped:
            # if in the global index, some supported classes need modification. prompt_class is the local index
            if class_vector is not None:
                neg_id, pos_id = get_point_label(class_vector[id])
                if class_vector[id].item() in MERGE_LIST.keys():
                    for m in MERGE_LIST[class_vector[id].item()]:
                        _gt = torch.logical_or(_gt, (gt==m)[0,0])       
            else:
                neg_id, pos_id = get_point_label(prompt_class[id])
                if prompt_class[id].item() in MERGE_LIST.keys():
                    for m in MERGE_LIST[prompt_class[id].item()]:
                        _gt = torch.logical_or(_gt, (gt==m)[0,0])          
        _gt = (gt == prompt_class[id])[0,0]
        if (prompt_class[id] == 0 and not include_background) or _gt.sum() == 0:
            # if background or no foreground and no false positive
            if _gt.sum() == 0 and  (pred[id][0]>pred_thresh).sum() > 0:
                fp_mask = pred[id][0]>pred_thresh
                new_points.append(random.choices(torch.nonzero(fp_mask), k=1))
                new_points_label.append([torch.tensor(neg_id, device=pred.device)])
            else:
                new_points.append([torch.tensor([0,0,0], device=pred.device)])
                new_points_label.append([torch.tensor(-1, device=pred.device)])
            continue

            # DO NOT MERGE HERE. The merge classs evaluation will not perform merge.
        if use_fg:
            plabelpoints = torch.nonzero(_gt) 
            new_points.append(random.choices(plabelpoints, k=1))
            new_points_label.append([torch.tensor(pos_id, device=pred.device)])
            continue     
        _new_points, _new_points_label = [], []               
        fn_mask = torch.logical_and(_gt, pred[id][0] < pred_thresh)
        fn_mask = erode3d(fn_mask)
        fn_mask = monai.transforms.utils.get_largest_connected_component_mask(fn_mask)
        fp_mask = torch.logical_and(torch.logical_not(_gt), pred[id][0] > pred_thresh)
        fp_mask = erode3d(fp_mask)
        fp_mask = monai.transforms.utils.get_largest_connected_component_mask(fp_mask)
        if (fn_mask.sum() <= fp_mask.sum()):
            # if false positive is larger than false negative, we will sample a negative point and one from foreground.
            # if all of them are 0, will sample a foreground
            plabelpoints = torch.nonzero(_gt)
            _new_points.extend(random.choices(plabelpoints, k=1))
            _new_points_label.extend([torch.tensor(pos_id, device=pred.device)])    
            plabelpoints = torch.nonzero(fp_mask)
            if len(plabelpoints) > 0:
                _new_points.extend(random.choices(plabelpoints,k=1))
                _new_points_label.extend([torch.tensor(neg_id, device=pred.device)])                    
            else:
                _new_points.extend([torch.tensor([0,0,0], device=pred.device)])
                _new_points_label.extend([torch.tensor(-1, device=pred.device)])           
        else:
            plabelpoints = torch.nonzero(fn_mask)
            if len(plabelpoints) > 0:
                _new_points.extend(random.choices(plabelpoints,k=1))
                _new_points_label.extend([torch.tensor(pos_id, device=pred.device)]) 
            else:
                _new_points.extend([torch.tensor([0,0,0], device=pred.device)])
                _new_points_label.extend([torch.tensor(-1, device=pred.device)])
        new_points.append(_new_points)
        new_points_label.append(_new_points_label)

    max_len = max([len(_) for _ in new_points])
    x = []
    for _ in new_points:
        x.append(_ + [torch.tensor([0,0,0]).to(pred.device)] * (max_len - len(_)))
    new_points = torch.stack([torch.stack(_) for _ in x])
    # new_points = torch.stack([torch.stack(_) for _ in [x + [torch.tensor([0,0,0]).to(pred.device)] * (max_len - len(x)) for x in new_points]])
    x = []
    for _ in new_points_label:
        x.append(_ + [torch.tensor(-1).to(pred.device)] * (max_len - len(_)))
    new_points_label = torch.vstack([torch.stack(_) for _ in x])  

    return new_points, new_points_label


def get_next_points(pred, gt, prompt_class, point, point_label, pred_thresh=0.5, mapped=True, include_background=False, **kwargs):
    """ Iterative training. Sample points from false positve or false negative. This is used in training.
        pred [bs, 1, h, w, d]
        gt [1,1,h,w,d]
        point [bs, n, 3]
        Args:
        mapped: If the input prompt_class are mapped to the global index, we will use special index. If not mapped (zero-shot),
        the special index will not be enabled.
    """
    new_points = []
    new_points_label = []
    offset = 10
    window = torch.ones_like(pred[0,0], dtype=torch.bool)
    window[offset:-offset, offset:-offset, : ] = False
    for id in range(len(prompt_class)):
        if prompt_class[id] == 0 and not include_background:
            new_points.append(torch.tensor([0,0,0], device=pred.device))
            new_points_label.append(torch.tensor(-1, device=pred.device))
            continue
        neg_id, pos_id = get_point_label(-1)
        _gt = (gt == prompt_class[id])[0,0]
        if mapped:
            # if in the global index, some supported classes need modification.
            neg_id, pos_id = get_point_label(prompt_class[id])
            if prompt_class[id].item() in MERGE_LIST.keys():
                for m in MERGE_LIST[prompt_class[id].item()]:
                    _gt = torch.logical_or(_gt, (gt==m)[0,0])    
        fn_mask = torch.logical_and(_gt, pred[id][0] < pred_thresh)
        fn_mask = erode3d(fn_mask)
        fn_mask[window] = 0
        fp_mask = torch.logical_and(torch.logical_not(_gt), pred[id][0] > pred_thresh)
        fp_mask = erode3d(fp_mask)
        fp_mask[window] = 0
        # random select a false negative
        fnlabelpoints = torch.nonzero(fn_mask)
        fplabelpoints = torch.nonzero(fp_mask)
        _new_points = []
        _new_points_label = []
        if len(fnlabelpoints) > 0:
            _new_points.append(random.choices(fnlabelpoints,k=1)[0])
            _new_points_label.append(torch.tensor(pos_id, device=pred.device))
        else:
            _new_points.append(torch.tensor([0,0,0], device=pred.device))
            _new_points_label.append(torch.tensor(-1, device=pred.device))
        if len(fplabelpoints) > 0:
            _new_points.append(random.choices(fplabelpoints,k=1)[0])
            _new_points_label.append(torch.tensor(neg_id, device=pred.device))
        else:
            _new_points.append(torch.tensor([0,0,0], device=pred.device))
            _new_points_label.append(torch.tensor(-1, device=pred.device))
        new_points.append(torch.stack(_new_points))
        new_points_label.append(torch.stack(_new_points_label))
    if len(new_points) > 0:
        new_points = torch.stack(new_points)
        new_points_label = torch.stack(new_points_label)     
        if point is not None:
            point = torch.cat([point, new_points], dim=1)
            point_label = torch.cat([point_label, new_points_label], dim=1)
        else:
            point = new_points
            point_label = new_points_label
            
    return point, point_label