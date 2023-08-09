import torch
import torch.nn.functional as F

import random
import pdb
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
def debug_next_point(output, point, point_label, point_mask, labels, prompt_class, idx):
    bs = output.shape[0]
    fig, axs = plt.subplots(bs, 6)
    marker_size=1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    for b in range(bs):
        fn, fp = point[b,-2,:].cpu().numpy(), point[b,-1,:].cpu().numpy()
        axs[b, 0].imshow(output.data.cpu().numpy()[b,0,:,:,fn[2]]>0.5, vmin=0, vmax=1)
        axs[b, 0].plot(fn[1], fn[0], marker='*', color='r', markersize=marker_size)
        axs[b, 0].get_xaxis().set_ticks([])
        axs[b, 0].get_yaxis().set_ticks([])
        # axs[b, 0].text(fn[1], fn[0], f"{point_label[b,-2].data.cpu().numpy()}")
        # axs[b, 0].set_title(label=f"{output.data.cpu().numpy()[b,0,fn[0],fn[1],fn[2]]:.2f}")
        axs[b, 1].imshow(labels.data.cpu().numpy()[0,0,:,:,fn[2]] == prompt_class[b][0].data.cpu().numpy(), vmin=0, vmax=1)
        axs[b, 1].plot(fn[1], fn[0], marker='*', color='r', markersize=marker_size)
        axs[b, 1].get_xaxis().set_ticks([])
        axs[b, 1].get_yaxis().set_ticks([])
        # axs[b, 1].set_title(label=f"{labels.data.cpu().numpy()[0,0,fn[0],fn[1],fn[2]]:.2f}")
        axs[b, 2].imshow(output.data.cpu().numpy()[b,0,:,:,fp[2]]>0.5, vmin=0, vmax=1)
        axs[b, 2].plot(fp[1], fp[0], marker='*', color='b', markersize=marker_size)
        axs[b, 2].get_xaxis().set_ticks([])
        axs[b, 2].get_yaxis().set_ticks([])
        # axs[b, 2].set_title(label=f"{output.data.cpu().numpy()[b,0,fp[0],fp[1],fp[2]]:.2f}")
        # axs[b, 2].text(fp[1], fp[0], f"{point_label[b,-1].data.cpu().numpy()}")
        axs[b, 3].imshow(labels.data.cpu().numpy()[0,0,:,:,fp[2]] == prompt_class[b][0].data.cpu().numpy(), vmin=0, vmax=1)
        axs[b, 3].plot(fp[1], fp[0], marker='*', color='b', markersize=marker_size)
        axs[b, 3].get_xaxis().set_ticks([])
        axs[b, 3].get_yaxis().set_ticks([])
        # axs[b, 3].set_title(label=f"{labels.data.cpu().numpy()[0,0,fp[0],fp[1],fp[2]]}")
        axs[b, 4].imshow(labels.data.cpu().numpy()[0,0,:,:,fn[2]] == prompt_class[b][0].data.cpu().numpy(), vmin=0, vmax=1)
        axs[b, 4].get_xaxis().set_ticks([])
        axs[b, 4].get_yaxis().set_ticks([])
        axs[b, 5].imshow(labels.data.cpu().numpy()[0,0,:,:,fp[2]] == prompt_class[b][0].data.cpu().numpy(), vmin=0, vmax=1)
        axs[b, 5].get_xaxis().set_ticks([])
        axs[b, 5].get_yaxis().set_ticks([])
    fig.savefig(f"{idx}.png", dpi = 600)

def debug_erosion(inputs, eroded, name):
    """ Usage: inputs is the original diff, eroded is the output from erode3d       
            debug_erosion(_fn_mask, fn_mask, f"{id}_fn.png")
            debug_erosion(_fp_mask, fp_mask, f"{id}_fp.png")
    """
    slice_to_show = eroded.sum(0).sum(0).argmax()
    fig, axis = plt.subplots(2)
    axis[0].imshow(inputs[:,:,slice_to_show].data.cpu().numpy())
    axis[1].imshow(eroded[:,:,slice_to_show].data.cpu().numpy())
    fig.savefig(name)

def erode3d(input_tensor, erosion=3):
    # Define the structuring element
    structuring_element = torch.ones(1, 1, erosion, erosion, erosion).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(input_tensor.float().unsqueeze(0).unsqueeze(0), (erosion//2, erosion//2, erosion//2, erosion//2, erosion//2, erosion//2), mode='constant', value=1.0)

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)

def generate_prompt_pairs_val(labels, label_set=None, image_size=None, max_point=1, 
                          convert_to_disc=False, radius=2, device='cpu'):
    """ 
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
    point_mask = None
    label_prompt = torch.tensor(label_set).to(device).unsqueeze(-1)
    unique_labels = np.unique(labels.cpu().numpy()).tolist()
    _point = []
    _point_label = []
    Nn = max_point//2
    Np = max_point - Nn
    for id in label_set:
        if id in unique_labels:
            plabels = labels == int(id)
            nlabels = ~plabels
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
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
    if point is not None and convert_to_disc:
        point_mask = convert_points_to_disc(image_size, point, point_label, radius=radius)
    return label_prompt, point, point_label, prompt_class, point_mask

def generate_prompt_pairs(labels, label_set=None, image_size=None, max_prompt=None, max_backprompt=1, max_point=20, 
                          include_background=True, drop_label_prob=0.2, drop_point_prob=0.2, 
                          convert_to_disc=True, radius=2):
    """ 
    Args:
        labels: torch.tensor from dataload, [1,1,H,W,D]
        label_set: the label list for the specific dataset
        total_prompt: int, number of total prompt
        max_point: maximum number of points for each object
        include_background: if include label=0 into training prompt. May casue issue in partial label
                            trainig.
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
    unique_labels = np.unique(labels.cpu().numpy())
    if include_background:
        unique_labels = list(unique_labels)
    else:
        unique_labels = list(set(unique_labels) - set([0]))
    background_labels = list(set(label_set) - set(unique_labels))
    # during training, balance background and foreground prompts
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            if len(unique_labels) == 0:
                # avoid training error if both prompt are None
                max_backprompt = max(1, max_backprompt)
            background_labels = background_labels[:max_backprompt]
    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                unique_labels = random.sample(unique_labels, max_prompt)
                background_labels = []
            else:
                background_labels = random.sample(background_labels, max_prompt - len(unique_labels))
    _point = []
    _point_label = []     
    Np = min(max_point, int(np.abs(random.gauss(mu=0,sigma=max_point//2)))+1)
    Nn = min(max_point, int(np.abs(random.gauss(mu=0,sigma=max_point//2))))
    for id in unique_labels:
        plabels = labels == int(id)
        nlabels = ~plabels
        plabelpoints = torch.nonzero(plabels)
        nlabelpoints = torch.nonzero(nlabels)
        _point.append(torch.stack(random.choices(plabelpoints, k=min(len(plabelpoints), Np)) + 
                                  random.choices(nlabelpoints, k=min(len(nlabelpoints), Nn)) + 
                                  [torch.tensor([0,0,0],device=device)] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))))
        _point_label.append(torch.tensor([1] * min(len(plabelpoints), Np) + [0.] * min(len(nlabelpoints), Nn) + \
                                         [-1] * (Np +Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))).to(device))
    for id in background_labels:
        # pad the background labels
        _point.append(torch.zeros(Np+Nn, 3).to(device)) # all 0
        _point_label.append(torch.zeros(Np+Nn).to(device) - 1) # -1 not a point
    label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    prompt_class = copy.deepcopy(label_prompt)
    if random.uniform(0,1) < drop_label_prob and len(unique_labels) > 0:
        label_prompt = None
        # drop out the padded 
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
    
def convert_points_to_disc(image_size, point, point_label, radius=2):
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
                masks[b,int(point_label[b,n])] += torch.exp(-torch.pow(coords[b,int(point_label[b,n])] - point[b,n].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),2).sum(0)/ (2 *radius ** 2)) 
    # masks[masks>1] = 1
    return masks
    
def get_next_points(pred, gt, prompt_class, point, point_label, image_size=None, pred_thresh=0.5, previous_mask=None, convert_to_disc=False, radius=5, use_erosion=False):
    """ Iterative training. Sample points from false positve or false negative
        pred [bs, 1, h, w, d]
        gt [1,1,h,w,d]
        point [bs, n, 3]
    """

    new_points = []
    new_points_label = []
    # mask_new must be previous_mask if no new point added.
    mask_new = previous_mask
    for id in range(len(prompt_class)):
        _gt = (gt == prompt_class[id])[0,0]
        fn_mask = torch.logical_and(_gt, pred[id][0] < pred_thresh)
        if use_erosion:
            fn_mask = erode3d(fn_mask)
        fp_mask = torch.logical_and(torch.logical_not(_gt), pred[id][0] > pred_thresh)
        if use_erosion:
            fp_mask = erode3d(fp_mask)
        # debug_erosion(_fn_mask, fn_mask, f"{id}_fn.png")
        # debug_erosion(_fp_mask, fp_mask, f"{id}_fp.png")
        # random select a false negative
        fnlabelpoints = torch.nonzero(fn_mask)
        fplabelpoints = torch.nonzero(fp_mask)
        _new_points = []
        _new_points_label = []
        if len(fnlabelpoints) > 0:
            _new_points.append(random.choices(fnlabelpoints,k=1)[0])
            _new_points_label.append(torch.tensor(1, device=pred.device))
        else:
            _new_points.append(torch.tensor([0,0,0], device=pred.device))
            _new_points_label.append(torch.tensor(-1, device=pred.device))
        if len(fplabelpoints) > 0:
            _new_points.append(random.choices(fplabelpoints,k=1)[0])
            _new_points_label.append(torch.tensor(0, device=pred.device))
        else:
            _new_points.append(torch.tensor([0,0,0], device=pred.device))
            _new_points_label.append(torch.tensor(-1, device=pred.device))  
        new_points.append(torch.stack(_new_points))
        new_points_label.append(torch.stack(_new_points_label))
    if len(new_points) > 0:
        new_points = torch.stack(new_points)
        new_points_label = torch.stack(new_points_label)     
        if (new_points_label > -1).any():
            if point is not None:
                point = torch.cat([point, new_points], dim=1)
                point_label = torch.cat([point_label, new_points_label], dim=1)
            else:
                point = new_points
                point_label = new_points_label
            if convert_to_disc:
                # speed up the mask generation by reusing the previous one
                if previous_mask is None:
                    mask_new = convert_points_to_disc(image_size, point, point_label, radius=radius)
                else:
                    mask_new = convert_points_to_disc(image_size, new_points, new_points_label, radius=radius)
                    # combine two masks
                    assert mask_new.shape == previous_mask.shape
                    mask_new = torch.logical_or(mask_new, previous_mask)
    return point, point_label, mask_new

def get_window_idx_c(p, roi, s):
    if p - roi//2 < 0:
        l, r = 0, roi
    elif p + roi//2 > s:
        l, r = s - roi, s
    else:
        l, r = int(p)-roi//2, int(p)+roi//2
    return l, r

def get_window_idx(p, roi, s, center_only=True, margin=5):
    l, r = get_window_idx_c(p, roi, s)
    if center_only:
        return [l], [r]
    left_most = max(0, p - roi + margin)
    right_most = min(s, p + roi - margin)
    left = [left_most, right_most-roi, l]
    right = [left_most + roi, right_most, r]
    return left, right

def pad_previous_mask(inputs, roi_size, padvalue=0):
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = torch.nn.functional.pad(inputs, pad=pad_size, mode="constant", value=padvalue)    
    return inputs, pad_size
    
def point_based_window_inferer(inputs, roi_size, sw_batch_size,  predictor, mode, overlap, sw_device, device, point_coords, point_labels, class_vector, masks, point_mask):
    # point_coords: [1,N,3]
    image, pad = pad_previous_mask(copy.deepcopy(inputs), roi_size)
    stitched_output = None
    center_only = True
    for p in point_coords[0]:
        lx_, rx_ = get_window_idx(p[0], roi_size[0], image.shape[-3], center_only=center_only, margin=5)
        ly_, ry_ = get_window_idx(p[1], roi_size[1], image.shape[-2], center_only=center_only, margin=5)
        lz_, rz_ = get_window_idx(p[2], roi_size[2], image.shape[-1], center_only=center_only, margin=5)
        for i in range(len(lx_)):
            for j in range(len(ly_)):
                for k in range(len(lz_)):
                    lx, rx, ly, ry, lz, rz = lx_[i], rx_[i], ly_[j], ry_[j], lz_[k], rz_[k]
                    unravel_slice = [slice(None), slice(None), slice(int(lx), int(rx)), slice(int(ly), int(ry)), slice(int(lz), int(rz))]
                    batch_image = image[unravel_slice]
                    ball = get_gaussian_ball(batch_image.shape[-3:])
                    output = predictor(batch_image, 
                                point_coords=point_coords,
                                point_labels=point_labels,
                                class_vector=class_vector,
                                patch_coords=unravel_slice,
                                masks=masks,
                                point_mask=point_mask)
                    if stitched_output is None:
                        stitched_output = torch.zeros([1, output.shape[1], image.shape[-3],image.shape[-2],image.shape[-1]],device='cpu')
                        stitched_mask = torch.zeros([1, output.shape[1], image.shape[-3],image.shape[-2],image.shape[-1]],device='cpu')
                    stitched_output[unravel_slice] += ball * output.to('cpu')
                    stitched_mask[unravel_slice] = 1
    # if stitched_mask is 0, then NaN value
    stitched_output = stitched_output/stitched_mask
    # revert padding
    stitched_output = stitched_output[:,:,pad[4]:image.shape[-3]-pad[5], pad[2]:image.shape[-2]-pad[3], pad[0]:image.shape[-1]-pad[1]]
    stitched_mask = stitched_mask[:,:,pad[4]:image.shape[-3]-pad[5], pad[2]:image.shape[-2]-pad[3], pad[0]:image.shape[-1]-pad[1]]
    if masks is not None:
        masks = masks.to('cpu')
        # for un-calculated place, use previous mask
        stitched_output[stitched_mask < 1] = masks[stitched_mask < 1]
        # for calculated place, use weighted avg.
        avg_mask = torch.logical_and(stitched_mask > 1, ~torch.isnan(masks))
        stitched_output[avg_mask] = 0.5*stitched_output[avg_mask] +  0.5*masks[avg_mask]
    return stitched_output

def _point_based_window_inferer(inputs, roi_size, sw_batch_size,  predictor, mode, overlap, sw_device, device, point_coords, point_labels, class_vector, masks, point_mask):
    # point_coords: [1,N,3]
    image, pad = pad_previous_mask(copy.deepcopy(inputs), roi_size)
    stitched_output = torch.zeros([point_coords.shape[0],1,image.shape[-3],image.shape[-2],image.shape[-1]],device=device)
    stitched_mask = torch.zeros([point_coords.shape[0],1,image.shape[-3],image.shape[-2],image.shape[-1]],device=device)
    for i, p in enumerate(point_coords[0]):
        lx, rx = get_window_idx(p[0], roi_size[0], image.shape[-3])
        ly, ry = get_window_idx(p[1], roi_size[1], image.shape[-2])
        lz, rz = get_window_idx(p[2], roi_size[2], image.shape[-1])
        unravel_slice = [slice(None), slice(None), slice(int(lx), int(rx)), slice(int(ly), int(ry)), slice(int(lz), int(rz))]
        batch_image = image[unravel_slice]
        # the predictor is model_inferer wrapper which transpose the mask
        output = predictor(batch_image, 
                       point_coords=point_coords,
                       point_labels=point_labels,
                       class_vector=class_vector,
                       patch_coords=unravel_slice,
                       masks=masks,
                       point_mask=point_mask).transpose(1,0)
        stitched_output[unravel_slice] += output.to(device)
        stitched_mask[unravel_slice] += ~(output[:,:,0,0,0] == -torch.inf).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    untouched_mask = stitched_mask==0
    stitched_mask[untouched_mask] = 1
    stitched_output[untouched_mask] = -torch.inf
    stitched_output = stitched_output/stitched_mask
    # revert padding
    stitched_output = stitched_output[:,:,pad[4]:image.shape[-3]-pad[5], pad[2]:image.shape[-2]-pad[3], pad[0]:image.shape[-1]-pad[1]].transpose(1,0)
    return stitched_output

if __name__ == '__main__':
    label = torch.zeros(1,1,96,96,96)
    label[...,:10,:10,:10] = 1
    label[...,10:20,10:20,10:20] = 2
    label_set = [1,2,3,4]
    max_point = 20
    label_prompt, point, point_label, prompt_class = generate_prompt_pairs(label, label_set, max_point)
    # print(label_prompt)
    # print(point)
    # print(point_label)
    # print(prompt_class)
    # test window inferer
    label = torch.zeros(1,1,196,196,196)
    label[...,:10,:10,:10] = 1
    label[...,10:20,10:20,10:20] = 2
    label_set = [1,2,3,4]
    max_point = 20
    label_prompt, point, point_label, prompt_class = generate_prompt_pairs(label, label_set, max_point)
    image = torch.zeros(1,1,196,196,196)
    model = lambda x,point_coords,point_labels,class_vector,patch_coords,masks: x
    roi_size=(96,96,96)
    masks = point_based_window_inferer(label, model, roi_size, point, point_label, None, None)
    print(masks)
