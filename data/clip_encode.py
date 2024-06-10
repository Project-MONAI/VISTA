import clip
import torch

from .datasets import get_label_dict_k_folds, get_class_names

"""
['background', 'liver', 'kidney', 'spleen', 'pancreas', 'right kidney', 'aorta', 'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney', 'postcava', 'bladder', 'prostate or uterus', 'portal vein and splenic vein', 'uterus', 'rectum', 'small bowel', 'lung', 'bone', 'brain', 'lung tumor', 'pancreatic tumor', 'hepatic vessel', 'hepatic tumor', 'colon cancer primaries', 'left lung upper lobe', 'left lung lower lobe', 'right lung upper lobe', 'right lung middle lobe', 'right lung lower lobe', 'vertebrae L5', 'vertebrae L4', 'vertebrae L3', 'vertebrae L2', 'vertebrae L1', 'vertebrae T12', 'vertebrae T11', 'vertebrae T10', 'vertebrae T9', 'vertebrae T8', 'vertebrae T7', 'vertebrae T6', 'vertebrae T5', 'vertebrae T4', 'vertebrae T3', 'vertebrae T2', 'vertebrae T1', 'vertebrae C7', 'vertebrae C6', 'vertebrae C5', 'vertebrae C4', 'vertebrae C3', 'vertebrae C2', 'vertebrae C1', 'trachea', 'heart myocardium', 'left heart atrium', 'left heart ventricle', 'right heart atrium', 'right heart ventricle', 'pulmonary artery', 'left iliac artery', 'right iliac artery', 'left iliac vena', 'right iliac vena', 'colon', 'left rib 1', 'left rib 2', 'left rib 3', 'left rib 4', 'left rib 5', 'left rib 6', 'left rib 7', 'left rib 8', 'left rib 9', 'left rib 10', 'left rib 11', 'left rib 12', 'right rib 1', 'right rib 2', 'right rib 3', 'right rib 4', 'right rib 5', 'right rib 6', 'right rib 7', 'right rib 8', 'right rib 9', 'right rib 10', 'right rib 11', 'right rib 12', 'left humerus', 'right humerus', 'left scapula', 'right scapula', 'left clavicula', 'right clavicula', 'left femur', 'right femur', 'left hip', 'right hip', 'sacrum', 'face', 'left gluteus maximus', 'right gluteus maximus', 'left gluteus medius', 'right gluteus medius', 'left gluteus minimus', 'right gluteus minimus', 'left autochthon', 'right autochthon', 'left iliopsoas', 'right iliopsoas']


to generate the embedding for the text, we use the following code::

    python -m data.clip_encode   # output shape [117, 512] float32
    mv clip_embedding_117_v3_structures.pth segment_anything3d/modeling

"""

ORGAN_NAME = get_class_names()
print("labels", len(ORGAN_NAME))
print("labels", ORGAN_NAME)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
text_inputs = torch.cat([clip.tokenize(f"A computerized tomography of {item}") for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, "clip_embedding_117_v3_structures.pth")
