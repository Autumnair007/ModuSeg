import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append("..")

import numpy as np
from PIL import Image
from pathlib import Path

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmengine.structures import PixelData

from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn

from open_clip import create_model, tokenizer
from myutils import UnNormalize
from prompts.imagenet_template import openai_imagenet_template
from project_utils.imagelevel_utils import load_imagelevel_labels, build_class_mask

try:
    from CropFormer.demo_mask2former.demo import get_entityseg
    from detectron2.data.detection_utils import read_image
except:
    print("EntitySeg is not installed")


@MODELS.register_module()
class CorrCLIPSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, dino_type, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=336, instance_mask_path=None, mask_generator=None,
                 imagelevel_json_path=None, dataset_type=None):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer.tokenize
        self.dino = torch.hub.load('facebookresearch/dino:main', dino_type)
        
        self.dino.eval().to(device)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino = self.dino.half()

        self.dino_qkv_output = None
        self.dino.blocks[-1].attn.qkv.register_forward_hook(self._hook_fn_forward_qkv)

        self.dummy = nn.Linear(1, 1)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.generate_category_embeddings(name_path, device)

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.instance_mask_path = instance_mask_path
        self.device = device

        # Image-level label filtering
        self.use_imagelevel_filter = imagelevel_json_path is not None
        self.dataset_type = dataset_type
        self.imagelevel_json_path = imagelevel_json_path
        if self.use_imagelevel_filter:
            self.imagelevel_labels = load_imagelevel_labels(json_path=imagelevel_json_path)
            if not self.imagelevel_labels:
                raise ValueError(f"Failed to load image-level labels: {imagelevel_json_path}")
        else:
            self.imagelevel_labels = None
        
        self._current_imagelevel_mask = None

        self.set_mask_generator(mask_generator)

    def _hook_fn_forward_qkv(self, module, input, output):
        self.dino_qkv_output = output

    @torch.inference_mode()
    def forward_feature(self, img, masks):
        if type(img) == list:
            img = img[0]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        # Forward pass in the model
        self.dino_qkv_output = None
        feat = self.dino.get_intermediate_layers(imgs_norm, n=1)[-1]

        patch_size = self.dino.patch_embed.patch_size
        feat_shape = (imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size)
        nb_im = feat.shape[0]  # Batch size
        nb_tokens = feat.shape[1]  # Number of tokens

        qkv = self.dino_qkv_output.reshape(nb_im, nb_tokens, 3, -1).permute(2, 0, 1, 3)
        dino_feats = qkv[0] + qkv[1]  #B, L, C
        dino_feats = dino_feats[:, 1:, ]
        dino_feats = F.normalize(dino_feats, dim=-1)

        image_features = self.clip.encode_image(img.half(), dino_feats=dino_feats, feat_shape=feat_shape, instance_masks=masks)

        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], *feat_shape)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')

        return logits

    def forward_slide(self, img, instance_masks, img_metas, stride=112, crop_size=224):
        """Sliding window inference over large images."""
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_instance_masks = instance_masks[:, :, y1:y2, x1:x2]

                # Pad if needed
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)
                    crop_instance_masks = nn.functional.pad(crop_instance_masks, pad, value=10000)
                
                # Extract features
                crop_seg_logit = self.forward_feature(crop_img, crop_instance_masks).detach()

                # Remove padding
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                # Accumulate
                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        
        assert (count_mat == 0).sum() == 0

        # Average over overlapping regions
        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        torch.cuda.empty_cache()

        return logits

    def predict(self, inputs, data_samples):
        self._current_imagelevel_mask = None

        if data_samples is None:
            inputs, img_path = inputs
            batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
            instance_masks = (self.generate_mask(img_path)).unsqueeze(0)

            # Image-level label filtering
            if self.use_imagelevel_filter:
                img_stem = Path(img_path).stem
                # COCO: COCO_train2014_000000057870 -> 57870
                # VOC: 2007_000032 -> 2007_000032
                if img_stem.startswith('COCO_'):
                    img_id = img_stem.split('_')[-1].lstrip('0') or '0'
                else:
                    img_id = img_stem
                
                allowed = self.imagelevel_labels.get(img_id)
                if allowed is None:
                    raise KeyError(f"Image {img_id} not found in image-level labels")
                
                class_mask = build_class_mask(
                    allowed=allowed,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                if class_mask is None:
                    raise ValueError(f"Failed to build class mask for image {img_id}")

                self._current_imagelevel_mask = class_mask[self.query_idx.to(class_mask.device)]
        else:   # evaluation
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
            instance_masks = [self.generate_mask(data_sample.img_path) for data_sample in data_samples]
            instance_masks = torch.stack(instance_masks, dim=0)

        self.instance_masks = instance_masks.int()
        instance_masks = F.interpolate(instance_masks.unsqueeze(1).float(), size=inputs.shape[2:], mode='nearest').int()

        seg_logits = self.forward_slide(inputs, instance_masks, batch_img_metas, self.slide_stride, self.slide_crop)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logit = seg_logits[i] * self.logit_scale  # [C, H, W]

            # Image-level label filtering: mask out disallowed classes
            if self.use_imagelevel_filter:
                mask = getattr(self, "_current_imagelevel_mask", None)
                if mask is None:
                    raise RuntimeError("Image-level label mask not set")
                
                deny = (~mask).view(-1, 1, 1)
                seg_logit = seg_logit.masked_fill(deny, float('-inf'))

            seg_logit = seg_logit.softmax(0)  # n_queries * h * w

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits_background = seg_logit[:num_queries - num_cls + 1]
                seg_logits_background = seg_logits_background.max(0, keepdim=True)[0]
                seg_logits_stuff = seg_logit[num_queries - num_cls + 1:]
                seg_logit = torch.cat([seg_logits_background, seg_logits_stuff])

            seg_pred = seg_logit.argmax(0, keepdim=True)
            seg_pred[seg_logit.max(0, keepdim=True)[0] < self.prob_thd] = 0

            # Map Correction
            mask_values = torch.unique(self.instance_masks[i])
            mask_values = mask_values[1:]
            masks = mask_values.unsqueeze(1).unsqueeze(1) == self.instance_masks[i].unsqueeze(0).expand(len(mask_values), -1, -1)
            masks = masks.unsqueeze(1)
            for mask in masks:
                seg_pred[mask] = torch.mode(seg_pred[mask])[0]

            if data_samples is None:    # demo_gradio
                return seg_pred
            else:   # evaluation
                data_samples[i].set_data({
                    'seg_logit':
                        PixelData(**{'data': seg_logit}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def generate_category_embeddings(self, name_path, device=torch.device('cuda')):
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.inference_mode():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

    def set_mask_generator(self, generator_type):
        self.mask_generator_type = generator_type
        if generator_type == 'entityseg':
            self.confidence_threshold = 0.5
            ckpt_path = os.path.join('pretrain_model', 'Mask2Former_hornet_3x_576d0b.pth')
            self.mask_generator = get_entityseg(cfg_file="mask2former_hornet_3x.yaml", ckpt_path=ckpt_path)

    def generate_mask(self, img_path):
        # Output: [H, W] int tensor

        if self.mask_generator_type is None:
            instance_mask = np.load(os.path.join(self.instance_mask_path, Path(img_path).stem + '.npz'))['instance_mask']
            instance_mask = torch.from_numpy(instance_mask).to(self.device)
        elif self.mask_generator_type == 'entityseg':
            img = read_image(img_path, format="BGR")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                predictions = self.mask_generator(img)

            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores

            selected_indexes = (pred_scores >= self.confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            instance_mask = torch.zeros((m_H, m_W), dtype=torch.int, device=self.device)

            selected_scores, ranks = torch.sort(selected_scores)
            ranks = ranks + 1
            for index in ranks:
                instance_mask[(selected_masks[index - 1] == 1)] = int(index)
        else:
            instance_mask = None

        return instance_mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
