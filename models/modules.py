from pyts.image import RecurrencePlot, MarkovTransitionField, GramianAngularField

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
from typing import Optional, Tuple, Union, List


class MTA(nn.Module):
    """Multimodal Temporal-attention Adapter"""

    def __init__(self, feature_dim: int, reduction_ratio: int = 8):
        super().__init__()
        self.feature_dim = feature_dim

        # Query, Key, Value projections for cross attention
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Bottleneck module
        self.omega_alpha = nn.Linear(feature_dim, feature_dim // reduction_ratio)
        self.omega_beta = nn.Linear(feature_dim // reduction_ratio, feature_dim)

    def forward(self, F_f_l: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_f_l: Feature map from CLIP's visual encoder
            H: Time series features
        """
        Q = self.q_proj(F_f_l)
        K = self.k_proj(H)
        V = self.v_proj(H)

        # Cross attention
        attention_weights = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.feature_dim),
            dim=-1
        )
        F_a_f_l = torch.matmul(attention_weights, V)

        # Bottleneck transformation
        F_a_f_l_prime = self.omega_beta(F.relu(self.omega_alpha(F_a_f_l)))
        return F_a_f_l_prime


class ModifiedCLIPVisualEncoder(nn.Module):
    """Modified CLIP Visual Encoder with MTA layers"""

    def __init__(self, original_encoder, feature_dim: int = 768, reduction_ratio: int = 8):
        super().__init__()
        self.feature_dim = feature_dim

        # Create layer list from original encoder
        self.layers = nn.ModuleList()
        stages = [
            original_encoder.embeddings.patch_embedding,
            original_encoder.pre_layrnorm,
            *original_encoder.encoder.layers
        ]

        # Insert MTA after specific transformer layers
        for i, stage in enumerate(stages):
            self.layers.append(stage)
            if i in [4, 8, 11]:  # After selected transformer layers
                self.layers.append(MTA(feature_dim=feature_dim, reduction_ratio=reduction_ratio))

        self.post_layernorm = original_encoder.post_layernorm
        self.pooler = original_encoder.pooler

    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = x

        for layer in self.layers:
            if isinstance(layer, MTA):
                if time_features is not None:
                    hidden_states = layer(hidden_states, time_features)
            else:
                hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        pooled = self.pooler(hidden_states)
        return pooled


class ImageEncoder:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.rp = RecurrencePlot(size=image_size, flatten=False)
        self.mtf = MarkovTransitionField(size=image_size, n_bins=10)
        self.gaf = GramianAngularField(size=image_size, method='summation', flatten=False)

    def encode(self, x: np.ndarray) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        images = []

        for b in range(batch_size):
            img_batch = []
            for c in range(channels):
                ts = x[b, c]
                rp_img = self.rp.transform(ts.reshape(1, -1))[0]
                mtf_img = self.mtf.transform(ts.reshape(1, -1))[0]
                gaf_img = self.gaf.transform(ts.reshape(1, -1))[0]
                img_batch.extend([rp_img, mtf_img, gaf_img])
            images.append(img_batch)

        images = torch.tensor(images).float()
        return images.view(batch_size, channels * 3, self.image_size, self.image_size)


class TimeSeriesUDA(nn.Module):
    def __init__(self, num_classes: int, clip_model: str = "openai/clip-vit-base-patch32",
                 temperature: float = 0.07, reduction_ratio: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.device = device

        # CLIP model and processors
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)

        # Modified visual encoder with MTA
        self.clip_model.vision_model = ModifiedCLIPVisualEncoder(
            original_encoder=self.clip_model.vision_model,
            feature_dim=self.clip_model.config.hidden_size,
            reduction_ratio=reduction_ratio
        )

        # Domain-specific prompt tokens
        self.num_domain_tokens = 10
        self.domain_tokens = nn.Parameter(
            torch.randn(1, self.num_domain_tokens, self.clip_model.config.hidden_size)
        )

        # Image encoder for time series
        self.image_encoder = ImageEncoder()

        # CNN for time series feature extraction
        self.time_series_cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def compute_semantic_preserved_feature(self, z_s: torch.Tensor, p_ds: torch.Tensor,
                                           p: torch.Tensor) -> torch.Tensor:
        """Compute semantic-preserved feature"""
        semantic_diff = (p_ds - p) / torch.norm(p_ds - p, p=2)
        return z_s + semantic_diff

    def get_prompt_features(self, class_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get general and domain-specific prompt features"""
        # General prompt
        general_prompt = f"a {class_label}"
        inputs = self.tokenizer(general_prompt, return_tensors="pt", padding=True).to(self.device)
        p = self.clip_model.text_model(**inputs).pooler_output

        # Domain-specific prompt
        class_tokens = self.tokenizer(class_label, return_tensors="pt", padding=True).to(self.device)
        class_features = self.clip_model.text_model(**class_tokens).last_hidden_state
        domain_features = torch.cat([self.domain_tokens.expand(1, -1, -1), class_features], dim=1)
        p_ds = self.clip_model.text_model.pooler(domain_features)

        return p, p_ds

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference on input time series data"""
        # 1. Extract time series features for MTA
        time_features = self.time_series_cnn(x)

        # 2. Encode time series to images
        imgs = self.image_encoder.encode(x.cpu().numpy())

        # 3. Get visual features through modified CLIP encoder with MTA
        image_features = self.clip_model.vision_model(imgs.to(self.device), time_features)

        # 4. Get text features and compute semantic-preserved features for each class
        class_features = []
        for class_name in self.class_names:
            p, p_ds = self.get_prompt_features(class_name)
            semantic_preserved = self.compute_semantic_preserved_feature(image_features, p_ds, p)
            class_features.append(semantic_preserved)

        class_features = torch.stack(class_features)

        # 5. Compute similarity and apply temperature scaling
        logits = (image_features @ class_features.T) / self.temperature

        return logits