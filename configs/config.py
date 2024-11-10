from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class ModelConfig:
    """Configuration for TimeSeriesUDA model with HAR dataset settings"""

    # Model architecture parameters
    num_classes: int = 6  # HAR classes: walking, walking_upstairs, walking_downstairs, sitting, standing, lying
    clip_model: str = "openai/clip-vit-base-patch32"  # CLIP model identifier
    feature_dim: int = 768  # CLIP's hidden size
    reduction_ratio: int = 8  # MTA reduction ratio
    temperature: float = 0.07  # Temperature for contrastive learning

    # Time series parameters
    sequence_length: int = 128  # HAR sequence length
    num_channels: int = 9  # HAR channels: total_acc_xyz, body_acc_xyz, body_gyro_xyz

    # Image encoding parameters
    image_size: int = 224  # Size for RP, MTF, GAF encodings
    mtf_bins: int = 10  # Number of bins for Markov Transition Field

    # Prompt learning parameters
    num_domain_tokens: int = 10  # Number of learnable domain tokens

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4

    # Loss weights
    lambda_s_cl: float = 1.0  # Source contrastive loss weight
    lambda_t_cl: float = 1.0  # Target contrastive loss weight
    lambda_ce: float = 1.0  # Cross entropy loss weight
    lambda_pda: float = 1.0  # Prototypical domain alignment loss weight

    # Data augmentation parameters
    jitter_scale: float = 0.1
    permutation_segments: int = 5
    crop_ratio: float = 0.9

    # Class names for HAR dataset
    class_names: List[str] = None

    def __post_init__(self):
        """Initialize default class names if not provided"""
        if self.class_names is None:
            self.class_names = [
                'walking',
                'walking_upstairs',
                'walking_downstairs',
                'sitting',
                'standing',
                'lying'
            ]

    @property
    def model_kwargs(self) -> dict:
        """Get kwargs for model initialization"""
        return {
            'num_classes': self.num_classes,
            'clip_model': self.clip_model,
            'temperature': self.temperature,
            'reduction_ratio': self.reduction_ratio
        }

    @property
    def image_encoder_kwargs(self) -> dict:
        """Get kwargs for image encoder initialization"""
        return {
            'image_size': self.image_size,
            'mtf_bins': self.mtf_bins
        }

    def validate(self):
        """Validate configuration parameters"""
        assert self.num_classes == len(self.class_names), \
            f"Number of classes ({self.num_classes}) must match number of class names ({len(self.class_names)})"

        assert self.sequence_length > 0, "Sequence length must be positive"
        assert self.num_channels == 9, "HAR dataset requires 9 channels"
        assert self.image_size > 0, "Image size must be positive"
        assert 0 < self.temperature <= 1, "Temperature must be in (0, 1]"
        assert self.reduction_ratio > 0, "Reduction ratio must be positive"

        assert 0 <= self.jitter_scale <= 1, "Jitter scale must be in [0, 1]"
        assert 0 < self.crop_ratio <= 1, "Crop ratio must be in (0, 1]"
        assert self.permutation_segments > 0, "Number of permutation segments must be positive"

        assert all(w >= 0 for w in [self.lambda_s_cl, self.lambda_t_cl, self.lambda_ce, self.lambda_pda]), \
            "Loss weights must be non-negative"


def get_har_config() -> ModelConfig:
    """Get default configuration for HAR dataset"""
    return ModelConfig()


def get_config_from_dict(config_dict: dict) -> ModelConfig:
    """Create config from dictionary"""
    return ModelConfig(**config_dict)