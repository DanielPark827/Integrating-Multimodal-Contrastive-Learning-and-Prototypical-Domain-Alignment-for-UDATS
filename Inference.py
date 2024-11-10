import torch
import numpy as np
from typing import List, Optional, Union
from configs.config import ModelConfig
from models.modules import TimeSeriesUDA


def load_model(config: ModelConfig, checkpoint_path: str, device: str = 'cuda') -> TimeSeriesUDA:
    """Load trained TimeSeriesUDA model"""
    model = TimeSeriesUDA(
        num_classes=config.num_classes,
        clip_model=config.clip_model,
        temperature=config.temperature,
        reduction_ratio=config.reduction_ratio,
        device=device
    ).to(device)

    # Set class names
    model.class_names = config.class_names

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def preprocess_data(data: np.ndarray, config: ModelConfig) -> torch.Tensor:
    """
    Preprocess sensor data for inference
    Expected shape: (batch_size, channels=9, sequence_length=128)
    """
    # Shape validation
    if len(data.shape) == 2:
        data = data.reshape(1, *data.shape)
    elif len(data.shape) != 3:
        raise ValueError(f"Expected 2D or 3D input, got shape {data.shape}")

    if data.shape[1] != config.num_channels:
        raise ValueError(f"Expected {config.num_channels} channels, got {data.shape[1]}")

    if data.shape[2] != config.sequence_length:
        raise ValueError(
            f"Expected sequence length {config.sequence_length}, got {data.shape[2]}"
        )

    # Normalize
    data = data.astype(np.float32)
    means = data.mean(axis=2, keepdims=True)
    stds = data.std(axis=2, keepdims=True)
    normalized_data = (data - means) / (stds + 1e-8)

    return torch.FloatTensor(normalized_data)


def inference(
        model: TimeSeriesUDA,
        data: Union[np.ndarray, torch.Tensor],
        config: ModelConfig,
        device: str = 'cuda',
        return_probabilities: bool = False,
        batch_size: Optional[int] = 32
) -> Union[List[str], np.ndarray]:
    """
    Perform inference following the paper's logic:
    1. Encode time series to images (RP, MTF, GAF)
    2. Extract visual features using modified CLIP encoder with MTA
    3. Get prompt features and compute semantic-preserved features
    4. Calculate similarity scores
    """
    if isinstance(data, np.ndarray):
        data = preprocess_data(data, config)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)

            # Get logits using model's inference method
            # This internally handles:
            # - Time series to image encoding
            # - Visual feature extraction with MTA
            # - Prompt feature processing
            # - Semantic-preserved feature computation
            # - Similarity calculation
            logits = model.inference(batch)

            if return_probabilities:
                probs = torch.softmax(logits / config.temperature, dim=1)
                predictions.append(probs.cpu().numpy())
            else:
                preds = torch.argmax(logits, dim=1)
                predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    if not return_probabilities:
        return [model.class_names[pred] for pred in predictions]
    return predictions


def main():
    # Example usage
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(
        config=config,
        checkpoint_path='path/to/checkpoint.pth',
        device=device
    )

    # Example data
    sample_data = np.random.randn(10, config.num_channels, config.sequence_length)

    # Get predictions
    predictions = inference(
        model=model,
        data=sample_data,
        config=config,
        device=device,
        return_probabilities=True
    )

    # Print results
    if predictions.ndim == 2:
        for i, probs in enumerate(predictions):
            print(f"\nSample {i + 1} probabilities:")
            for activity, prob in zip(model.class_names, probs):
                print(f"{activity}: {prob:.4f}")
    else:
        print("\nPredicted activities:", predictions)


if __name__ == "__main__":
    main()