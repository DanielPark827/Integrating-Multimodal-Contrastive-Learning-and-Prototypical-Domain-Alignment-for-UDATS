# Integrating Multimodal Contrastive Learning with Prototypical Domain Alignment for UDATS

Official implementation of the paper ["Integrating Multimodal Contrastive Learning with Prototypical Domain Alignment for Unsupervised Domain Adaptation of Time Series"](https://www.sciencedirect.com/science/article/abs/pii/S0952197624013630).

## Quick Start

### Installation
```bash
git clone https://github.com/username/MultimodalUDATS.git
cd MultimodalUDATS
pip install -r requirements.txt
```

### Download Dataset
This implementation uses the HAR (Human Activity Recognition) dataset. Please download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and place it in the `data` directory.

```bash
mkdir data
cd data
# Download and extract HAR dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip 'UCI HAR Dataset.zip'
```

### Inference
To run inference on your data:

```bash
python inference.py \
    --checkpoint path/to/checkpoint.pth \
    --input_data path/to/your/data.csv \
    --output_path results.csv \
    --batch_size 32
```

## Citation
If you find this code useful for your research, please cite our paper:
```
@article{park2024integrating,
  title={Integrating multimodal contrastive learning with prototypical domain alignment for unsupervised domain adaptation of time series},
  author={Park, Seo-Hyeong and Syazwany, Nur Suriza and Nam, Ju-Hyeon and Lee, Sang-Chul},
  journal={Engineering Applications of Artificial Intelligence},
  volume={137},
  pages={109205},
  year={2024},
  publisher={Elsevier}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.