# Image-based Intrusion Detection System using CNN

This repository contains an implementation of a Convolutional Neural Network (CNN) for intrusion detection using image-based representation of network traffic data from the NSL-KDD dataset.

## 🎯 Overview

The project converts network traffic features into grayscale images and applies deep learning techniques for binary classification of network traffic as either normal or attack patterns.

## 🚀 Features

- **CNN Architecture**: Custom CNN model with batch normalization and dropout
- **Image Processing**: Converts network features to 7x7 grayscale images
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Performance Monitoring**: GPU usage monitoring and training time tracking
- **Comprehensive Evaluation**: Confusion matrix, ROC curve, and classification metrics
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Prevents overfitting with validation monitoring

## 📋 Requirements

```bash
tensorflow>=2.8.0
tensorflow-addons
wandb
scikit-learn
seaborn
matplotlib
pandas
numpy
nvidia-ml-py3
tqdm
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/image-based-ids.git
cd image-based-ids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

## 📁 Project Structure

Organize project folder as follows:
```
project/
├── model_experiments.py                    # CNN training script
├── data_preprocessing_onehot_encode.py    # One-hot preprocessing
├── data_preprocessing_label_encode.py     # Label preprocessing
├── image_generator.py         # Image Generation
├── README.md                 
├── preprocessed_csv_files     # Preprocessed data
└── img_samples/               # Image dataset samples
```

## 📁 Dataset Structure

Organize your dataset as follows:
```
project/
└── img_samples/
    └── label_encoding_img/
        ├── train/
        │   ├── attack/
        │   │   ├── attack1.png
        │   │   └── ...
        │   └── normal/
        │       ├── normal1.png
        │       └── ...
        └── test/
            ├── attack/
            │   ├── attack1.png
            │   └── ...
            └── normal/
                ├── normal1.png
                └── ...
```

## ⚙️ Configuration

Modify the `CONFIG` dictionary in the main script to adjust parameters:

```python
CONFIG = {
    'learning_rate': 0.01,      # Initial learning rate
    'epochs': 100,              # Maximum training epochs
    'batch_size': 128,          # Batch size for training
    'img_height': 6,            # Image height in pixels
    'img_width': 6,             # Image width in pixels
    'experiment_count': 1,      # Number of experiments to run
    'validation_split': 0.2,    # Validation split ratio
    'patience': 10,             # Early stopping patience
    'project_name': 'nsl_kdd', # Wandb project name
    'entity': 'your_entity'     # Your wandb entity name
}
```

## 📊 Model Architecture

The CNN model consists of the following layers (based on Table 6 parameters):

- **Input Layer**: 6×6×1 grayscale images (6, 6, 1 output shape)
- **Conv2D_1**: 32 filters producing 6×6×32 output with ReLU activation (320 weights)
- **MaxPool2D_1**: Max pooling layer reducing to 3×3×32 output
- **Conv2D_2**: 64 filters producing 3×3×64 output with ReLU activation (18,496 weights)
- **MaxPool2D_2**: Max pooling layer reducing to 2×2×64 output
- **Flatten**: Flattening layer converting to 256-dimensional vector
- **Dense1**: Fully connected layer with 128 units and ReLU activation (32,896 weights)
- **Dense2 (Output)**: Single neuron with sigmoid activation for binary classification (129 weights)

Total Parameters: 51,841 weights across all trainable layers
The model uses:
- Batch Normalization: Applied after convolutional and dense layers
- Dropout: Applied for regularization to prevent overfitting
- Binary Classification: Sigmoid activation in output layer for normal/attack classification

## 📈 Monitoring and Visualization

The script automatically generates:

- **Training History**: Accuracy and loss plots
- **Confusion Matrix**: Visual representation of classification results
- **ROC Curve**: Performance evaluation with AUC score
- **GPU Usage**: Memory utilization monitoring
- **Wandb Dashboard**: Real-time experiment tracking

All plots are saved in the `plots/` directory.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NSL-KDD dataset creators
- TensorFlow and Keras teams
- Weights & Biases for experiment tracking

## 👥 Authors

| Name | Affiliation | Email | Google Scholar |
|------|-------------|--------|---------------|
| **Doo-Seop Choi** | Department of Computer Science, Hanyang University, Seoul, Republic of Korea | dslab0915@hanyang.ac.kr | [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=YOUR_ID_1) |
| **Taeguen Kim** | Department of AI Cyber Security, Korea University, Sejong, Republic of Korea | taeguen_kim@korea.ac.kr | [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=YOUR_ID_2) |
| **BooJoong Kang** | School of Electronics and Computer Science, University of Southampton, Southampton, United Kingdom | b.kang@southampton.ac.uk | [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=YOUR_ID_3) |
| **Eul Gyu Im** ⭐ | Department of Computer Science, Hanyang University, Seoul, Republic of Korea | imeg@hanyang.ac.kr | [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=YOUR_ID_4) |

⭐ *Corresponding Author*

---

## 📖 Citation

If you use this repository or find this work helpful in your research, please cite our paper:

### BibTeX
```bibtex
@article{choi2024image,
  title={Image-based Malicious Network Traffic Detection Framework: Data-centric approach},
  author={Choi, Doo-Seop and Kim, Taeguen and Kang, BooJoong and Im, Eul Gyu},
  journal={Journal Name},
  volume={XX},
  number={X},
  pages={XXX--XXX},
  year={2025},
  publisher={Publisher Name},
  doi={10.XXXX/XXXXXXX}
}
```

### APA Style
Choi, D., Kim, T., Kang, B., & Im, E.G. (2025). Image-based Malicious Network Traffic Detection Framework: Data-centric approach. *Journal Name*, *XX*(X), XXX-XXX. https://doi.org/10.XXXX/XXXXXXX

### IEEE Style
D.-S. Choi, T. Kim, B. Kang, and E.G. Im, "Image-based Malicious Network Traffic Detection Framework: Data-centric approach," *Journal Name*, vol. XX, no. X, pp. XXX-XXX, 2024, doi: 10.XXXX/XXXXXXX.

---

**Note**: 
- The citation information will be updated with the correct journal details, DOI, and page numbers once the paper is officially published.
- Please note that the code may not work properly without proper configuration. Before using the code provided in this repository, you must modify the file paths, wandb entity information, and other settings to match your environment.