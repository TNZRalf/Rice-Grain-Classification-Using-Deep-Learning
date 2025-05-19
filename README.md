# Rice-Grain-Classification-Using-Deep-Learning

Overview
This project develops a deep learning model to classify rice grain images into five varieties using transfer learning with the ResNet-18 architecture. Implemented in Python and MATLAB, the model achieves 100% accuracy on a 1,100-image subset and 99.8% accuracy on the full 75,000-image dataset, demonstrating robust performance and efficient training.
Technologies Used

Programming Languages: Python, MATLAB
Libraries: PyTorch, MATLAB Deep Learning Toolbox
Tools: Kaggle (dataset hosting), Git, Jupyter Notebook, MATLAB Editor

Methodology

Data Preprocessing: Normalized images and applied data augmentation (e.g., rotation, flipping) to improve model generalization.
Model Architecture: Employed ResNet-18 with transfer learning, fine-tuning the fully connected layers for rice grain classification.
Training: Conducted two-stage training:
Stage 1: Trained on a 1,100-image subset with a batch size of 32 and Adam optimizer.
Stage 2: Scaled to the full 75,000-image dataset for enhanced performance.


Evaluation: Measured classification accuracy and training time, optimizing for efficiency.

Results

Accuracy (Subset): 100% on 1,100 images
Accuracy (Full Dataset): 99.8% on 75,000 images
Training Time: 384.33 seconds for the subset, scalable to larger datasets
Key Achievement: High accuracy with optimized computational efficiency

Installation

Clone the repository:
git clone https://github.com/TNZRalf/Rice-Grain-Classification-Using-Deep-Learning.git


Install Python 3.8 or later and required libraries:
pip install torch torchvision matplotlib numpy


(Optional) Install MATLAB R2023a or later for MATLAB-based scripts.

Download the rice grain dataset from Kaggle or Google Drive and extract it to the data/ folder.


Usage
Python Implementation

Navigate to the project directory:
cd Rice-Grain-Classification


Open rice_classification.py or rice_classification.ipynb in your preferred editor (e.g., VS Code, Jupyter).

Update the dataset path in the script to point to data/.

Run the script:
python rice_classification.py


Outputs include accuracy metrics, confusion matrix, and sample classification visualizations.


MATLAB Implementation (Optional)

Open MATLAB and set the working directory to the project folder.

Open rice_classification.m.

Ensure the dataset is in the data/ folder.

Run the script:
run rice_classification.m


Outputs include accuracy metrics and visualization plots.


Project Structure

rice_classification.py: Main Python script for training and evaluating the model.
rice_classification.ipynb: Jupyter Notebook with a detailed workflow.
rice_classification.m: MATLAB script for alternative implementation.
data/: Placeholder directory for the rice grain dataset (not included in the repository).
README.md: Project documentation.

Dataset
The dataset contains 75,000 rice grain images across five varieties. Due to its size, it is not included in the repository. Download it from:

Kaggle (recommended)
Google Drive (alternative, if uploaded)

Place the dataset in the data/ folder and update the script paths accordingly.
Future Improvements

Explore alternative architectures like EfficientNet or Vision Transformers for improved accuracy.
Develop a web application for real-time rice grain classification.
Optimize training for lower-resource environments using model pruning or quantization.

Contact
For questions or collaboration, contact Zakaria Tanani at zakaria.tanani12@gmail.com.
License
This project is licensed under the MIT License. See the LICENSE file for details.

