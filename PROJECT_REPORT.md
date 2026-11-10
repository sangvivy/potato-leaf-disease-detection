# Potato Leaf Disease Detection Using Deep Learning

## Project Technical Report

**Application Area:** Agriculture  
**Project Type:** Deep Learning and Image Classification  
**Models:** Baseline CNN and MobileNetV2  
**Frameworks:** TensorFlow, Keras and Streamlit

---

## 1. Abstract

This project investigates the application of deep learning to potato leaf disease classification. It uses a selected subset of the PlantVillage colour-image dataset containing three classes: Potato Early Blight, Potato Late Blight, and Potato Healthy.

A total of 2,152 images were explored, preprocessed, and divided into training, validation, and testing sets. Two models were developed and evaluated: a baseline Convolutional Neural Network (CNN) and MobileNetV2 using transfer learning.

The baseline CNN achieved a test accuracy of 82.35%, while MobileNetV2 achieved 96.90%. MobileNetV2 also recorded a lower test loss and higher F1-scores.

An interactive Streamlit dashboard was developed to present the project and allow users to upload potato leaf images for classification.

The results demonstrate the potential of transfer learning for this image-classification task. However, further testing using images collected under real agricultural conditions is necessary to establish how well the models generalise beyond the selected dataset.

## 2. Introduction

Agriculture is an important area where artificial intelligence and data science can support decision-making. Plant diseases can affect crop health and productivity, making the identification of visible symptoms an important agricultural challenge.

Deep learning, particularly Convolutional Neural Networks, provides techniques for analysing images and identifying visual patterns.

This project applies deep learning to potato leaf images to classify them into three categories: early blight, late blight, and healthy leaves.

The project compares a CNN developed from scratch with MobileNetV2, a pretrained neural network adapted through transfer learning.

The project is an experimental image-classification prototype and is not intended to replace professional agricultural diagnosis.

## 3. Problem Statement

Identifying visible symptoms on plant leaves can be challenging, particularly when different conditions have similar appearances.

This project investigates whether deep-learning models can learn visual patterns from labelled potato leaf images and automatically classify them into the selected categories.

**Research question:**

How effectively can a baseline CNN and a transfer-learning model based on MobileNetV2 classify potato leaf images into early blight, late blight, and healthy categories?

## 4. Project Objectives

The project aimed to:

1. Obtain and explore a labelled potato leaf image dataset.
2. Prepare the image data for deep-learning experiments.
3. Develop and train a baseline CNN.
4. Implement MobileNetV2 using transfer learning.
5. Evaluate both models using classification metrics.
6. Compare the models using a held-out test dataset.
7. Develop an interactive Streamlit dashboard for image classification.

## 5. Dataset Collection and Exploration

### 5.1 Dataset Source

The project uses the colour-image portion of the PlantVillage dataset.

**Dataset repository:**

https://github.com/spMohanty/PlantVillage-Dataset

The selected subset contains three potato leaf categories.

Original image dimensions: **256 × 256 pixels**

### 5.2 Dataset Distribution

| Class | Number of Images | Percentage |
|---|---:|---:|
| Potato Early Blight | 1,000 | 46.47% |
| Potato Late Blight | 1,000 | 46.47% |
| Potato Healthy | 152 | 7.06% |
| **Total** | **2,152** | **100%** |

### 5.3 Dataset Exploration

The first notebook, `01_dataset_exploration.ipynb`, was used to inspect the dataset before model development.

The exploration involved:

- Counting images in each class.
- Visualising class distribution.
- Displaying sample images.
- Checking image dimensions.
- Checking for corrupted image files.

No corrupted images were found during the completed check.

The exploration outputs included:

- `dataset/class_distribution.csv`
- `dataset/image_dimensions.csv`

### 5.4 Dataset Imbalance

The dataset was imbalanced because the healthy class contained significantly fewer images than the early-blight and late-blight classes.

This was considered during model training through the use of class weights.

## 6. Data Preparation and Preprocessing

### 6.1 Dataset Splitting

The second notebook, `02_data_preprocessing.ipynb`, was used to divide the dataset into training, validation, and testing subsets.

A stratified split was applied using `random_state=42` to preserve the approximate class proportions.

| Dataset | Percentage | Number of Images |
|---|---:|---:|
| Training | 70% | 1,506 |
| Validation | 15% | 323 |
| Testing | 15% | 323 |
| **Total** | **100%** | **2,152** |

The training set was used to train the models, the validation set was used to monitor performance during training, and the test set was reserved for final evaluation.

The split records were saved as:

- `dataset/train_split.csv`
- `dataset/validation_split.csv`
- `dataset/test_split.csv`

### 6.2 Class Weight Calculation

Class weights were calculated to address the unequal class distribution.

| Class | Weight |
|---|---:|
| Early Blight | 0.717 |
| Late Blight | 0.717 |
| Healthy | 4.736 |

The healthy class received a higher weight because it contained fewer images.

Class weighting gives errors involving the underrepresented class greater influence during training. It does not create additional images.

### 6.3 Image Preprocessing

The third notebook, `03_image_preprocessing.ipynb`, prepared the images for model training.

The preprocessing steps included:

- Resizing images to 224 × 224 pixels.
- Loading images in batches.
- Normalising pixel values for the baseline CNN.
- Preparing image augmentation operations.
- Optimising dataset loading.

Image augmentation included:

- Random horizontal flipping.
- Random rotation.
- Random zoom.

These transformations expose the model to variations of training images and can help it learn more general visual features.

For MobileNetV2, the model-specific `preprocess_input` function was used because the pretrained architecture expects a particular input preprocessing convention.

## 7. Model Development

Two models were developed to compare a CNN trained for this project with a pretrained model adapted through transfer learning.

### 7.1 Baseline Convolutional Neural Network

The baseline CNN was developed in `04_cnn_baseline.ipynb` using TensorFlow and Keras.

Its architecture included:

1. Input layer accepting images of shape 224 × 224 × 3.
2. Data augmentation layers.
3. Convolutional layer with 32 filters.
4. Max-pooling layer.
5. Convolutional layer with 64 filters.
6. Max-pooling layer.
7. Convolutional layer with 128 filters.
8. Max-pooling layer.
9. Global average pooling.
10. Dense layer with 64 units and ReLU activation.
11. Dropout layer with a rate of 0.3.
12. Output layer with three softmax units.

The model contained **101,699 trainable parameters**.

#### Training Configuration

| Parameter | Configuration |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Evaluation metric | Accuracy |
| Class weights | Applied |
| Early stopping | Applied |
| Model checkpointing | Applied |

Early stopping monitored validation performance and restored the best weights when training stopped.

The trained model was saved as:

`models/cnn_baseline.keras`

### 7.2 MobileNetV2 Using Transfer Learning

MobileNetV2 was implemented in `05_mobilenetv2.ipynb`.

MobileNetV2 is a pretrained convolutional neural network. Transfer learning uses features learned by an existing model on a large dataset and adapts them to a new classification task.

This approach is useful when the available task-specific dataset is relatively small.

The implementation involved:

1. Loading MobileNetV2 with ImageNet pretrained weights.
2. Setting `include_top=False` to remove the original classification layer.
3. Freezing the pretrained base model.
4. Adding a global average pooling layer.
5. Adding dropout with a rate of 0.3.
6. Adding a new dense output layer with three softmax units.
7. Applying MobileNetV2-specific image preprocessing.

The model was trained using Adam with a learning rate of 0.001 and sparse categorical cross-entropy loss.

Class weights, early stopping, and model checkpointing were also applied.

The pretrained base model was frozen during this experiment, meaning the added classification head was trained rather than fine-tuning the entire MobileNetV2 network.

The trained model was saved as:

`models/mobilenetv2.keras`

## 8. Model Evaluation

Both models were evaluated using the same held-out test dataset containing 323 images.

The evaluation included:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrices.
- Test loss.

### 8.1 Evaluation Metrics

**Accuracy:** Measures the proportion of all test images correctly classified.

**Precision:** Measures how many images predicted as a particular class actually belong to that class.

**Recall:** Measures how many actual images belonging to a class were correctly identified.

**F1-score:** Combines precision and recall into one measure.

**Macro F1-score:** Calculates the F1-score for each class and averages them equally.

**Weighted F1-score:** Calculates the average F1-score while accounting for the number of examples in each class.

**Confusion matrix:** Shows the actual and predicted classes, helping identify which categories the model confuses.

These metrics provide a more complete understanding of model performance than accuracy alone.

### 8.2 Overall Model Comparison

| Evaluation Metric | Baseline CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 82.35% | 96.90% |
| Test Loss | 0.368500 | 0.089542 |
| Macro F1-score | 0.806400 | 0.952047 |
| Weighted F1-score | 0.818500 | 0.968937 |

MobileNetV2 achieved approximately **14.55 percentage points higher test accuracy** than the baseline CNN.

It also achieved a lower test loss and higher macro and weighted F1-scores.

Based on these results, MobileNetV2 performed better overall on the selected dataset and test split.

### 8.3 MobileNetV2 Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Early Blight | 0.9932 | 0.9667 | 0.9797 | 150 |
| Late Blight | 0.9487 | 0.9867 | 0.9673 | 150 |
| Healthy | 0.9524 | 0.8696 | 0.9091 | 23 |

The healthy class achieved a recall of approximately 86.96%, meaning that 20 of the 23 actual healthy test images were correctly identified.

The small number of healthy test images should be considered when interpreting this result.

### 8.4 Results Interpretation

MobileNetV2 performed better than the baseline CNN across the reported evaluation metrics.

One possible explanation is that MobileNetV2 started with visual features learned from ImageNet, while the baseline CNN had to learn its visual representations from the project training dataset.

The results suggest that transfer learning was beneficial for this experiment.

However, these findings apply to the selected dataset, model configurations, and held-out test split. They do not guarantee the same performance on other datasets or real farm images.

## 9. Streamlit Dashboard Development

An interactive dashboard was developed using Streamlit in:

`dashboard/app.py`

The dashboard provides a graphical interface for presenting the project and interacting with the trained model.

It contains four main sections:

### 9.1 About Project

Introduces the project, its purpose, objectives, and agricultural application.

### 9.2 Dashboard

Displays project information and selected model-performance summaries.

### 9.3 Disease Detection

Provides an image-upload interface where users can upload a potato leaf image and view a classification prediction.

### 9.4 Model Performance

Presents model evaluation results and comparisons between the baseline CNN and MobileNetV2.

The application uses:

- TensorFlow for model loading and prediction.
- NumPy for numerical operations.
- Pandas for data handling.
- Matplotlib for visualisation.
- Streamlit for the interactive interface.

The dashboard layout was adjusted to improve navigation, colour contrast, and readability.

## 10. Project Structure

```text
Potato_Leaf_Disease_Project/
│
├── dashboard/
│   └── app.py
│
├── dataset/
│   ├── processed/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── class_distribution.csv
│   ├── image_dimensions.csv
│   ├── train_split.csv
│   ├── validation_split.csv
│   ├── test_split.csv
│   ├── mobilenetv2_results.csv
│   ├── mobilenetv2_classification_report.csv
│   ├── model_comparison.csv
│   └── model_accuracy_comparison.png
│
├── models/
│   ├── cnn_baseline.keras
│   └── mobilenetv2.keras
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_image_preprocessing.ipynb
│   ├── 04_cnn_baseline.ipynb
│   └── 05_mobilenetv2.ipynb
│
├── plantvillage/
├── .gitignore
├── README.md
└── requirements.txt
```

The exact repository contents may vary depending on which large dataset and model files are stored in GitHub. Raw and processed image directories may be excluded from version control to keep the repository manageable.

## 11. Installation and Execution

### 11.1 Requirements

The project was developed using Python 3.12.7 in a virtual environment on Windows.

The main dependencies include:

- TensorFlow
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter

### 11.2 Clone the Repository

```powershell
git clone https://github.com/sangvivy/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection
```

### 11.3 Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 11.4 Install Dependencies

```powershell
pip install -r requirements.txt
```

### 11.5 Prepare the Required Files

Ensure the trained models are available at the paths configured in `dashboard/app.py`:

- `models/cnn_baseline.keras`
- `models/mobilenetv2.keras`

The dataset is required to reproduce the training notebooks. Obtain it from the PlantVillage repository and arrange it according to the paths used in the notebooks.

If model or dataset files are not included in GitHub, they must be obtained separately and placed in the expected locations.

### 11.6 Run the Dashboard

From the project root directory, execute:

```powershell
streamlit run dashboard/app.py
```

Open the local address displayed in the terminal. The default address is usually:

`http://localhost:8501`

## 12. Limitations

The project has several limitations:

1. **Class imbalance:** The healthy class contains considerably fewer images than the disease classes.
2. **Controlled image conditions:** PlantVillage images were collected under controlled conditions. Real farm images may have different lighting, backgrounds, camera quality, and leaf orientations.
3. **Limited class coverage:** The model only classifies the three selected categories and cannot be assumed to recognise every potato disease.
4. **Generalisation:** High test accuracy does not guarantee similar performance on independently collected farm images.
5. **Evaluation sample size:** The test set contains only 23 healthy images, limiting the certainty of conclusions about that class.
6. **Computational constraints:** Model development was performed using available local computing resources.

The application should therefore be treated as an experimental image-classification prototype rather than a field-validated agricultural diagnostic system.

## 13. Future Improvements

Potential future improvements include:

- Collecting additional potato leaf images from real agricultural environments.
- Increasing the number and variety of healthy leaf images.
- Testing the model on an independent dataset captured under field conditions.
- Fine-tuning selected MobileNetV2 layers and comparing the results.
- Evaluating additional pretrained architectures.
- Investigating prediction errors using explainability techniques.
- Improving the dashboard with additional prediction information and clearer guidance about the limitations of automated classification.

Future improvements should be evaluated using data that was not used during training or model selection.

## 14. Conclusion

This project developed a deep-learning prototype for classifying potato leaf images into early blight, late blight, and healthy categories.

The work involved dataset exploration, preprocessing, class-weight calculation, model development, evaluation, model comparison, and implementation of an interactive Streamlit dashboard.

The baseline CNN achieved 82.35% test accuracy, while MobileNetV2 achieved 96.90% on the same held-out test set. MobileNetV2 also achieved better macro and weighted F1-scores and a lower test loss.

These results indicate that transfer learning was more effective for the selected dataset and experimental setup.

The project demonstrates the potential of deep learning in agricultural image classification. Further testing using independently collected field images is necessary before drawing conclusions about performance in real farming environments.

## References

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060. https://arxiv.org/abs/1511.08060
2. PlantVillage-Dataset. GitHub repository. https://github.com/spMohanty/PlantVillage-Dataset
3. TensorFlow. TensorFlow documentation. https://www.tensorflow.org/
4. Keras. Keras Applications: MobileNetV2. https://keras.io/api/applications/mobilenet/
5. Streamlit. Streamlit documentation. https://docs.streamlit.io/

---

**GitHub Repository:** https://github.com/sangvivy/potato-leaf-disease-detection

**GitHub Account:** sangvivy