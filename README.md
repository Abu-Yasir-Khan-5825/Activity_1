# Image Recognition Project

## Description
This project explores image recognition and classification using deep learning techniques with TensorFlow and Keras. It demonstrates building and training Convolutional Neural Networks (CNNs) on different datasets, including MNIST, CIFAR-10, and a custom dataset of cats and dogs. The project also covers concepts like data preprocessing, data augmentation, transfer learning with MobileNetV2, model evaluation, and making predictions on new data.

## Installation
To run this project, you need to install the following libraries:
- TensorFlow
- Keras (included with TensorFlow)
- Matplotlib
- NumPy
- Scikit-learn
- Seaborn
- Kaggle (for dataset download)

You can install these using pip:
```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn kaggle
```

You will also need to download the necessary datasets. The notebook uses `tf.keras.datasets.mnist.load_data()` and `tf.keras.datasets.cifar10.load_data()`. The cats and dogs dataset is downloaded using the Kaggle API. Ensure you have your Kaggle API key set up correctly to download the cats and dogs dataset.

## Usage
The project is presented as a Jupyter Notebook. You can execute the cells sequentially to:
1.  Load and preprocess the MNIST and CIFAR-10 datasets.
2.  Explore the datasets.
3.  Build and train a CNN on the MNIST dataset.
4.  Evaluate the MNIST model's performance.
5.  Implement data augmentation for the CIFAR-10 dataset.
6.  Build and train a deeper CNN with Batch Normalization and Dropout on the augmented CIFAR-10 dataset.
7.  Evaluate the CIFAR-10 model using advanced metrics like classification report and confusion matrix.
8.  Download and prepare the cats and dogs dataset.
9.  Implement transfer learning using a pre-trained MobileNetV2 model on the cats and dogs dataset.
10. Fine-tune the MobileNetV2 model.
11. Save and load the trained model.
12. Evaluate the cats and dogs model using the ROC curve and AUC.
13. Make predictions on new image data.
14. Visualize and present the results.

## Results
The project demonstrates the process of training image classification models and evaluating their performance. Key results include:
-   **MNIST Dataset:** A simple CNN achieves high accuracy (approximately 98-99%) on the test set.
-   **CIFAR-10 Dataset:** A deeper CNN with data augmentation and batch normalization shows improved performance compared to a basic model, with an accuracy of approximately 70% on the test set. A classification report and confusion matrix provide detailed insights into the model's performance across different classes.
-   **Cats vs. Dogs Dataset:** Transfer learning with MobileNetV2, followed by fine-tuning, results in an accuracy of approximately 84% on the validation set. The ROC curve and AUC are used to evaluate the model's ability to distinguish between the two classes.

Visualizations are included in the notebook to show:
-   Examples of images from the datasets.
-   Training and validation accuracy/loss curves for the MNIST model.
-   Confusion matrix for the CIFAR-10 model.
-   ROC curve for the Cats vs. Dogs model.
-   A bar plot comparing the accuracies achieved on the three datasets.
