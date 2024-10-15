# Segmentation-on-Kvasir-SEG
Implementing a U-Net architecture for image segmentation using TensorFlow and Keras

# Introduction
Image segmentation is a fundamental task in computer vision, aiming to partition images into meaningful regions. U-Net is a convolutional neural network architecture widely used for semantic segmentation tasks due to its ability to capture context information eﬀectively. In this report, we present the implementation of a U-Net model for image segmentation using TensorFlow and Keras. The objective is to achieve high segmentation accuracy, targeting a metric of 0.9748.

# Dataset Details
- **Name:** Kvasir-SEG
- **Size:** 46.2 MB
- **Content:** 1000 polyp images with corresponding ground truth masks
- **Resolution:** Varies from 332x487 to 1920x1072 pixels
- **Format:** JPEG images and JSON for bounding box coordinates
- **Availability:** Open-access for research and edu
- [**Link**](https://datasets.simula.no/kvasir-seg/)

# Dataset Preparation
We start by preparing the dataset for training and validation:
- We load the images and corresponding masks from the dataset directory.
- The images are resized to a fixed size of 256x256 pixels.
- The pixel values are normalized and stored in NumPy arrays.
- We split the dataset into training and validation sets, with 900 samples for training and the remaining for validatio

# Model Architecture
The U-Net architecture consists of a contracting path (encoder) followed by an expanding path (decoder). We implement the following architecture:
- The contracting path comprises convolutional and max-pooling layers to extract features and reduce spatial dimensions.
- The expanding path involves transposed convolutional layers to upsample the feature maps and restore spatial information.
- Skip connections are incorporated to concatenate feature maps from the contracting path to the corresponding layers in the expanding path, aiding in precise localization.

# Training
We compile the model using the Adam optimizer and binary cross-entropy loss function. The training procedure involves:
- Training the model on the training dataset for 30 epochs with a batch size of 8.
- Utilizing 1% of the training data for validation to monitor model performance.
- Employing TensorBoard for visualization and monitoring of training metrics.
- Employing callbacks such as early stopping to prevent overfitting.

# Evaluation
We evaluate the trained model using various metrics:
- We compute accuracy, loss, and other relevant metrics on both training and validation sets.
- We visualize the training history to understand the model's learning progress and identify the best epochs.

We calculate the confusion matrix to analyze the model's performance in binary classification.A confusion matrix shows the counts of true positive (TP), false positive (FP), true negative (TN), and false negative (FN) predictions.
- TP: Number of correctly predicted positive instances.
- FP: Number of incorrectly predicted positive instances.
- TN: Number of correctly predicted negative instances.
- FN: Number of incorrectly predicted negative instances.
- This plot helps in understanding the performance of the model in terms of classification accuracy.

We plot the ROC curve and calculate the AUC to assess the model's ability to discriminate between classes.The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
- TPR (also known as sensitivity) measures the proportion of true positive predictions among all actual positive instances.
- FPR measures the proportion of false positive predictions among all actual negative instances.
- The ROC curve helps in understanding the trade-oﬀ between sensitivity and specificity (1 - FPR) for diﬀerent threshold values.

A the end we plot the IoU (Intersection over Union). IoU measures the overlap between the predicted segmentation mask and the ground truth mask. The IoU curve plots the IoU score for diﬀerent threshold values used to binarize the predicted masks.It helps in understanding the model's ability to accurately segment objects of interest across diﬀerent levels of thresholding.
