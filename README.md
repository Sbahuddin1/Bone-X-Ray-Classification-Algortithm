# Bone X-ray Classification

This repository contains an implementation of a Bone X-ray Classification model using deep learning. The model is trained to classify X-ray images of bones based on provided training data.

## Dataset

- **Train Data:** Located in the `train_data/` directory, consisting of labeled images used for training the model.
- **Test Data:** Located in the `test_data/` directory, consisting of images used for evaluating the model's performance.

## Dependencies

Ensure you have the following dependencies installed before running the notebook:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn opencv-python
```

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bone-xray.git
   cd bone-xray
   ```
2. Install required dependencies as mentioned above.
3. Place training and test data in the respective directories (`train_data/` and `test_data/`).
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook bone-xray.ipynb
   ```
5. Run the notebook to train and evaluate the model.


### Sample X-ray Image
![sample](https://github.com/Sbahuddin1/Bone-X-Ray-Classification-Algortithm/blob/master/files/image_png.png)   

## Model Implementation

- The model is implemented in `bone-xray.ipynb` using TensorFlow/Keras.
- It is based on a Convolutional Neural Network (CNN) architecture designed for image classification.
- The CNN consists of multiple convolutional layers followed by max-pooling layers to extract important features from the X-ray images.
- Fully connected layers process the extracted features to classify the images into relevant categories.
- The model is trained using a suitable optimizer and loss function, with validation results monitored to prevent overfitting.
- Data preprocessing, augmentation, model training, and evaluation steps are included.
- Performance metrics and visualizations are provided.

### Preprocessing of Images
![output image](https://github.com/Sbahuddin1/Bone-X-Ray-Classification-Algortithm/blob/master/files/preprocess.png)

## Results

- After training, the model performance is evaluated on the test dataset.
- Accuracy, loss curves, and confusion matrices are plotted for better understanding.

### Classified Images
![output image](https://github.com/Sbahuddin1/Bone-X-Ray-Classification-Algortithm/blob/master/files/output.png)

## Contributions

Feel free to contribute to this project by submitting a pull request or reporting issues.

## License

This project is licensed under the MIT License.
