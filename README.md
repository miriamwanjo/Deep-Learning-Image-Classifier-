# Deep Learning - Image Classifier Project

> This project was part of Udacity coursework on deep learning methods, specifically developing an image classifier that would recognize different species of flowers.

### Project Outline
1. Load and preprocess the image dataset 
> > Using torchvison to load the data split into 3 parts: training, validation and testing
> > Apply transformation using random scaleing, cropping and flipping
2. Build and train the image classifier on the dataset
> > 1. Load a pre-trained network from torchvision.models
> > 2. Build and train a new feed-forward classifier using the features from teh pre-trained model
> > 3. Train the clasifier layers using backpropagation
> > 4. Track the loss and accuracy on the validation set

3. Use the trained classifier to predict image content

### Libraries/packages Used
1. torch - nn, optim, nn.functional
2. torchvision - datasets, transforms, models
3. matplotlib
4. numpy
5. time
6. json
7. PIL - image

### Files

1. Jupyter notebook 
2. HTML file of the project
3. predict.py - python script to use the trained model for prediction
4. train.py - python script to train a new network

### Licenses/Acknowledgements

Major thanks to Udacity courses and student mentors for help with the project. Feel free to use the code as you wish. 
