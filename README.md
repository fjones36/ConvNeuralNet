# ConvNeuralNet
Convolutional Neural Net - Kaggle Facial Keypoint Detection

Single Layer

Convolutional Network

Data Augmentation
*If your network doesnt overfit, make the network bigger
*If it is overfitting, adding more training data can help
We add more data by transforming and/or adding noise to existing training
samples.

Use batch iterators to perform transformations in place so we dont need to store new samples in memory. It can be done while the network is training on the previous data, so it is basically free.

Using augmentation makes it take longer to learn because it generalizes better and doesnt overfit as much
