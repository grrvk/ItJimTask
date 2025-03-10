from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.models import FeedForwardNN, CNN
import torch.nn as nn
from torch.optim import SGD, Adam

# When running with command-line uncomment to see confusion matrices
# import matplotlib
# matplotlib.use("TkAgg")


def full_report(y, y_pred):
    """
    Calculates accuracy, macro-averaged f1 score
    Prints classification report and visualizes confusion matrix
    Parameters:
            y (numpy.ndarray): Ground truth labels
            y_pred (numpy.ndarray): Predicted labels
    """
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    print(f'Accuracy {accuracy}, macro f1 {macro_f1}')

    clr = classification_report(y, y_pred, labels=np.unique(y))
    print(clr)

    cf_matrix = confusion_matrix(y, y_pred)
    hm = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.show()


class MnistClassifierInterface(ABC):
    """
    Abstract interface class that defines methods for subclasses
    """
    def __init__(self, model):
        """
        Initializes the model parameter for subclasses
        """
        self.model = model

    @abstractmethod
    def train(self, train_dataloader):
        """
        Trains the model on train data
        Parameters:
            train_dataloader(torch.utils.data.Dataloader): Dataloader for train data
        """
        pass

    @abstractmethod
    def test(self, test_dataloader):
        """
        Test model performance on test data
        Parameters:
            test_dataloader(torch.utils.data.Dataloader): Dataloader for test data
        """
        pass

    @abstractmethod
    def predict(self, dataloader):
        """
        Uses trained model for prediction on data from user
        Parameters:
            dataloader(torch.utils.data.Dataloader): Dataloader for user data
        """
        return


class RFCClassifier(MnistClassifierInterface):
    """
    Classifier that uses Random Forest classification algorithm
    """
    def __init__(self):
        """
        Initializes class parameter 'model' as Random Forest
        """
        model = RandomForestClassifier(n_estimators=100)
        super().__init__(model)

    @staticmethod
    def prepare_data(dataloader):
        """
        Converts data from dataloader to suitable for sklearn model.fit format
        Parameters:
            dataloader(torch.utils.data.Dataloader): Dataloader for used data
        Return:
            X(numpy.ndarray): Data features
            y(numpy.ndarray): Data labels
        """
        X, y = [], []
        for images, labels in dataloader:
            X.extend(images.view(images.size(0), -1).numpy())
            y.extend(labels.numpy())
        return X, y

    def train(self, train_dataloader):
        print('Training RFC...')
        X_train, y_train = self.prepare_data(train_dataloader)
        self.model.fit(X_train, y_train)
        print('Training finished')

    def test(self, dataloader):
        print('Performing prediction on test set...')
        X_test, y_test = self.prepare_data(dataloader)
        y_pred = self.model.predict(X_test)

        full_report(np.array(y_test), y_pred)
        return y_pred, y_test

    def predict(self, data):
        data = data.reshape(-1, 784)
        outputs = self.model.predict(data)
        return outputs


class FFNClassifier(MnistClassifierInterface):
    """
    Classifier that uses Feed-Forward neural network
    """
    def __init__(self):
        """
        Initializes class parameter 'model' as Feed-Forward Network
        Sets optimizer and criterion for training
        """
        model = FeedForwardNN()
        super().__init__(model)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader, n_epochs=15):
        print('Training FFN...')
        self.model.train()
        for i in range(n_epochs):
            epoch_loss = 0.0
            for images, labels in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            print(f'Epoch {i + 1}: loss = {epoch_loss / len(train_dataloader)}')
        print('Training finished')

    def test(self, test_dataloader):
        print('Performing prediction on test set...')
        y_test, y_pred = [], []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.numpy())
                y_test.extend(labels.numpy())

        y_test, y_pred = np.array(y_test), np.array(y_pred)
        full_report(y_test, y_pred)
        return y_pred, y_test

    def predict(self, data):
        with torch.no_grad():
            outputs = self.model(torch.tensor(data))
            _, predicted = torch.max(outputs, 1)
        return predicted


class CNNClassifier(MnistClassifierInterface):
    """
    Initializes class parameter 'model' as Convolutional Neural Network
    Sets optimizer and criterion for training
    """
    def __init__(self):
        model = CNN()
        super().__init__(model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader, n_epochs=10):
        print('Training CNN...')
        self.model.train()
        for i in range(n_epochs):
            epoch_loss = 0.0
            for images, labels in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            print(f'Epoch {i + 1}: loss = {epoch_loss / len(train_dataloader)}')
        print('Training finished')

    def test(self, test_dataloader):
        print('Performing prediction on test set...')
        y_test, y_pred = [], []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.numpy())
                y_test.extend(labels.numpy())

        y_test, y_pred = np.array(y_test), np.array(y_pred)
        full_report(y_test, y_pred)
        return y_pred, y_test

    def predict(self, data):
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted


class MnistClassifier:
    """
    Wrapper class for Classifiers to provide one logic for using different models
    """
    def __init__(self, algorithm):
        """
        Sets classifier according to algorithm name
        :Parameters:
            algorithm(str): algorithm name ('rf', 'nn', 'cnn')
        """
        if algorithm == 'rf':
            self.classifier = RFCClassifier()
        elif algorithm == 'nn':
            self.classifier = FFNClassifier()
        elif algorithm == 'cnn':
            self.classifier = CNNClassifier()
        else:
            raise ValueError("Unsupported algorithm")

    def train(self, train_loader):
        self.classifier.train(train_loader)

    def test(self, test_loader):
        y_pred = self.classifier.test(test_loader)
        return y_pred

    def predict(self, prepared_data):
        predictions = self.classifier.predict(prepared_data)
        return predictions
