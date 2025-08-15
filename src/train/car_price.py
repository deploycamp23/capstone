from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CarPriceApp:
    """
    App to train and evaluate a linear regression model for car-price prediction
    based on seven car features: length, width, height, engine size, horsepower, mileage in city, mileage in highway
    """

    def __init__(self, dataset, batch_size=16, seed=100, model_path=None):
        """
        args:
        dataset (tuple): (x, y) where x and y are arrays
        """

        self.history_dic = None  # will store training history
        if model_path is not None:
            self.model_path = model_path  # where to store and/or load model's parameter
        else:
            self.model_path = f"./model_s{seed}.pth"

        self.input, self.output = dataset
        self.batch_size = batch_size
        self.seed = seed  # for reproducibility
        torch.manual_seed(self.seed)

        # split into training and test sets
        self.train_size = 0.7
        self.input_train, self.input_test, self.target_train, self.target_test = (
            train_test_split(
                self.input,
                self.output,
                train_size=self.train_size,
                random_state=self.seed,
                shuffle=True,
            )
        )

        # from inputs to features via feature map
        self.features_train = feature_map(self.input_train)
        self.features_test = feature_map(self.input_test)
        self.features_size = self.features_train.shape[-1]
        self.output_size = self.output.shape[-1]

        # store parameters for feature normalization later
        self.min_max_scaler = {
            "min": self.features_train.min(axis=0),
            "max": self.features_train.max(axis=0),
        }

        # train and test sets as members of Pytorch Dataset class
        self.trainset = CarPriceDataset(
            self.features_train,
            self.target_train,
            features_transform=self.feature_transform,
            target_transform=self.target_transform,
        )

        self.testset = CarPriceDataset(
            self.features_test,
            self.target_test,
            features_transform=self.feature_transform,
            target_transform=self.target_transform,
        )

        # training set a dataloader for training in shuffled batches
        self.train_loader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        # model
        self.model = nn.Linear(self.features_size + 1, self.output_size, bias=False)
        self.best_state = (
            self.model.state_dict()
        )  # will store optimal parameters (state)

        # loss function
        self.loss = nn.MSELoss

        # optimizer
        self.optimizer = optim.SGD

    def feature_transform(self, x):
        """
        args:
        x (array)
        return:
        x_new (tensor)
        """
        # normalize using minmax scaling for better performance during training
        min_ = self.min_max_scaler["min"]
        max_ = self.min_max_scaler["max"]
        x_new = (x - min_) / (max_ - min_)
        # convert into tensor
        x_new = torch.tensor(x_new, dtype=torch.float32)
        # add intercept term
        x_new = torch.cat((torch.ones(x_new.shape[0], 1), x_new), dim=1)
        return x_new

    def target_transform(self, y):
        """
        args:
        y (array)
        return:
        y_new (tensor)
        """
        # convert into tensor
        return torch.tensor(y, dtype=torch.float32)

    def train_model(self, lr=1e-2, epochs=2000, display_freq=10, save_model=True):

        # initialize loss and optimizer
        loss = self.loss()
        optimizer = self.optimizer(self.model.parameters(), lr=lr)

        # use for evaluating the model during training
        x_test, y_test = self.testset[:]

        # store training history
        history_dic = {
            "train": torch.zeros(epochs),
            "test": torch.zeros(epochs),
            "weights": torch.zeros([epochs, 1, self.features_size + 1]),
        }

        # training loop
        for epoch in range(epochs):
            self.model.train()  # activates training mode

            for x, y in self.train_loader:
                optimizer.zero_grad()  # restarts gradient for each batch so not to accumulate
                y_pred = self.model(x)  # predicted value
                loss_train = loss(y_pred, y)
                loss_train.backward()  # computes gradient
                optimizer.step()  # updates parameters

            # evaluate trained model on test data using R2 score
            self.model.eval()  # activates evaluation model
            with torch.no_grad():
                y_test_pred = self.model(x_test).numpy()
                r2 = r2_score(y_test_np, y_test_pred)

            # store loss for train and test data and parameters
            history_dic["train"][epoch] = loss_train.item()
            history_dic["test"][epoch] = r2
            history_dic["weights"][epoch] = next(self.model.parameters()).data

            # display every display_freq epochs
            if epoch % display_freq == 0:
                print(
                    f"epoch {epoch}: Training loss {loss_train.item()}, R2 score on test data {round(r2, 5)}"
                )

        # best epoch chosen from maximum R2 score
        best_epoch = torch.argmax(history_dic["test"])

        print(f"\nBest epoch = {best_epoch}")
        print(f'Training loss = {history_dic["train"][best_epoch]}')
        print(f'Highest R2 score on test data = {history_dic["test"][best_epoch]}')

        self.history_dic = history_dic
        optimal_weights = history_dic["weights"][best_epoch]

        if save_model:
            # save the optimal model, not the last one, and the transformation parameters (min, max)
            self.best_state = OrderedDict({"weight": optimal_weights})
            torch.save(
                {
                    "state": self.best_state,
                    "transformation_parameters": self.min_max_scaler,
                },
                self.model_path,
            )

    def load_model(self, path):
        # load
        loaded_model = torch.load(path)
        state = loaded_model["state"]
        min_max = loaded_model["transformation_parameters"]
        # update
        self.model.load_state_dict(state)
        self.min_max_scaler = min_max

    def predict_price(self, x):
        """
        predict car price from car's features
        args:
        x (array): 7-dim array (length, width, height, engine size, horsepower, mileage in city, mileage in highway)
        return:
        y_pred (tensor): price
        """
        phi = feature_map(x)
        phi_norm = self.feature_transform(phi)
        y_pred = self.model(phi_norm)
        return y_pred


def feature_map(x):
    """
    args:
    x (array): 7-dim array
    return:
    phi (array): (x0, x1, x2, x3, x4, x0^2, x1^2, x2^2, x3^2, x4^2, 1/x5, 1/x6, 1/x5^2, 1/x6^2)
    """
    x_pos = x[:, :5]
    x_neg = x[:, 5:]
    phi = np.concatenate([x_pos, x_pos**2, 1 / x_neg, 1 / x_neg**2], axis=1)
    return phi


class CarPriceDataset(Dataset):
    def __init__(
        self, features, target, features_transform=None, target_transform=None
    ):
        self.features = features
        self.target = target
        self.features_transform = features_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]

        # catch cases with no slice index
        if type(idx) == int:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)

        # transform input
        if self.features_transform:
            x = self.features_transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
