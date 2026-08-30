"""Part C."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import Perceptron
import torch
import torch.nn as nn
import itertools

def load_data():
    df = pd.read_csv('feat_data.csv')

    y = df['dendrite_type']                    # label
    X = df.drop(columns=['dendrite_type'])     # never used as a feature

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_log_r(X_train,X_test, y_train, y_test):
    log_reg = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))
    # standardize the data to 0 mean, 1 var. done on test so there is no leakage.
    # the standard scaler is fit on the training data and then done on the test data.
    log_reg.fit(X_train, y_train) # leanr the mean,std of the training data, z=xw+b-> sig(z)
    # predict on the test data
    y_pred = log_reg.predict(X_test) 
    print('accuracy :', accuracy_score(y_test, y_pred))
    print('precision:', precision_score(y_test, y_pred, pos_label='spiny'))
    print('recall   :', recall_score(y_test, y_pred, pos_label='spiny'))
    cm = confusion_matrix(y_test, y_pred, labels=['aspiny', 'spiny'])
    ConfusionMatrixDisplay(cm, display_labels=['aspiny', 'spiny']).plot()

def com_log_per(X_train, X_test, y_train, y_test,):
    perc = make_pipeline(StandardScaler(), Perceptron(max_iter=1000, random_state=42)) #same as log_reg, just without sigmoid, put sign(z)
    perc.fit(X_train, y_train)
    log_reg = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))
    log_reg.fit(X_train, y_train)
    acc ={}
    for name, model in [('logreg', log_reg), ('perceptron', perc)]:
        for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            yp = model.predict(X)
            print(name, split, accuracy_score(y, yp),
                precision_score(y, yp, pos_label='spiny'),
                recall_score(y, yp, pos_label='spiny'),
                f1_score(y, yp, pos_label='spiny'))
            acc[name] = accuracy_score(y, yp),precision_score(y, yp, pos_label='spiny'),recall_score(y, yp, pos_label='spiny'),f1_score(y, yp, pos_label='spiny')
    return acc

def make_mlp():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ), nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )

def comp_2(X_train, X_test, y_train, y_test):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.map({'aspiny': 0, 'spiny': 1}).values, dtype=torch.long)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.map({'aspiny': 0, 'spiny': 1}).values, dtype=torch.long)
    acc = {}
    for name, model in [('mlp', make_mlp()[0]), ('mlp_dropout', make_mlp()[1])]:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
            precision = precision_score(y_test_t, predicted, pos_label=1)
            recall = recall_score(y_test_t, predicted, pos_label=1)
            f1 = f1_score(y_test_t, predicted, pos_label=1)
            print(f"{name} test accuracy: {accuracy:.4f}")
            acc[name] = accuracy, precision, recall, f1
            print(f"{name} test accuracy: {accuracy:.4f}")
    return acc

def comp_mlpVSlog_per(X_train, X_test, y_train, y_test):
    results = {}
    results.update(com_log_per(X_train, X_test, y_train, y_test))  # logreg + perceptron
    results.update(comp_2(X_train, X_test, y_train, y_test))       # MLP architecture(s)

    for name, (acc, prec, rec, f1) in results.items():
        print(f"{name}: accuracy={acc:.4f}, precision={prec:.4f}, "
              f"recall={rec:.4f}, f1={f1:.4f}")
    return results

def comp_mlp(X_train, X_test, y_train, y_test):
    lrs=[0.01, 0.001, 0.0001]
    hidden_layers = [2,3]
    functions =[nn.ReLU(), nn.GELU(), nn.Sigmoid(), nn.Tanh()]
    hidden_dim = [32, 64]
    epochs = [50, 100]
    results = {}
    for lr, hl, func, hdim, epoch in itertools.product(lrs, hidden_layers, functions, hidden_dim, epochs):
        layers = []
        input_dim = 16
        for _ in range(hl):
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(func)
            input_dim = hdim
        layers.append(nn.Linear(input_dim, 2))
        model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.map({'aspiny': 0, 'spiny': 1}).values, dtype=torch.long)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.map({'aspiny': 0, 'spiny': 1}).values, dtype=torch.long)

        for epoch in range(epoch):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
            precision = precision_score(y_test_t, predicted, pos_label=1)
            recall = recall_score(y_test_t, predicted, pos_label=1)
            f1 = f1_score(y_test_t, predicted, pos_label=1)

            config_name = f"lr={lr}_hl={hl}_func={func.__class__.__name__}_hdim={hdim}_epoch={epoch}"
            results[config_name] = (accuracy, precision, recall, f1)
    return results
def train_mlp_with_GD(X_train, X_test, y_train, y_test, learning_rate=0.01, epochs=100):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_valid_t = torch.FloatTensor(X_valid)
    y_valid_t = torch.LongTensor(y_valid)

    torch.manual_seed(42)
    n_features = X_train.shape[1]
    mlp = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        loss = criterion(mlp(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        mlp.eval()
        with torch.no_grad():
            val_losses.append(criterion(mlp(X_valid_t), y_valid_t).item())
    return mlp, train_losses, val_losses


def main():
    pass


if __name__ == "__main__":
    main()
