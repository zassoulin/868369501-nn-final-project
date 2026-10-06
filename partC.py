"""Part C."""
import random

import numpy as np
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
import matplotlib.pyplot as plt

# seed
SEED = 42


def set_seed(seed=SEED):
    """sets the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    df = pd.read_csv('feat_data.csv')

    y = df['dendrite_type']                    # label
    X = df.drop(columns=['dendrite_type'])     # never used as a feature
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test
LABELS = {'aspiny': 0, 'spiny': 1}

def to_tensors(X_train, X_test, y_train, y_test):
    """We standard scale the data so its approprate for this task (cross entophy)
    """
    scaler = StandardScaler().fit(X_train)
    return (torch.tensor(scaler.transform(X_train), dtype=torch.float32),
            torch.tensor(y_train.map(LABELS).values, dtype=torch.long),
            torch.tensor(scaler.transform(X_test), dtype=torch.float32),
            torch.tensor(y_test.map(LABELS).values, dtype=torch.long))


def make_log_reg():
    """Logistic regression on standardized features.

    The StandardScaler lives inside the pipeline so its mean/std are learned from the
    training fold only and merely applied to the test fold -- fitting the scaler on the
    full dataset would leak test statistics into training.
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=SEED))





def train_log_r(X_train, X_test, y_train, y_test):
    log_reg = make_log_reg()
    log_reg.fit(X_train, y_train)  # learns mean/std + weights: z = xw + b -> sigmoid(z)
    y_pred = log_reg.predict(X_test)
    metrics = report_errors(y_test, y_pred, 'logistic regression')
    return log_reg, metrics
def report_errors(y_true, y_pred, title):
    """Confusion matrix + the metrics of Q1a, with `spiny` as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=['aspiny', 'spiny'])
    (tn, fp), (fn, tp) = cm

    metrics = (accuracy_score(y_true, y_pred),
               precision_score(y_true, y_pred, pos_label='spiny'),
               recall_score(y_true, y_pred, pos_label='spiny'),
               f1_score(y_true, y_pred, pos_label='spiny'))
    print(f'{title}: accuracy={metrics[0]:.4f}, precision={metrics[1]:.4f}, '
          f'recall={metrics[2]:.4f}, f1={metrics[3]:.4f}')
    # Break the errors out by direction: precision and recall summarize them, but the
    # counts say which mistake the model actually makes.
    print(f'  true negatives  (aspiny -> aspiny): {tn}')
    print(f'  false positives (aspiny -> spiny) : {fp}')
    print(f'  false negatives (spiny  -> aspiny): {fn}')
    print(f'  true positives  (spiny  -> spiny) : {tp}')

    disp = ConfusionMatrixDisplay(cm, display_labels=['aspiny', 'spiny'])
    disp.plot(colorbar=False)
    disp.ax_.set_title(f'{title}\nconfusion matrix (test set, counts)')
    plt.tight_layout()
    plt.show()
    return metrics
METRIC_NAMES = ('accuracy', 'precision', 'recall', 'f1')


def compare_models(results, names, split='test'):
    """Q2a: which model generalized better, judged on the held-out split.

    Generalization is measured on data neither model was fitted to, so the comparison is
    made on `test`; ties are reported as such rather than broken arbitrarily.
    """
    header = f"{'metric':10s}" + ''.join(f'{n:>13s}' for n in names) + f"{'better':>14s}"
    print(header)
    print('-' * len(header))
    wins = {name: 0 for name in names}
    for i, metric in enumerate(METRIC_NAMES):
        values = [results[(name, split)][i] for name in names]
        best = max(values)
        leaders = [n for n, v in zip(names, values) if v == best]
        verdict = 'tie' if len(leaders) > 1 else leaders[0]
        if len(leaders) == 1:
            wins[leaders[0]] += 1
        print(f'{metric:10s}' + ''.join(f'{v:13.4f}' for v in values) + f'{verdict:>14s}')
    ranked = sorted(wins.items(), key=lambda kv: kv[1], reverse=True)
    print(f'-> {ranked[0][0]} generalizes better '
          f'({ranked[0][1]} of {len(METRIC_NAMES)} metrics on {split})')


def report_overfitting(results, names):
    """Q2b: train-test gap for each model in `results`, keyed by (name, split).

    A large positive gap (train well above test) is the signature of overfitting: the
    model has memorized the training set instead of learning a rule that transfers. A gap
    near zero -- or negative, where test happens to score higher -- means it has not.
    """
    header = f"{'model':12s} {'metric':10s} {'train':>8s} {'test':>8s} {'gap':>8s}"
    print(header)
    print('-' * len(header))
    for name in names:
        train, test = results[(name, 'train')], results[(name, 'test')]
        for metric, tr, te in zip(METRIC_NAMES, train, test):
            print(f'{name:12s} {metric:10s} {tr:8.4f} {te:8.4f} {tr - te:+8.4f}')


def com_log_per(X_train, X_test, y_train, y_test, log_reg=None):
    perc = make_pipeline(StandardScaler(), Perceptron(max_iter=1000, random_state=SEED)) #same as log_reg, just without sigmoid, put sign(z)
    perc.fit(X_train, y_train)
    if log_reg is None:  # reuse the model fitted in Q1 so both sections report one model
        log_reg = make_log_reg().fit(X_train, y_train)
    results = {}
    for name, model in [('logreg', log_reg), ('perceptron', perc)]:
        for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            yp = model.predict(X)
            # Key on (model, split): keying on `name` alone let the test row overwrite the
            # train row, discarding the train metrics that Q2b needs for the overfitting gap.
            results[(name, split)] = (accuracy_score(y, yp),
                                      precision_score(y, yp, pos_label='spiny'),
                                      recall_score(y, yp, pos_label='spiny'),
                                      f1_score(y, yp, pos_label='spiny'))
    compare_models(results, ['logreg', 'perceptron'])
    print()
    report_overfitting(results, ['logreg', 'perceptron'])
    return results

def make_mlp(n_features):
    """The two architectures compared in Q3a.

    Deliberately different on three axes at once (depth, activation, regularization) so
    the comparison is between two genuinely distinct designs:
      mlp         -- 1 hidden layer, ReLU, no regularization  (the smaller baseline)
      mlp_dropout -- 2 hidden layers, GELU, dropout 0.2       (deeper + regularized)
    """
    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ), nn.Sequential(
        nn.Linear(n_features, 32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )


# Training hyper-parameters for Q3
MLP_LR = 0.001
MLP_EPOCHS = 100
MLP_BATCH = 32


def describe_model(name, model, optimizer='Adam', lr=MLP_LR, epochs=MLP_EPOCHS,
                   batch_size=MLP_BATCH):
    """Q3a: report the six required properties of an architecture.

    Parameter count is the trainable-tensor total (weights + biases); the activation list
    is read off the module itself so it cannot drift from the model actually trained.
    """
    linears = [m for m in model if isinstance(m, nn.Linear)]
    widths = ' -> '.join([str(linears[0].in_features)]
                         + [str(layer.out_features) for layer in linears])
    activations = [type(m).__name__ for m in model if not isinstance(m, nn.Linear)]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'{name}')
    print(f'layers: {widths}')
    print(f'activations: {", ".join(activations) or "none"}')
    print(f'parameters: {n_params}')
    print(f'optimizer: {optimizer}')
    print(f'learning rate :{lr}')
    print(f'epochs: {epochs}')
    print(f'batch size: {batch_size}')


def train_torch(model, X_t, y_t, lr=MLP_LR, epochs=MLP_EPOCHS, batch_size=MLP_BATCH):
    """Mini-batch Adam training. Returns the per-epoch mean training loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_train = X_t.shape[0]

    losses = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n_train)
    return losses


def evaluate_torch(model, X_t, y_t):
    """Accuracy/precision/recall/F1 with spiny as the positive class."""
    model.eval()
    with torch.no_grad():
        predicted = model(X_t).argmax(dim=1)
    return (accuracy_score(y_t, predicted),
            precision_score(y_t, predicted, pos_label=1),
            recall_score(y_t, predicted, pos_label=1),
            f1_score(y_t, predicted, pos_label=1))


def comp_2(X_train, X_test, y_train, y_test):
    X_train_t, y_train_t, X_test_t, y_test_t = to_tensors(X_train, X_test, y_train, y_test)
    n_features = X_train_t.shape[1]
    results = {}
    set_seed()  # same init draw for both architectures, and stable across reruns
    models = make_mlp(n_features)
    for i, name in enumerate(['mlp', 'mlp_dropout']):
        model = models[i]
        describe_model(name, model)
        train_torch(model, X_train_t, y_train_t)
        # Train metrics as well as test: Q3b compares against the baselines on test, and
        # Q4c needs the train-test gap for the same models.
        results[(name, 'train')] = evaluate_torch(model, X_train_t, y_train_t)
        results[(name, 'test')] = evaluate_torch(model, X_test_t, y_test_t)
        print(f'  test accuracy : {results[(name, "test")][0]:.4f}\n')
    return results


def comp_mlpVSlog_per(X_train, X_test, y_train, y_test, log_reg=None):
    results = {}
    results.update(com_log_per(X_train, X_test, y_train, y_test, log_reg))  # logreg + perceptron
    results.update(comp_2(X_train, X_test, y_train, y_test))       # MLP architecture(s)

    # Q3b: the MLPs against both baselines on identical test data, spiny = positive class.
    names = ['logreg', 'perceptron', 'mlp', 'mlp_dropout']
    compare_models(results, names)
    print()
    report_overfitting(results, names)
    return results

def build_mlp(n_features, n_layers=2, hidden_dim=32, activation=nn.ReLU, dropout=0.0):
    """A configurable MLP, so a sweep can vary one structural knob at a time."""
    layers = []
    in_dim = n_features
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


# The configuration every sweep holds fixed while it varies its own hyper-parameter.
BASE_CONFIG = dict(n_layers=2, hidden_dim=32, activation=nn.ReLU, dropout=0.0,
                   lr=MLP_LR, epochs=MLP_EPOCHS, batch_size=MLP_BATCH)

SWEEPS = {
    'learning rate': ('lr', [0.0001, 0.001, 0.01, 0.1]),
    'hidden units per layer': ('hidden_dim', [8, 16, 32, 64, 128]),
    'hidden layers': ('n_layers', [1, 2, 3, 4]),
    'dropout': ('dropout', [0.0, 0.2, 0.5]),
    'batch size': ('batch_size', [8, 16, 32, 64, 128]),
}


def sensitivity_analysis(X_train, y_train, sweeps=SWEEPS):
    """Q3c: vary one hyper-parameter at a time, holding the rest at BASE_CONFIG.

    Two decisions worth stating. (1) One knob at a time -- a full cross-product would
    confound the effects and no single hyper-parameter's influence could be read off it.
    (2) Scored on a 10% validation split carved out of the *training* set, never on test:
    picking hyper-parameters by test accuracy turns the reported test number into a
    best-of-N and it stops being an estimate of generalization.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    X_tr_t, y_tr_t, X_val_t, y_val_t = to_tensors(X_tr, X_val, y_tr, y_val)
    n_features = X_tr_t.shape[1]

    results, diagnostics = {}, {}
    for label, (param, values) in sweeps.items():
        for value in values:
            cfg = dict(BASE_CONFIG, **{param: value})
            set_seed()  # identical init across the sweep: differences are the knob, not luck
            model = build_mlp(n_features, cfg['n_layers'], cfg['hidden_dim'],
                              cfg['activation'], cfg['dropout'])
            losses = train_torch(model, X_tr_t, y_tr_t, lr=cfg['lr'], epochs=cfg['epochs'],
                                 batch_size=cfg['batch_size'])
            results[(label, value)] = evaluate_torch(model, X_val_t, y_val_t)
            # Evidence that the configurations really differ, and that the flat validation
            # curves are saturation rather than noise: parameter count, how far training
            # actually got, and *which* validation neurons each run gets wrong.
            model.eval()
            with torch.no_grad():
                wrong = (model(X_val_t).argmax(dim=1) != y_val_t).nonzero().flatten()
            diagnostics[(label, value)] = {
                'params': sum(p.numel() for p in model.parameters()),
                'final_train_loss': losses[-1],
                'train_accuracy': evaluate_torch(model, X_tr_t, y_tr_t)[0],
                'wrong': wrong.tolist(),
            }
    return results, diagnostics


def report_sensitivity(results, diagnostics=None, sweeps=SWEEPS):
    """Q3c: print the sweep as a table and plot validation accuracy against each knob.

    With `diagnostics` from sensitivity_analysis it also prints the evidence behind the
    reading of those curves: that the models genuinely differ, and that the flat
    validation accuracy is one stubborn neuron rather than random variation.
    """
    for label, (_, values) in sweeps.items():
        print(f'{label}')
        print(f"  {'value':>8s} {'accuracy':>9s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
        for value in values:
            acc, prec, rec, f1 = results[(label, value)]
            print(f'  {value:>8} {acc:9.4f} {prec:10.4f} {rec:8.4f} {f1:8.4f}')
        best = max(values, key=lambda v: results[(label, v)][0])
        print(f'  -> best {label}: {best} '
              f'(validation accuracy {results[(label, best)][0]:.4f})\n')

    if diagnostics is not None:
        print('Diagnostics: do the configurations actually differ?')
        print(f"  {'configuration':34s} {'params':>8s} {'train loss':>11s} "
              f"{'train acc':>10s} {'val errors':>11s}")
        for label, (_, values) in sweeps.items():
            for value in values:
                d = diagnostics[(label, value)]
                print(f'  {label + " = " + str(value):34s} {d["params"]:8d} '
                      f'{d["final_train_loss"]:11.5f} {d["train_accuracy"]:10.4f} '
                      f'{len(d["wrong"]):11d}')

        # Which validation neurons are responsible for the flat curves?
        tally = {}
        for d in diagnostics.values():
            for i in d['wrong']:
                tally[i] = tally.get(i, 0) + 1
        n_runs = len(diagnostics)
        print(f'\n  validation neurons misclassified, across all {n_runs} runs:')
        for i, count in sorted(tally.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            print(f'    validation index {i:4d}: wrong in {count:2d} / {n_runs} runs')
        print()

    # 2x3 grid rather than one row: five panels side by side are unreadable once the
    # figure is shrunk to page width in the report.
    n_cols = 3
    n_rows = -(-len(sweeps) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for ax in axes[len(sweeps):]:
        ax.set_visible(False)
    for ax, (label, (_, values)) in zip(axes, sweeps.items()):
        accuracies = [results[(label, v)][0] for v in values]
        ax.plot(values, accuracies, marker='o')
        ax.set_title(f'Sensitivity to {label}')
        # Learning rate, batch size and width are swept over powers of 10 / 2, so a log
        # axis spaces them evenly; the rest are small integers or fractions.
        log_x = label in ('learning rate', 'batch size', 'hidden units per layer')
        ax.set_xlabel(label + (' (log scale)' if log_x else ''))
        ax.set_ylabel('validation accuracy (fraction correct)')
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        if log_x:
            ax.set_xscale('log')
            ax.minorticks_off()
        # Tick only the values actually tested, so no axis shows e.g. "2.5 hidden layers".
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])
    fig.suptitle('Q3c: MLP hyper-parameter sensitivity '
                 '(one knob varied, all others fixed; 10% validation split)')
    plt.tight_layout()
    plt.show()


# Q4 training hyper-parameters. Plain SGD (not Adam) so the GD/SGD comparison in 4b is
# about the batch size alone, and a higher lr than Q3 because plain SGD converges slower.
GD_LR = 0.01
GD_EPOCHS = 300
SGD_BATCH = 10
INV_LABELS = {v: k for k, v in LABELS.items()}


def to_label_names(indices):
    """Torch class indices -> the original string labels, so report_errors can be reused."""
    return np.array([INV_LABELS[int(i)] for i in indices])


def train_with_curves(model, X_tr, y_tr, X_val, y_val, lr=GD_LR, epochs=GD_EPOCHS,
                      batch_size=None):
    """Train and record the per-epoch train and validation loss.

    batch_size=None is full-batch Gradient Descent: one parameter update per epoch over
    the whole training set. Any integer is mini-batch SGD with that batch size. The train
    loss is the batch-size-weighted mean over the epoch, so the two regimes are comparable
    on the same axes; validation loss is always a single full-set forward pass in eval
    mode (dropout off, no gradients).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_train = X_tr.shape[0]
    step = n_train if batch_size is None else batch_size

    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_train) if batch_size else torch.arange(n_train)
        epoch_loss = 0.0
        for begin in range(0, n_train, step):
            idx = perm[begin:begin + step]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(X_val), y_val).item())
    return train_losses, val_losses


def q4_tensors(X_train, y_train, X_test, y_test):
    """The data split shared by every Q4 run: 90/10 train/validation out of the training
    set, plus the test set.

    All three are standardized with ONE scaler fitted on the 90% training part only. This
    matters: the models train on the 90% part, so the test set has to be transformed with
    that same scaler. Fitting a separate scaler on the full training set would feed the
    models test features on a slightly different scale than the ones they trained on.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    scaler = StandardScaler().fit(X_tr)
    features = lambda X: torch.tensor(scaler.transform(X), dtype=torch.float32)
    labels = lambda y: torch.tensor(y.map(LABELS).values, dtype=torch.long)
    return (features(X_tr), labels(y_tr), features(X_val), labels(y_val),
            features(X_test), labels(y_test))


def train_mlp_with_GD(X_tr_t, y_tr_t, X_val_t, y_val_t, batch_size=None, dropout=0.0,
                      hidden_dim=32, lr=GD_LR, epochs=GD_EPOCHS):
    """Q4: one hidden layer trained by GD (batch_size=None) or SGD, on pre-scaled tensors.

    Takes tensors rather than frames so every Q4 run shares one split and one scaler.
    Returns the fitted model and the two loss curves.
    """
    set_seed()
    # With n_layers=1 the dropout sits between the hidden layer and the output layer,
    # i.e. immediately before the classification layer, which is what Q4d asks for.
    model = build_mlp(X_tr_t.shape[1], n_layers=1, hidden_dim=hidden_dim, dropout=dropout)
    curves = train_with_curves(model, X_tr_t, y_tr_t, X_val_t, y_val_t,
                               lr=lr, epochs=epochs, batch_size=batch_size)
    return model, curves


def describe_curve(name, train_losses, val_losses, threshold=0.1, tail=50):
    """Q4b/4c/4d: the numbers behind the loss curves, which a plot can only suggest.

    Prints where training got to, the lowest validation loss and the epoch it occurred
    (the point after which further training only overfits), whether the training loss is
    still falling at the end, and how many epochs it took to get below `threshold`.
    """
    best = min(range(len(val_losses)), key=lambda i: val_losses[i])
    below = next((i + 1 for i, v in enumerate(train_losses) if v < threshold), None)
    tail_change = train_losses[-1] - train_losses[-1 - tail]
    print(f'{name}:')
    print(f'train loss      {train_losses[0]:.4f} -> {train_losses[-1]:.4f}'
          f'last {tail} epochs: {tail_change:+.4f})')
    print(f'validation loss {val_losses[0]:.4f} -> {val_losses[-1]:.4f}'
          f'(final gap {val_losses[-1] - train_losses[-1]:+.4f})')
    print(f'lowest validation loss {val_losses[best]:.4f} at epoch {best + 1} '
          f'of {len(val_losses)}')
    print(f'train loss first below {threshold}: '
          f'{"epoch " + str(below) if below else "never reached"}')


def plot_loss_curves(runs, title):
    """Q4/4b: train vs validation cross-entropy against epoch, one panel per run."""
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.2), squeeze=False)
    y_max = max(max(max(tr), max(vl)) for _, tr, vl in runs) * 1.1
    for ax, (name, train_losses, val_losses) in zip(axes[0], runs):
        ax.plot(train_losses, label='train')
        ax.plot(val_losses, label='validation (10% of train)')
        ax.set_title(name)
        ax.set_xlabel('epoch (number of passes over the training set)')
        ax.set_ylabel('cross-entropy loss (nats)')
        ax.set_ylim(0, y_max)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def error_analysis(model, X_train, y_train, X_test, y_test, X_test_t, y_test_t,
                   n_show=5, n_separating=10):
    """Q4e: which test neurons the MLP gets wrong, and whether they resemble the other class.

    Two views of each mistake:
    1. The features on which it deviates most from its own class's training mean, in
       training standard deviations -- is this neuron atypical for its class?
    2. On the `n_separating` features that best separate the classes in the training set
       (largest |Cohen's d|), how many put it closer to the *other* class's mean? The same
       count over the correctly classified neurons is the baseline to compare against.
    """
    model.eval()
    with torch.no_grad():
        p_spiny = torch.softmax(model(X_test_t), dim=1)[:, 1]
    predicted = (p_spiny > 0.5).long()
    wrong = (predicted != y_test_t).nonzero().flatten().tolist()
    print(f'{len(wrong)} misclassified out of {len(y_test_t)}')

    means = {label: X_train[y_train == label].mean() for label in LABELS}
    variances = {label: X_train[y_train == label].var() for label in LABELS}
    stds = X_train.std()
    cohens_d = ((means['spiny'] - means['aspiny'])
                / np.sqrt((variances['spiny'] + variances['aspiny']) / 2)).abs()
    ranked_features = cohens_d.sort_values(ascending=False).index
    separating = ranked_features[:n_separating]
    # Rank of each feature by class separation, so "unusual on this feature" can be read
    # against "does this feature distinguish the classes at all".
    rank = {f: i + 1 for i, f in enumerate(ranked_features)}

    def closer_to_other(i):
        row, true_label = X_test.iloc[i], y_test.iloc[i]
        other = [label for label in LABELS if label != true_label][0]
        return sum(abs(row[f] - means[other][f]) < abs(row[f] - means[true_label][f])
                   for f in separating)

    for i in wrong[:n_show]:
        row = X_test.iloc[i]
        true_label = y_test.iloc[i]
        other = [label for label in LABELS if label != true_label][0]
        z_true = (row - means[true_label]) / stds
        z_other = (row - means[other]) / stds
        top = z_true.abs().sort_values(ascending=False).head(3)
        print(f'\n  test index {i}: true={true_label}, predicted={INV_LABELS[int(predicted[i])]}, '
              f'P(spiny)={p_spiny[i]:.3f}')
        print(f'    closer to {other} on {closer_to_other(i)}/{n_separating} '
              f'most class-separating features')
        for feature in top.index:
            closer = 'other' if abs(z_other[feature]) < abs(z_true[feature]) else 'own'
            print(f'    {feature:34s} value={row[feature]:10.3f} '
                  f'z vs {true_label:6s}={z_true[feature]:+6.2f} '
                  f'z vs {other:6s}={z_other[feature]:+6.2f}  -> closer to {closer}  '
                  f'(|d|={cohens_d[feature]:.2f}, separation rank '
                  f'{rank[feature]}/{len(ranked_features)})')

    right = [i for i in range(len(y_test_t)) if i not in wrong]
    counts = [closer_to_other(i) for i in right]
    print(f'\n  baseline, {len(right)} correctly classified neurons: closer to the other class '
          f'on a median of {np.median(counts):.0f}/{n_separating} '
          f'(mean {np.mean(counts):.2f}) most class-separating features')
    print(f'  most class-separating features: {", ".join(separating)}')
    return wrong


def run_q4(X_train, X_test, y_train, y_test):
    """Q4 a-e: GD, SGD(batch=10), overfitting, dropout, error analysis."""
    X_tr_t, y_tr_t, X_val_t, y_val_t, X_test_t, y_test_t = q4_tensors(
        X_train, y_train, X_test, y_test
    )

    # --- GD (full batch) and SGD (batch=10) on identical splits and initialization ---
    gd_model, (gd_train, gd_val) = train_mlp_with_GD(X_tr_t, y_tr_t, X_val_t, y_val_t)
    sgd_model, (sgd_train, sgd_val) = train_mlp_with_GD(X_tr_t, y_tr_t, X_val_t, y_val_t,
                                                        batch_size=SGD_BATCH)

    plot_loss_curves([('Gradient Descent (full batch)', gd_train, gd_val),
                      (f'Stochastic GD (batch={SGD_BATCH})', sgd_train, sgd_val)],
                     'Q4/4b: MLP training loss, GD vs SGD '
                     f'(1 hidden layer of 32, plain SGD optimizer, lr={GD_LR})')

    print('4a: error analysis of the GD-trained MLP on the test set')
    with torch.no_grad():
        gd_model.eval()
        gd_pred = gd_model(X_test_t).argmax(dim=1)
    report_errors(y_test, to_label_names(gd_pred), 'MLP trained with GD')

    print('\n4b: loss curves in numbers')
    # The gap is validation - train so the sign matches the metric tables: positive
    # always means "worse on held-out data", i.e. the overfitting direction.
    describe_curve('GD (full batch)', gd_train, gd_val)
    describe_curve(f'SGD (batch={SGD_BATCH})', sgd_train, sgd_val)

    # --- 4c: overfitting, judged on both the loss gap and the test metrics ---
    print('\n4c: train / validation / test metrics')
    results = {}
    for name, model in [('mlp_gd', gd_model), ('mlp_sgd', sgd_model)]:
        results[(name, 'train')] = evaluate_torch(model, X_tr_t, y_tr_t)
        results[(name, 'test')] = evaluate_torch(model, X_test_t, y_test_t)
    report_overfitting(results, ['mlp_gd', 'mlp_sgd'])

    # --- 4d: 50% dropout immediately before the classification layer ---
    print('\n4d: same network with dropout(0.5) before the output layer')
    drop_model, (drop_train, drop_val) = train_mlp_with_GD(
        X_tr_t, y_tr_t, X_val_t, y_val_t, batch_size=SGD_BATCH, dropout=0.5
    )
    plot_loss_curves([(f'SGD (batch={SGD_BATCH}) + dropout 0.5', drop_train, drop_val)],
                     'Q4d: effect of 50% dropout before the classification layer')
    results[('mlp_dropout50', 'train')] = evaluate_torch(drop_model, X_tr_t, y_tr_t)
    results[('mlp_dropout50', 'test')] = evaluate_torch(drop_model, X_test_t, y_test_t)
    report_overfitting(results, ['mlp_sgd', 'mlp_dropout50'])
    print()
    describe_curve(f'SGD (batch={SGD_BATCH}) + dropout 0.5', drop_train, drop_val)

    # --- 4e: what the surviving mistakes look like ---
    print('\n4e: error analysis of the misclassified test neurons')
    error_analysis(sgd_model, X_train, y_train, X_test, y_test, X_test_t, y_test_t)
    return results


def main():
    set_seed()
    X_train, X_test, y_train, y_test = load_data()
    log_reg, _ = train_log_r(X_train, X_test, y_train, y_test)
    comp_mlpVSlog_per(X_train, X_test, y_train, y_test, log_reg=log_reg)
    sensitivity, sensitivity_diag = sensitivity_analysis(X_train, y_train)
    report_sensitivity(sensitivity, sensitivity_diag)
    run_q4(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
