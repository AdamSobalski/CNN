import csv
import json
import os
import random
import statistics
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.flop_counter import FlopCounterMode
import matplotlib
matplotlib.use("Agg")  # wykresy tylko zapisywane do plików (wiele eksperymentów pod rząd)
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

#Konfiguracja eksperymentów
# Parametry stałe dla wszystkich eksperymentów (batch size i liczba epok nie są badane)
CONFIG = {
    "weight_decay": 0.0,
    "batch_size": 128,
    "epochs": 30,
}
# Etap I (E01–E04): porównanie architektur w takich samych warunkach uczenia
STAGE1_MODELS = ["simple_cnn", "vgg16", "resnet18", "mobilenet_v2"]
STAGE1_SETTINGS = {"optimizer": "adam", "lr": 0.001, "augment": True}
# Etap II (E05–E09): parametry uczenia badane tylko na najlepszym modelu z etapu I
STAGE2_SETTINGS = [
    {"optimizer": "adam", "lr": 0.001, "augment": False},  # E05: bez augmentacji
    {"optimizer": "adam", "lr": 0.001, "augment": True},   # E06: punkt odniesienia
    {"optimizer": "adam", "lr": 0.0001, "augment": True},  # E07: mniejszy learning rate
    {"optimizer": "sgd", "lr": 0.1, "augment": True},      # E08: SGD z dużym learning rate
    {"optimizer": "sgd", "lr": 0.01, "augment": True},     # E09: SGD z mniejszym learning rate
]
# Każdy eksperyment jest powtarzany dla każdego seeda; wyniki podajemy jako średnia ± SD
SEEDS = [42, 43, 44]

VAL_SIZE = 5000        # 45 000 train + 5 000 validation; oryginalne 10 000 test tylko do oceny końcowej
SPLIT_SEED = 0         # stały podział, identyczny dla wszystkich eksperymentów
DETERMINISTIC = True   # True: powtarzalność (cudnn.deterministic), False: szybciej (cudnn.benchmark)
RESULTS_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["samolot", "auto", "ptak", "kot", "jeleń",
           "pies", "żaba", "koń", "statek", "ciężarówka"]
MODEL_LABELS = {"simple_cnn": "SimpleCNN", "vgg16": "VGG16", "resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2"}


def experiment_plan(best_model=None):
    # Lista (ID, konfiguracja). Etap II dochodzi dopiero, gdy znany jest najlepszy model etapu I
    plan = [(f"E{i:02d}", {**CONFIG, **STAGE1_SETTINGS, "model": m})
            for i, m in enumerate(STAGE1_MODELS, start=1)]
    if best_model is not None:
        plan += [(f"E{i:02d}", {**CONFIG, **s, "model": best_model})
                 for i, s in enumerate(STAGE2_SETTINGS, start=len(STAGE1_MODELS) + 1)]
    return plan


#Powtarzalność
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = not DETERMINISTIC


#Dane CIFAR-10
# Cały zbiór (~700 MB jako float32) mieści się w pamięci GPU, więc trzymamy go tam
# i robimy augmentację na GPU. Odpada kopiowanie CPU->GPU i dekodowanie obrazów
# w każdej epoce, które przy tak małych obrazach i modelu jest wąskim gardłem DataLoadera.
# Średnia i odchylenie standardowe kanałów RGB do normalizacji; liczone w load_data() tylko na zbiorze uczącym
MEAN, STD = None, None


def load_cifar10(train):
    ds = torchvision.datasets.CIFAR10(root="./data", train=train, download=True)
    x = torch.tensor(ds.data).float().div(255)  # N x 32 x 32 x 3, wartości 0-1
    y = torch.tensor(ds.targets)
    return x, y


def load_data():
    # Oryginalna część testowa CIFAR-10 (10 000 obrazów) służy wyłącznie do oceny końcowej.
    # Z części uczącej (50 000) wydzielamy zbiór walidacyjny: po VAL_SIZE/10 obrazów z każdej klasy
    x, y = load_cifar10(train=True)
    x_test, y_test = load_cifar10(train=False)

    g = torch.Generator().manual_seed(SPLIT_SEED)
    per_class = VAL_SIZE // len(CLASSES)
    val_idx = []
    for c in range(len(CLASSES)):
        idx = torch.where(y == c)[0]
        val_idx.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
    val_idx = torch.cat(val_idx)
    train_mask = torch.ones(len(y), dtype=torch.bool)
    train_mask[val_idx] = False
    # Każdy obraz części uczącej trafia do dokładnie jednego zbioru: train albo val
    assert len(torch.unique(val_idx)) == VAL_SIZE
    assert int(train_mask.sum()) == len(y) - VAL_SIZE

    global MEAN, STD
    x_train = x[train_mask]
    MEAN = x_train.mean(dim=(0, 1, 2)).view(1, 3, 1, 1).to(DEVICE)
    STD = x_train.std(dim=(0, 1, 2)).view(1, 3, 1, 1).to(DEVICE)
    splits = {
        "train": (x_train, y[train_mask]),
        "val": (x[val_idx], y[val_idx]),
        "test": (x_test, y_test),
    }
    return {name: (xs.to(DEVICE), ys.to(DEVICE)) for name, (xs, ys) in splits.items()}


def augment(x):
    # Losowe odbicie lustrzane + losowe przycięcie 32x32 z obrazu z ramką 4 px (na GPU)
    b = x.size(0)
    flip = torch.rand(b, device=DEVICE) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(2), x)
    padded = F.pad(x, (0, 0, 4, 4, 4, 4))
    dy = torch.randint(0, 9, (b, 1, 1), device=DEVICE)
    dx = torch.randint(0, 9, (b, 1, 1), device=DEVICE)
    r = torch.arange(32, device=DEVICE)
    rows = dy + r.view(1, 32, 1)
    cols = dx + r.view(1, 1, 32)
    return padded[torch.arange(b, device=DEVICE).view(-1, 1, 1), rows, cols]


def batches(x, y, batch_size, shuffle=False, aug=False):
    n = x.size(0)
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        xb = augment(x[j]) if aug else x[j]
        xb = (xb.permute(0, 3, 1, 2) - MEAN) / STD
        yield xb.contiguous(), y[j]


#Modele
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Blok 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Blok 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG16(nn.Module):
    # VGG16 (13 warstw splotowych + klasyfikator) z BatchNorm, w wersji dla obrazów 32x32:
    # po 5 poolingach zostaje mapa 512x1x1, więc klasyfikator jest mniejszy niż w oryginale (224x224)
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes=10):
        super().__init__()
        layers, in_ch = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU()]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def resnet18(num_classes=10):
    # ResNet18 z torchvision, trenowany od zera (bez wag ImageNet), dostosowany do 32x32:
    # pierwszy splot 3x3 ze stride 1 zamiast 7x7 ze stride 2 i bez początkowego max poolingu
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def mobilenet_v2(num_classes=10):
    # MobileNetV2 z torchvision, trenowany od zera, dostosowany do 32x32: pierwszy splot
    # i pierwszy blok z 24 kanałami mają stride 1 zamiast 2, więc obraz jest zmniejszany
    # 8 razy zamiast 32
    model = torchvision.models.mobilenet_v2(num_classes=num_classes)
    model.features[0][0].stride = (1, 1)
    model.features[2].conv[1][0].stride = (1, 1)
    return model


# Nowe architektury dopisujemy tutaj: "nazwa": klasa/funkcja tworząca model
MODELS = {
    "simple_cnn": SimpleCNN,
    "vgg16": VGG16,
    "resnet18": resnet18,
    "mobilenet_v2": mobilenet_v2,
}


def build_model(name):
    return MODELS[name](num_classes=len(CLASSES))


def build_optimizer(model, config):
    name, lr, wd = config["optimizer"], config["lr"], config["weight_decay"]
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Nieznany optimizer: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model):
    # Liczba operacji (FLOP) potrzebnych do klasyfikacji jednego obrazu 32x32 (jedno przejście w przód)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 3, 32, 32, device=DEVICE))
    return counter.get_total_flops()


#Trening i ewaluacja
def train_one_epoch(model, data, criterion, optimizer, batch_size, aug):
    model.train()
    x, y = data
    total_loss = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    for images, labels in batches(x, y, batch_size, shuffle=True, aug=aug):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum()
    n = x.size(0)
    return total_loss.item() / n, correct.item() / n


def evaluate(model, data, criterion, batch_size):
    model.eval()
    x, y = data
    total_loss = torch.zeros((), device=DEVICE)
    preds = []
    with torch.no_grad():
        for images, labels in batches(x, y, batch_size):
            outputs = model(images)
            total_loss += criterion(outputs, labels) * images.size(0)
            preds.append(outputs.argmax(1))
    y_true = y.cpu().numpy()
    y_pred = torch.cat(preds).cpu().numpy()
    return {
        "loss": total_loss.item() / x.size(0),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(model, splits, criterion, optimizer, config):
    # Podczas treningu używamy tylko train i val; najlepsza epoka wybierana po val accuracy
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, None

    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, splits["train"], criterion, optimizer,
                                                config["batch_size"], config["augment"])
        val = evaluate(model, splits["val"], criterion, config["batch_size"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val["acc"])

        if val["acc"] > best_acc:
            best_acc, best_epoch = val["acc"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"Epoka {epoch:2d}/{config['epochs']} | "
              f"train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"val loss: {val['loss']:.4f}, acc: {val['acc']:.4f}")
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state)
    return history, best_epoch, best_acc, train_time


#Wykresy
def plot_curves(history, path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="trening")
    ax1.plot(epochs, history["val_loss"], "o-", label="walidacja")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoka")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="trening")
    ax2.plot(epochs, history["val_acc"], "o-", label="walidacja")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoka")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def config_label(config):
    opt = "SGD" if config["optimizer"] == "sgd" else "Adam"
    return f"{opt}, LR={config['lr']}\n{'z augmentacją' if config['augment'] else 'bez augmentacji'}"


def load_histories(config):
    # Historie uczenia ze wszystkich seedów jako tablice (seedy x epoki)
    histories = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, make_run_name({**config, "seed": seed}), "run.json")
        with open(path, encoding="utf-8") as f:
            histories.append(json.load(f)["history"])
    return {k: np.array([h[k] for h in histories]) for k in histories[0]}


def plot_accuracy_bars(rows, labels, path, title):
    # Accuracy na zbiorze testowym (średnia ± SD po seedach); oś Y nie zaczyna się od zera, co widać w tytule
    mu = np.array([s["test_acc_mean"] for s in rows]) * 100
    sd = np.array([s["test_acc_sd"] for s in rows]) * 100
    bottom = 5 * np.floor((mu - sd).min() / 5 - 1)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.2 * len(rows) + 1.5, 4.8))
    ax.bar(x, mu, 0.6, yerr=sd, capsize=5, color=[f"C{k}" for k in x])
    for xi, m, d in zip(x, mu, sd):
        ax.text(xi, m + d + 0.3, f"{m:.2f} ± {d:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom, min(100, (mu + sd).max() + 2.5))
    ax.set_ylabel("Accuracy na zbiorze testowym [%]")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title} – średnia ± SD z {len(SEEDS)} seedów (oś Y od {bottom:.0f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_curves(items, path, title, first_color=0):
    # Przebieg uczenia uśredniony po seedach; items: lista (etykieta, konfiguracja)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for k, (label, config) in enumerate(items, start=first_color):
        h = load_histories(config)
        epochs = np.arange(1, h["train_loss"].shape[1] + 1)
        ax1.plot(epochs, h["train_loss"].mean(0), "--", color=f"C{k}")
        ax1.plot(epochs, h["val_loss"].mean(0), "-", color=f"C{k}", label=label)
        mu = h["val_acc"].mean(0) * 100
        sd = h["val_acc"].std(0, ddof=1) * 100 if len(SEEDS) > 1 else np.zeros_like(mu)
        ax2.plot(epochs, mu, "-", color=f"C{k}", label=label)
        ax2.fill_between(epochs, mu - sd, mu + sd, color=f"C{k}", alpha=0.2)
    ax1.set_title("Loss: trening (linia przerywana), walidacja (linia ciągła)")
    ax2.set_title("Accuracy na zbiorze walidacyjnym [%] (średnia ± SD)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoka")
        ax.grid(alpha=0.3)
    ax1.legend()
    ax2.legend(loc="lower right")
    fig.suptitle(f"{title} – średnia z {len(SEEDS)} seedów")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(rows, path):
    # Jakość a koszt: accuracy (test) względem liczby parametrów i czasu treningu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for k, s in enumerate(rows):
        label = MODEL_LABELS[s["model"]]
        acc, acc_sd = s["test_acc_mean"] * 100, s["test_acc_sd"] * 100
        params, minutes, minutes_sd = s["params"] / 1e6, s["train_time_s_mean"] / 60, s["train_time_s_sd"] / 60
        ax1.errorbar(params, acc, yerr=acc_sd, fmt="o", ms=8, capsize=4, color=f"C{k}")
        ax2.errorbar(minutes, acc, xerr=minutes_sd, yerr=acc_sd, fmt="o", ms=8, capsize=4, color=f"C{k}")
        ax1.annotate(label, (params, acc), textcoords="offset points", xytext=(8, -14), fontsize=10)
        ax2.annotate(label, (minutes, acc), textcoords="offset points", xytext=(8, -14), fontsize=10)
    ax1.set_xscale("log")
    ax1.set_xticks([0.5, 1, 2, 5, 10, 20], labels=["0.5", "1", "2", "5", "10", "20"])
    ax1.set_xlabel("Liczba parametrów [mln] (skala logarytmiczna)")
    ax1.set_ylabel("Accuracy na zbiorze testowym [%]")
    ax2.set_xlabel(f"Czas treningu ({CONFIG['epochs']} epok) [min]")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.margins(x=0.2, y=0.15)
    fig.suptitle(f"Skuteczność a koszt obliczeniowy (etap I) – średnia ± SD z {len(SEEDS)} seedów")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(run_name, splits, n=12):
    # Losowe (stałe ziarno) przykłady błędów na zbiorze testowym: prawdziwa klasa -> predykcja (pewność modelu)
    run_dir = os.path.join(RESULTS_DIR, run_name)
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        model = build_model(json.load(f)["model"]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pth"), map_location=DEVICE))
    model.eval()

    x, y = splits["test"]
    with torch.no_grad():
        probs = torch.cat([F.softmax(model(images), 1) for images, _ in batches(x, y, 256)])
    conf, pred = probs.max(1)
    wrong = torch.where(pred != y)[0].cpu()
    wrong = wrong[torch.randperm(len(wrong), generator=torch.Generator().manual_seed(SPLIT_SEED))[:n]].tolist()

    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(x[i].cpu().numpy(), interpolation="nearest")
        ax.set_title(f"{CLASSES[y[i]]} → {CLASSES[pred[i]]}\n({conf[i]:.0%})", fontsize=10)
    fig.suptitle(f"Przykłady błędów (test): prawdziwa klasa → predykcja (pewność)\n{run_name}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(run_dir, "misclassified.png"), dpi=150)
    plt.close(fig)


#Zapis wyników
RESULT_FIELDS = ["run_name", "model", "optimizer", "lr", "weight_decay", "batch_size", "epochs", "augment",
                 "seed", "params", "flops", "train_time_s", "best_epoch", "train_loss", "val_loss", "train_acc",
                 "val_acc", "test_acc", "test_precision_macro", "test_recall_macro", "test_f1_macro", "timestamp"]
GROUP_FIELDS = ["model", "optimizer", "lr", "weight_decay", "batch_size", "epochs", "augment"]
SUMMARY_METRICS = ["val_acc", "test_acc", "test_precision_macro", "test_recall_macro", "test_f1_macro",
                   "train_loss", "val_loss", "best_epoch", "train_time_s"]


def make_run_name(config):
    return (f"{config['model']}_{config['optimizer']}_lr{config['lr']}_wd{config['weight_decay']}"
            f"_bs{config['batch_size']}_ep{config['epochs']}_{'aug' if config['augment'] else 'noaug'}"
            f"_seed{config['seed']}")


def read_results():
    path = os.path.join(RESULTS_DIR, "results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_result(row):
    # Ponowne uruchomienie tej samej konfiguracji i seeda zastępuje stary wiersz zamiast go dublować
    rows = [r for r in read_results() if r["run_name"] != row["run_name"]] + [row]
    with open(os.path.join(RESULTS_DIR, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(plan):
    # Średnia ± SD po seedach dla każdego eksperymentu z planu -> results/summary.csv
    results = {r["run_name"]: r for r in read_results()}
    summary = []
    for exp_id, config in plan:
        runs = [results[name] for name in (make_run_name({**config, "seed": seed}) for seed in SEEDS)
                if name in results]
        s = {"id": exp_id, **{k: config[k] for k in GROUP_FIELDS}, "n_runs": len(runs),
             "params": int(runs[0]["params"]), "flops": int(runs[0]["flops"])}
        for m in SUMMARY_METRICS:
            values = [float(r[m]) for r in runs]
            s[m + "_mean"] = round(statistics.mean(values), 4)
            s[m + "_sd"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        summary.append(s)

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nPodsumowanie (test, średnia ± SD po seedach):")
    for s in summary:
        print(f"  {s['id']} {s['model']:12s} | {s['optimizer']:4s} lr={s['lr']:<6} | aug={s['augment']!s:5s} | "
              f"n={s['n_runs']} | params={s['params']:,} | czas={s['train_time_s_mean']:.0f} s | "
              f"val acc: {s['val_acc_mean']:.4f} | acc: {s['test_acc_mean']:.4f} ± {s['test_acc_sd']:.4f} | "
              f"F1: {s['test_f1_macro_mean']:.4f} ± {s['test_f1_macro_sd']:.4f}")
    return summary


#Pojedynczy eksperyment
def run_experiment(config, splits):
    set_seed(config["seed"])
    run_name = make_run_name(config)
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== {run_name} ===")

    model = build_model(config["model"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    n_params = count_params(model)
    n_flops = count_flops(model)
    print(f"Liczba parametrów: {n_params:,} | FLOP na obraz: {n_flops / 1e6:.1f} mln")

    history, best_epoch, best_val_acc, train_time = train_model(model, splits, criterion, optimizer, config)

    # Jedyne użycie zbioru testowego: ocena najlepszego (wg walidacji) modelu
    test = evaluate(model, splits["test"], criterion, config["batch_size"])
    print(f"Najlepsza epoka (val): {best_epoch} | czas treningu: {train_time:.1f} s")
    print(f"TEST  Accuracy: {test['acc']:.4f} | Precision: {test['precision']:.4f} | "
          f"Recall: {test['recall']:.4f} | F1: {test['f1']:.4f} (macro)")

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    plot_curves(history, os.path.join(run_dir, "training_curves.png"), run_name)
    plot_confusion(test["y_true"], test["y_pred"], os.path.join(run_dir, "confusion_matrix.png"),
                   f"Macierz pomyłek (test) – {run_name}")
    # Precision / recall / F1 dla każdej klasy osobno
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(test["y_true"], test["y_pred"], target_names=CLASSES, digits=4))

    i = best_epoch - 1  # loss i accuracy treningowe z najlepszej epoki
    row = {"run_name": run_name, **config, "params": n_params, "flops": n_flops,
           "train_time_s": round(train_time, 2), "best_epoch": best_epoch,
           "train_loss": round(history["train_loss"][i], 4), "val_loss": round(history["val_loss"][i], 4),
           "train_acc": round(history["train_acc"][i], 4), "val_acc": round(best_val_acc, 4),
           "test_acc": round(test["acc"], 4), "test_precision_macro": round(test["precision"], 4),
           "test_recall_macro": round(test["recall"], 4), "test_f1_macro": round(test["f1"], 4),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    per_class = {name: dict(zip(CLASSES, fn(test["y_true"], test["y_pred"], average=None, zero_division=0).tolist()))
                 for name, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]}
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({**row, "history": history, "per_class": per_class,
                   "confusion_matrix": confusion_matrix(test["y_true"], test["y_pred"]).tolist()},
                  f, indent=2, ensure_ascii=False)
    save_result(row)


def run_all(plan, splits):
    # Treningi już zapisane w results.csv są pomijane, więc przerwany skrypt można po prostu uruchomić ponownie
    done = {r["run_name"] for r in read_results()}
    for exp_id, config in plan:
        for seed in SEEDS:
            config_seed = {**config, "seed": seed}
            if make_run_name(config_seed) in done:
                print(f"Pomijam (już wykonany): {exp_id} {make_run_name(config_seed)}")
                continue
            print(f"\n[{exp_id}]", end="")
            run_experiment(config_seed, splits)
            done.add(make_run_name(config_seed))


def main():
    if DEVICE.type == "cuda":
        print(f"Trening na GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Trening na CPU (GPU niedostępne)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits = load_data()
    print("Rozmiary zbiorów: " + ", ".join(f"{k}={v[0].size(0)}" for k, v in splits.items()))

    # Etap I: porównanie architektur (E01–E04)
    stage1_plan = experiment_plan()
    run_all(stage1_plan, splits)
    stage1 = summarize_results(stage1_plan)
    # Najlepszy model wybieramy po średniej accuracy walidacyjnej;
    # zbiór testowy nie bierze udziału w wyborze
    best_model = max(stage1, key=lambda s: s["val_acc_mean"])["model"]
    print(f"\nNajlepszy model etapu I (walidacja): {MODEL_LABELS[best_model]}")

    # Etap II: parametry uczenia dla najlepszego modelu (E05–E09)
    plan = experiment_plan(best_model)
    run_all(plan, splits)
    summary = summarize_results(plan)

    # Wykresy zbiorcze
    stage2 = summary[len(STAGE1_MODELS):]
    name = MODEL_LABELS[best_model]
    plot_accuracy_bars(stage1, [f"{s['id']}\n{MODEL_LABELS[s['model']]}" for s in stage1],
                       os.path.join(RESULTS_DIR, "stage1_accuracy.png"), "Etap I: porównanie architektur")
    plot_mean_curves([(f"{s['id']} {MODEL_LABELS[s['model']]}", s) for s in stage1],
                     os.path.join(RESULTS_DIR, "stage1_curves.png"), "Etap I: przebieg uczenia")
    plot_cost(stage1, os.path.join(RESULTS_DIR, "stage1_cost.png"))
    plot_accuracy_bars(stage2, [f"{s['id']}\n{config_label(s)}" for s in stage2],
                       os.path.join(RESULTS_DIR, "stage2_accuracy.png"), f"Etap II: parametry uczenia ({name})")
    plot_mean_curves([(f"{s['id']} {config_label(s)}".replace("\n", ", "), s) for s in stage2[:2]],
                     os.path.join(RESULTS_DIR, "stage2_augmentation.png"), f"Etap II: wpływ augmentacji ({name})")
    plot_mean_curves([(f"{s['id']} {config_label(s)}".replace("\n", ", "), s) for s in stage2[1:]],
                     os.path.join(RESULTS_DIR, "stage2_optimizer_lr.png"),
                     f"Etap II: optymalizator i learning rate ({name})", first_color=1)
    # Przykłady błędów dla modeli z etapu I (pierwszy seed)
    for s in stage1:
        plot_misclassified(make_run_name({**s, "seed": SEEDS[0]}), splits)


if __name__ == "__main__":
    main()
