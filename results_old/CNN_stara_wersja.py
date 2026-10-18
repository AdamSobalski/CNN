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
import matplotlib
matplotlib.use("Agg")  # wykresy tylko zapisywane do plików (wiele eksperymentów pod rząd)
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

#Konfiguracja eksperymentów
# Parametry stałe dla wszystkich eksperymentów
CONFIG = {
    "weight_decay": 0.0,
    "batch_size": 128,
    "epochs": 30,
}
# Porównywane warianty: każdy model x każdy optimizer z dobranymi LR x augmentacja tak/nie
MODEL_NAMES = ["simple_cnn", "vgg16", "resnet18"]
OPTIMIZER_LRS = {"sgd": [0.1, 0.01], "adam": [0.01, 0.001]}
AUGMENT = [True, False]

EXPERIMENTS = [
    {**CONFIG, "model": m, "optimizer": opt, "lr": lr, "augment": aug}
    for m in MODEL_NAMES
    for opt, lrs in OPTIMIZER_LRS.items()
    for lr in lrs
    for aug in AUGMENT
]
# Każdy eksperyment jest powtarzany dla każdego seeda; do finalnych wyników np. [42, 43, 44]
SEEDS = [42, 43, 44]

VAL_SIZE = 5000        # 45 000 train + 5 000 validation; test (10 000) tylko do oceny końcowej
SPLIT_SEED = 0         # stały podział train/val, identyczny dla wszystkich eksperymentów
DETERMINISTIC = True   # True: powtarzalność (cudnn.deterministic), False: szybciej (cudnn.benchmark)
RESULTS_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["samolot", "auto", "ptak", "kot", "jeleń",
           "pies", "żaba", "koń", "statek", "ciężarówka"]


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
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device=DEVICE).view(1, 3, 1, 1)


def load_cifar10(train):
    ds = torchvision.datasets.CIFAR10(root="./data", train=train, download=True)
    x = torch.tensor(ds.data).float().div(255)  # N x 32 x 32 x 3, wartości 0-1
    y = torch.tensor(ds.targets)
    return x, y


def load_data():
    x, y = load_cifar10(train=True)
    x_test, y_test = load_cifar10(train=False)

    # Stratyfikowany podział: po VAL_SIZE/10 obrazów z każdej klasy do walidacji
    g = torch.Generator().manual_seed(SPLIT_SEED)
    per_class = VAL_SIZE // len(CLASSES)
    val_idx = []
    for c in range(len(CLASSES)):
        idx = torch.where(y == c)[0]
        val_idx.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
    val_idx = torch.cat(val_idx)
    train_mask = torch.ones(len(y), dtype=torch.bool)
    train_mask[val_idx] = False

    splits = {
        "train": (x[train_mask], y[train_mask]),
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


# Nowe architektury dopisujemy tutaj: "nazwa": klasa/funkcja tworząca model
MODELS = {
    "simple_cnn": SimpleCNN,
    "vgg16": VGG16,
    "resnet18": resnet18,
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


def plot_summary():
    # Wykresy zbiorcze z results.csv: accuracy (średnia ± SD po seedach) i zysk z augmentacji
    rows = read_results()
    variants = [(opt, lr) for opt, lrs in OPTIMIZER_LRS.items() for lr in lrs]
    labels = [f"{opt.upper() if opt == 'sgd' else opt.capitalize()}\nLR={lr}" for opt, lr in variants]

    def accs(model, opt, lr, aug):
        # {seed: accuracy [%]} dla jednej konfiguracji
        return {r["seed"]: float(r["test_acc"]) * 100 for r in rows
                if r["model"] == model and r["optimizer"] == opt and float(r["lr"]) == lr and r["augment"] == str(aug)}

    def mean_sd(values):
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    # 1. Accuracy każdej konfiguracji
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(15, 5), sharey=True)
    x = np.arange(len(variants))
    for ax, m in zip(axes, MODEL_NAMES):
        for k, (aug, label) in enumerate([(True, "z augmentacją"), (False, "bez augmentacji")]):
            stats = [mean_sd(list(accs(m, opt, lr, aug).values())) for opt, lr in variants]
            means, sds = [s[0] for s in stats], [s[1] for s in stats]
            pos = x + (k - 0.5) * 0.38
            ax.bar(pos, means, 0.38, yerr=sds, capsize=4, label=label)
            for p, mu, sd in zip(pos, means, sds):
                ax.text(p, mu + sd + 0.3, f"{mu:.1f}", ha="center", fontsize=8)
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(70, 93)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy na zbiorze testowym [%]")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Accuracy (test) – średnia ± SD z {len(SEEDS)} seedów (oś Y od 70%)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "summary_accuracy.png"), dpi=150)
    plt.close(fig)

    # 2. Zysk z augmentacji: różnica z augm. - bez augm. liczona dla tych samych seedów
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(MODEL_NAMES))
    for k, ((opt, lr), label) in enumerate(zip(variants, labels)):
        stats = []
        for m in MODEL_NAMES:
            a, n = accs(m, opt, lr, True), accs(m, opt, lr, False)
            stats.append(mean_sd([a[s] - n[s] for s in a if s in n]))
        pos = x + (k - 1.5) * 0.2
        ax.bar(pos, [s[0] for s in stats], 0.2, yerr=[s[1] for s in stats], capsize=3,
               label=label.replace("\n", ", "))
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Zmiana accuracy [pp]")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Zysk z augmentacji danych – średnia ± SD z {len(SEEDS)} seedów")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "summary_augmentation.png"), dpi=150)
    plt.close(fig)


def plot_misclassified(run_name, splits, n=16):
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
                 "seed", "params", "train_time_s", "best_epoch", "val_acc", "test_acc",
                 "test_precision_macro", "test_recall_macro", "test_f1_macro", "timestamp"]
GROUP_FIELDS = ["model", "optimizer", "lr", "weight_decay", "batch_size", "epochs", "augment"]
SUMMARY_METRICS = ["test_acc", "test_precision_macro", "test_recall_macro", "test_f1_macro", "train_time_s"]


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


def summarize_results():
    # Średnia ± SD po seedach dla każdej konfiguracji -> results/summary.csv
    groups = {}
    for r in read_results():
        groups.setdefault(tuple(r[k] for k in GROUP_FIELDS), []).append(r)

    summary = []
    for key, rows in groups.items():
        s = {**dict(zip(GROUP_FIELDS, key)), "n_runs": len(rows), "params": rows[0]["params"]}
        for m in SUMMARY_METRICS:
            values = [float(r[m]) for r in rows]
            s[m + "_mean"] = round(statistics.mean(values), 4)
            s[m + "_sd"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        summary.append(s)

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nPodsumowanie (test, średnia ± SD po seedach):")
    for s in summary:
        print(f"  {s['model']:10s} | {s['optimizer']:4s} lr={s['lr']:5s} | aug={s['augment']:5s} | n={s['n_runs']} | "
              f"params={int(s['params']):,} | czas={s['train_time_s_mean']:.0f} s | "
              f"acc: {s['test_acc_mean']:.4f} ± {s['test_acc_sd']:.4f} | "
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
    print(f"Liczba parametrów: {n_params:,}")

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

    row = {"run_name": run_name, **config, "params": n_params, "train_time_s": round(train_time, 2),
           "best_epoch": best_epoch, "val_acc": round(best_val_acc, 4), "test_acc": round(test["acc"], 4),
           "test_precision_macro": round(test["precision"], 4), "test_recall_macro": round(test["recall"], 4),
           "test_f1_macro": round(test["f1"], 4), "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({**row, "history": history}, f, indent=2)
    save_result(row)


def main():
    if DEVICE.type == "cuda":
        print(f"Trening na GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Trening na CPU (GPU niedostępne)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits = load_data()
    print("Rozmiary zbiorów: " + ", ".join(f"{k}={v[0].size(0)}" for k, v in splits.items()))
    print(f"Liczba treningów do wykonania: {len(EXPERIMENTS) * len(SEEDS)}")

    # Treningi już zapisane w results.csv są pomijane, więc przerwany skrypt można po prostu uruchomić ponownie
    done = {r["run_name"] for r in read_results()}
    for config in EXPERIMENTS:
        for seed in SEEDS:
            config_seed = {**config, "seed": seed}
            if make_run_name(config_seed) in done:
                print(f"Pomijam (już wykonany): {make_run_name(config_seed)}")
                continue
            run_experiment(config_seed, splits)

    summary = summarize_results()
    plot_summary()
    # Przykłady błędów dla najlepszej (wg średniej accuracy) konfiguracji każdego modelu, pierwszy seed
    for m in MODEL_NAMES:
        best = max((s for s in summary if s["model"] == m), key=lambda s: s["test_acc_mean"])
        plot_misclassified(make_run_name({**best, "augment": best["augment"] == "True", "seed": SEEDS[0]}), splits)


if __name__ == "__main__":
    main()
