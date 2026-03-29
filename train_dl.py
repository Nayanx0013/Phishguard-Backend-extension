

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from features import extract_features, features_to_list, get_feature_count

INPUT_SIZE = get_feature_count()   # Auto-detects from features.py (28)

print("=" * 60)
print(f"  PhishGuard Neural Network Trainer v4.0")
print(f"  (Auto-detected {INPUT_SIZE} features from features.py)")
print("=" * 60)


class PhishNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

PhishLSTM = PhishNet  # alias for app.py compatibility


def load_data():
    csv_path = "dataset/phishing_urls.csv"
    if os.path.exists(csv_path):
        import pandas as pd
        print(f"\n[1/5] Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        label_col = next((c for c in ["label","status","phishing","Result","class"] if c in df.columns), None)
        url_col   = next((c for c in ["url","URL","Url"] if c in df.columns), None)
        if not label_col or not url_col:
            raise ValueError(f"Cannot find url/label columns in {list(df.columns)}")

        le     = LabelEncoder()
        labels = le.fit_transform(df[label_col].astype(str))
        print(f"      Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        urls = df[url_col].tolist()
        print(f"      Extracting {INPUT_SIZE} features from {len(urls)} URLs...")

        X_list, y_list = [], []
        for i, (url, label) in enumerate(zip(urls, labels)):
            if i % 2000 == 0:
                print(f"      Progress: {i}/{len(urls)}...")
            try:
                feats = extract_features(str(url), deep_scan=False)
                X_list.append(features_to_list(feats))
                y_list.append(int(label))
            except Exception:
                continue

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        print(f"      Done! Shape: {X.shape}")

    else:
        print("\n[1/5] No dataset — using synthetic data...")
        safe_urls = [
            "https://www.google.com/search?q=python",
            "https://github.com/torvalds/linux",
            "https://stackoverflow.com/questions/12345",
            "https://docs.python.org/3/library/os.html",
            "https://www.wikipedia.org/wiki/Machine_learning",
            "https://www.bbc.com/news/technology",
            "https://www.reddit.com/r/cybersecurity/",
            "https://www.microsoft.com/en-us/windows",
            "https://www.youtube.com/watch?v=abc123",
            "https://www.linkedin.com/in/username",
        ]
        phishing_urls = [
            "http://192.168.1.1/login/paypal/verify?cmd=_login",
            "http://secure-paypal-login.tk/webscr?cmd=_login",
            "http://apple-id-verify.ml/account/confirm",
            "http://www.google.com.secure-login.xyz/signin",
            "http://192.168.0.1@malicious.com/banking",
            "http://paypal.com.account-verify.ga/login",
            "http://amazon-update-billing.click/account/signin",
            "http://microsoft-verify.tk/account/password",
            "http://asymptoticrelation.cfd/gateway/check.php",
            "http://ferdefor.vip/verify/identity",
            "http://secure-banking.cyou/login/confirm",
            "http://paypa1.com/signin/verify",
        ]
        np.random.seed(42)
        X_list, y_list = [], []
        for _ in range(1500):
            url = np.random.choice(safe_urls)
            X_list.append(features_to_list(extract_features(url, deep_scan=False)))
            y_list.append(0)
        for _ in range(1500):
            url = np.random.choice(phishing_urls)
            X_list.append(features_to_list(extract_features(url, deep_scan=False)))
            y_list.append(1)
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        print(f"      Generated {len(X)} samples")

    return X, y


def train():
    X, y = load_data()

    print("\n[2/5] Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print("\n[3/5] Splitting data (80/20)...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {len(X_tr)} | Test: {len(X_te)}")

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=32, shuffle=True
    )

    print(f"\n[4/5] Training Neural Network (input_size={INPUT_SIZE})...")
    model     = PhishNet(input_size=INPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn   = nn.CrossEntropyLoss()
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

    X_te_t = torch.tensor(X_te)
    y_te_t = torch.tensor(y_te)
    best_acc, best_state, no_improve = 0, None, 0

    for epoch in range(60):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).numpy()
        acc = accuracy_score(y_te_t.numpy(), preds)
        scheduler.step(total_loss)

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch+1) % 10 == 0:
            print(f"      Epoch {epoch+1:2d}/60 | Loss: {total_loss/len(train_dl):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if no_improve >= 15:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    print("\n[5/5] Final evaluation...")
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1).numpy()

    print("\n" + "="*60)
    print("  NEURAL NETWORK RESULTS")
    print("="*60)
    print(classification_report(y_te_t.numpy(), preds, target_names=["Safe","Phishing"]))
    print(f"  Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*60)
    return model, scaler


if __name__ == "__main__":
    model, scaler = train()

    torch.save(model.state_dict(), "lstm_model.pt")
    with open("char2idx.pkl", "wb") as f:
        pickle.dump({
            "type":       "feature_based",
            "input_size": INPUT_SIZE,
            "scaler":     scaler
        }, f)

    print(f"\n✅ lstm_model.pt saved (input_size={INPUT_SIZE})")
    print("✅ char2idx.pkl saved")
    print("\n🚀 Now restart: python app.py")