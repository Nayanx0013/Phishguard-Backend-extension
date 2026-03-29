

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from features import extract_features, features_to_list, get_feature_count

# Try XGBoost first, fall back to Random Forest
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("✅ XGBoost available — using for best accuracy")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGBOOST = False
    print("⚠️  XGBoost not installed — using Random Forest")
    print("   Install for +0.5% accuracy: pip install xgboost")

print("=" * 60)
print(f"  PhishGuard Trainer v4.0 ({get_feature_count()} features)")
print("=" * 60)


def load_dataset(csv_path):
    print(f"\n[1/5] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"      Shape: {df.shape}")

    label_col = next((c for c in ["label","status","phishing","Result","class"] if c in df.columns), None)
    url_col   = next((c for c in ["url","URL","Url"] if c in df.columns), None)

    if not label_col: raise ValueError(f"No label column found. Columns: {list(df.columns)}")
    if not url_col:   raise ValueError(f"No URL column found. Columns: {list(df.columns)}")

    print(f"      Label: '{label_col}' | URL: '{url_col}'")

    le     = LabelEncoder()
    labels = le.fit_transform(df[label_col].astype(str))
    print(f"      Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"      Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return df[url_col].tolist(), labels


def extract_all_features(urls, labels):
    import signal
    import socket
    socket.setdefaulttimeout(3)  # Hard cap on all socket/network calls

    def _timeout_handler(signum, frame):
        raise TimeoutError("extract_features timed out")

    USE_SIGNAL = hasattr(signal, "SIGALRM")  # Linux/Mac only; Windows skips this

    n = get_feature_count()
    print(f"\n[2/5] Extracting {n} features from {len(urls)} URLs...", flush=True)
    print(f"      (deep_scan=False | socket timeout=3s | progress every 100)", flush=True)

    X, y, skipped = [], [], 0

    for i, (url, label) in enumerate(zip(urls, labels)):
        if i % 100 == 0:
            print(f"      Progress: {i}/{len(urls)}...", flush=True)
        try:
            if USE_SIGNAL:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(3)
            feats = extract_features(str(url), deep_scan=False)
            if USE_SIGNAL:
                signal.alarm(0)
            X.append(features_to_list(feats))
            y.append(label)
        except TimeoutError:
            skipped += 1
            if skipped <= 5:
                print(f"      ⚠️  Timeout [{i}]: {str(url)[:70]}", flush=True)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"      ⚠️  Error  [{i}]: {e}", flush=True)
        finally:
            if USE_SIGNAL:
                signal.alarm(0)

    if skipped:
        print(f"      Total skipped: {skipped} URLs", flush=True)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"      Feature matrix: {X.shape} ✅", flush=True)
    return X, y


def generate_synthetic_data(n=3000):
    print(f"\n[1/5] No dataset — generating {n} synthetic samples...")
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
    X, y = [], []
    for _ in range(n//2):
        url = np.random.choice(safe_urls)
        X.append(features_to_list(extract_features(url, deep_scan=False)))
        y.append(0)
    for _ in range(n//2):
        url = np.random.choice(phishing_urls)
        X.append(features_to_list(extract_features(url, deep_scan=False)))
        y.append(1)
    return np.array(X, dtype=np.float32), np.array(y)


def train_model(X, y):
    print(f"\n[3/5] Splitting data (80/20)...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {len(X_tr)} | Test: {len(X_te)}")

    if USE_XGBOOST:
        print(f"\n[4/5] Training XGBoost (300 trees)...")
        model = XGBClassifier(
            n_estimators=300, max_depth=8,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            n_jobs=-1, eval_metric="logloss",
            verbosity=0,
        )
    else:
        print(f"\n[4/5] Training Random Forest (200 trees)...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )

    model.fit(X_tr, y_tr)
    n_feat = getattr(model, "n_features_in_", X_tr.shape[1])
    print(f"      Features used: {n_feat} ✅")

    print(f"\n[5/5] Evaluating...")
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    unique = np.unique(y_te)
    kw = {} if (len(unique)==2 and all(isinstance(v,(int,np.integer)) for v in unique)) else {"average":"weighted"}
    prec = precision_score(y_te, y_pred, zero_division=0, **kw)
    rec  = recall_score(y_te, y_pred, zero_division=0, **kw)
    f1   = f1_score(y_te, y_pred, zero_division=0, **kw)

    print("\n" + "="*60)
    print(f"  {'XGBOOST' if USE_XGBOOST else 'RANDOM FOREST'} RESULTS")
    print("="*60)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(classification_report(y_te, y_pred))

    cv = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print(f"  5-Fold CV : {cv.mean():.4f} ± {cv.std():.4f}")
    print("="*60)
    return model


if __name__ == "__main__":
    csv_path = "dataset/phishing_urls.csv"
    if os.path.exists(csv_path):
        urls, labels = load_dataset(csv_path)
        X, y = extract_all_features(urls, labels)
    else:
        print(f"\n⚠️  Dataset not found at '{csv_path}'")
        X, y = generate_synthetic_data(3000)

    model = train_model(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    n_feat = getattr(model, "n_features_in_", X.shape[1])
    print(f"\n✅ model.pkl saved ({n_feat} features)")
    print(f"\n🚀 Now run: python train_dl.py then python app.py")