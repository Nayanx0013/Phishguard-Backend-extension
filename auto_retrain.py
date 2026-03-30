import os
import sqlite3
import threading
import time
import pickle
import numpy as np
from datetime import datetime
from urllib.parse import urlparse   # FIX: moved to top — was imported inside loops

# ── Thresholds ────────────────────────────────────────────────────────────────
SAFE_REPORT_THRESHOLD      = 3
PHISHING_REPORT_THRESHOLD  = 2
CHECK_INTERVAL_MINUTES     = 30
MIN_NEW_REPORTS_TO_RETRAIN = 5

# New model must achieve at least this accuracy to replace the current one
MIN_ACCURACY_TO_REPLACE    = 0.88

DB = "scans.db"


# FIX: Use a shared get_db() that respects Turso when configured,
# instead of always hardcoding sqlite3.connect(DB).
# If app.py defines get_db, we import it; otherwise fall back to local SQLite.
def get_db():
    try:
        from app import get_db as app_get_db
        return app_get_db()
    except Exception:
        return sqlite3.connect(DB)


def init_retrain_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verified_urls (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            url           TEXT UNIQUE,
            label         TEXT,
            report_count  INTEGER DEFAULT 0,
            verified_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            used_in_train INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            new_safe     INTEGER DEFAULT 0,
            new_phishing INTEGER DEFAULT 0,
            accuracy     REAL DEFAULT 0.0,
            status       TEXT DEFAULT 'pending'
        )
    """)
    try:
        conn.execute("ALTER TABLE reports ADD COLUMN used_in_verify INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.commit()
    conn.close()
    print("✅ Auto-retrain DB tables ready")


def process_new_reports():
    conn = get_db()
    rows = conn.execute("""
        SELECT url, label, COUNT(*) as cnt
        FROM reports
        WHERE used_in_verify = 0
        GROUP BY url, label
    """).fetchall()

    new_safe     = []
    new_phishing = []

    for url, label, cnt in rows:
        label = label.lower().strip()

        if label == "safe" and cnt >= SAFE_REPORT_THRESHOLD:
            # FIX: Check for conflicting phishing reports before verifying as safe
            phish_count = conn.execute(
                "SELECT COUNT(*) FROM reports WHERE url=? AND label='phishing'", (url,)
            ).fetchone()[0]
            if phish_count >= PHISHING_REPORT_THRESHOLD:
                print(f"⚠️  Conflicting reports for {url[:50]} — skipping")
            else:
                new_safe.append(url)
                conn.execute("""
                    INSERT OR REPLACE INTO verified_urls (url, label, report_count)
                    VALUES (?, 'safe', ?)
                """, (url, cnt))
                conn.execute(
                    "UPDATE reports SET used_in_verify=1 WHERE url=? AND label='safe'", (url,)
                )
                print(f"✅ Verified SAFE ({cnt} reports): {url[:60]}")

        elif label == "phishing" and cnt >= PHISHING_REPORT_THRESHOLD:
            # FIX: Also check for conflicting safe reports before verifying as phishing
            #      Previously this was missing — 10 safe reports + 2 phishing would
            #      still get marked as verified phishing
            safe_count = conn.execute(
                "SELECT COUNT(*) FROM reports WHERE url=? AND label='safe'", (url,)
            ).fetchone()[0]
            if safe_count >= SAFE_REPORT_THRESHOLD:
                print(f"⚠️  Conflicting reports for {url[:50]} — skipping")
            else:
                new_phishing.append(url)
                conn.execute("""
                    INSERT OR REPLACE INTO verified_urls (url, label, report_count)
                    VALUES (?, 'phishing', ?)
                """, (url, cnt))
                conn.execute(
                    "UPDATE reports SET used_in_verify=1 WHERE url=? AND label='phishing'", (url,)
                )
                print(f"✅ Verified PHISHING ({cnt} reports): {url[:60]}")

    conn.commit()
    conn.close()
    return new_safe, new_phishing


def get_verified_training_data():
    conn = get_db()
    rows = conn.execute(
        "SELECT url, label FROM verified_urls WHERE used_in_train = 0"
    ).fetchall()
    conn.close()
    return rows


def mark_as_trained(urls):
    conn = get_db()
    for url in urls:
        conn.execute("UPDATE verified_urls SET used_in_train=1 WHERE url=?", (url,))
    conn.commit()
    conn.close()


def get_verified_safe_domains():
    conn = get_db()
    rows = conn.execute(
        "SELECT url FROM verified_urls WHERE label='safe' AND used_in_train=1"
    ).fetchall()
    conn.close()
    domains = set()
    for (url,) in rows:
        try:
            # FIX: urlparse now imported at top, not inside this loop
            domain = urlparse(url).netloc.lower().replace("www.", "")
            if domain:
                domains.add(domain)
        except Exception:
            pass
    return domains


def retrain_model(new_verified_urls, reload_callback=None):
    print("\n🔄 AUTO-RETRAIN TRIGGERED")
    print("=" * 50)

    conn         = get_db()
    safe_count   = sum(1 for u, l in new_verified_urls if l == "safe")
    phish_count  = sum(1 for u, l in new_verified_urls if l == "phishing")
    log_id       = conn.execute(
        "INSERT INTO retrain_log (new_safe, new_phishing, status) VALUES (?,?,'running')",
        (safe_count, phish_count)
    ).lastrowid
    conn.commit()
    conn.close()

    try:
        from features import extract_features, features_to_list, get_feature_count
        try:
            from xgboost import XGBClassifier
            USE_XGBOOST = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            USE_XGBOOST = False

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        csv_path     = "dataset/phishing_urls.csv"
        base_X, base_y = [], []

        if os.path.exists(csv_path):
            print(f"  Loading base dataset from {csv_path}...")
            df        = pd.read_csv(csv_path)
            label_col = next((c for c in ["label", "status", "phishing", "Result", "class"] if c in df.columns), None)
            url_col   = next((c for c in ["url", "URL", "Url"] if c in df.columns), None)
            if label_col and url_col:
                le     = LabelEncoder()
                labels = le.fit_transform(df[label_col].astype(str))
                urls   = df[url_col].tolist()
                for i, (url, label) in enumerate(zip(urls, labels)):
                    if i % 3000 == 0:
                        print(f"  Base data: {i}/{len(urls)}...")
                    try:
                        feats = extract_features(str(url), deep_scan=False)
                        base_X.append(features_to_list(feats))
                        base_y.append(int(label))
                    except Exception:
                        pass
                print(f"  Base dataset: {len(base_X)} samples")

        extra_X, extra_y = [], []
        print(f"  Adding {len(new_verified_urls)} verified user reports...")
        for url, label in new_verified_urls:
            try:
                feats = extract_features(str(url), deep_scan=False)
                flist = features_to_list(feats)
                y_val = 0 if label.lower() == "safe" else 1

                # FIX: Balanced multiplier — both classes now get 5x weight
                # Previously safe=3x, phishing=5x which biased model toward false positives
                multiplier = 5
                for _ in range(multiplier):
                    extra_X.append(flist)
                    extra_y.append(y_val)
                print(f"  Added ×{multiplier}: {label.upper()} — {url[:55]}")
            except Exception as e:
                print(f"  Skip {url[:40]}: {e}")

        if not base_X and not extra_X:
            raise ValueError("No training data available")

        all_X = np.array(base_X + extra_X, dtype=np.float32)
        all_y = np.array(base_y + extra_y)
        print(f"\n  Total training samples: {len(all_X)}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            all_X, all_y, test_size=0.2, random_state=42, stratify=all_y
        )

        if USE_XGBOOST:
            model = XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, eval_metric="logloss", verbosity=0
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=5, random_state=42, n_jobs=-1
            )

        print("  Training model...")
        model.fit(X_tr, y_tr)

        y_pred   = model.predict(X_te)
        accuracy = accuracy_score(y_te, y_pred)
        print(f"  New model accuracy: {accuracy * 100:.2f}%")

        # Accuracy gating — only replace if new model is good enough
        if accuracy < MIN_ACCURACY_TO_REPLACE:
            print(f"  ⚠️  Accuracy {accuracy:.2%} below threshold {MIN_ACCURACY_TO_REPLACE:.2%}")
            print("  Keeping existing model — new model not good enough")
            conn = get_db()
            conn.execute(
                "UPDATE retrain_log SET status='rejected_low_accuracy', accuracy=? WHERE id=?",
                (round(accuracy, 4), log_id)
            )
            conn.commit()
            conn.close()
            return False, accuracy

        # Backup old model before replacing
        if os.path.exists("model.pkl"):
            os.rename("model.pkl", "model_backup.pkl")
            print("  Old model backed up → model_backup.pkl")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("  ✅ New model.pkl saved")

        trained_urls = [url for url, _ in new_verified_urls]
        mark_as_trained(trained_urls)

        conn = get_db()
        conn.execute(
            "UPDATE retrain_log SET status='done', accuracy=? WHERE id=?",
            (round(accuracy, 4), log_id)
        )
        conn.commit()
        conn.close()

        print(f"\n✅ RETRAIN COMPLETE — Accuracy: {accuracy * 100:.2f}%")
        print("=" * 50)

        if reload_callback:
            reload_callback()
            print("✅ Models hot-reloaded — no server restart needed!")

        return True, accuracy

    except Exception as e:
        print(f"\n❌ RETRAIN FAILED: {e}")
        if os.path.exists("model_backup.pkl") and not os.path.exists("model.pkl"):
            os.rename("model_backup.pkl", "model.pkl")
            print("  Restored backup model")
        conn = get_db()
        conn.execute("UPDATE retrain_log SET status='failed' WHERE id=?", (log_id,))
        conn.commit()
        conn.close()
        return False, 0.0


class AutoRetrainWatcher:
    def __init__(self, reload_callback=None, interval_minutes=CHECK_INTERVAL_MINUTES):
        self.reload_callback   = reload_callback
        self.interval_seconds  = interval_minutes * 60
        self.is_retraining     = False
        self.last_retrain      = None
        self.dynamic_whitelist = set()
        self._thread           = None
        init_retrain_db()

    def start(self):
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        print(f"✅ Auto-retrain watcher started (checks every {self.interval_seconds // 60} min)")
        print(f"   Thresholds: {SAFE_REPORT_THRESHOLD} safe | {PHISHING_REPORT_THRESHOLD} phishing | min accuracy {MIN_ACCURACY_TO_REPLACE:.0%}")

    def _watch_loop(self):
        time.sleep(120)
        while True:
            try:
                self._check_and_retrain()
            except Exception as e:
                print(f"⚠️  Auto-retrain watcher error: {e}")
            time.sleep(self.interval_seconds)

    def _check_and_retrain(self):
        if self.is_retraining:
            print("⏳ Retraining already in progress — skipping check")
            return
        print(f"\n🔍 Checking for verified reports... [{datetime.now().strftime('%H:%M')}]")
        new_safe, new_phishing = process_new_reports()
        if new_safe:
            print(f"  New verified SAFE domains: {len(new_safe)}")
            for url in new_safe:
                try:
                    # FIX: urlparse now imported at top, not re-imported here
                    domain = urlparse(url).netloc.lower().replace("www.", "")
                    if domain:
                        self.dynamic_whitelist.add(domain)
                        print(f"  → Added to runtime whitelist: {domain}")
                except Exception:
                    pass
        if new_phishing:
            print(f"  New verified PHISHING URLs: {len(new_phishing)}")
        pending = get_verified_training_data()
        if len(pending) < MIN_NEW_REPORTS_TO_RETRAIN:
            remaining = MIN_NEW_REPORTS_TO_RETRAIN - len(pending)
            if pending:
                print(f"  {len(pending)} verified reports pending — need {remaining} more")
            else:
                print("  No new verified reports — all good")
            return
        print(f"\n🚀 {len(pending)} verified reports ready — triggering auto-retrain!")
        self.is_retraining = True
        def do_retrain():
            try:
                success, acc = retrain_model(pending, self.reload_callback)
                if success:
                    self.last_retrain = datetime.now()
                    print(f"✅ Auto-retrain complete at {self.last_retrain.strftime('%H:%M')}")
            finally:
                self.is_retraining = False
        threading.Thread(target=do_retrain, daemon=True).start()

    def is_domain_whitelisted(self, domain):
        domain_clean = domain.lower().replace("www.", "")
        return domain_clean in self.dynamic_whitelist

    def get_status(self):
        conn = get_db()
        pending_safe      = conn.execute("SELECT COUNT(DISTINCT url) FROM reports WHERE label='safe' AND used_in_verify=0").fetchone()[0]
        pending_phishing  = conn.execute("SELECT COUNT(DISTINCT url) FROM reports WHERE label='phishing' AND used_in_verify=0").fetchone()[0]
        verified_safe     = conn.execute("SELECT COUNT(*) FROM verified_urls WHERE label='safe'").fetchone()[0]
        verified_phishing = conn.execute("SELECT COUNT(*) FROM verified_urls WHERE label='phishing'").fetchone()[0]
        last = conn.execute(
            "SELECT triggered_at, accuracy, status FROM retrain_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return {
            "is_retraining":             self.is_retraining,
            "last_retrain":              self.last_retrain.isoformat() if self.last_retrain else None,
            "last_retrain_info":         {"at": last[0], "accuracy": last[1], "status": last[2]} if last else None,
            "pending_safe_reports":      pending_safe,
            "pending_phishing_reports":  pending_phishing,
            "verified_safe_urls":        verified_safe,
            "verified_phishing_urls":    verified_phishing,
            "dynamic_whitelist_size":    len(self.dynamic_whitelist),
            "thresholds": {
                "safe_reports_needed":     SAFE_REPORT_THRESHOLD,
                "phishing_reports_needed": PHISHING_REPORT_THRESHOLD,
                "min_to_trigger_retrain":  MIN_NEW_REPORTS_TO_RETRAIN,
                "min_accuracy_to_replace": MIN_ACCURACY_TO_REPLACE,
            }
        }