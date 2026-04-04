<div align="center">

<img src="https://img.shields.io/badge/PhishGuard-v5.1-blue?style=for-the-badge&logo=shield&logoColor=white" alt="PhishGuard v5.1"/>

# 🛡️ PhishGuard — AI-Powered Phishing Detection

**Real-time phishing protection powered by Ensemble ML + 10 Threat Intelligence Feeds**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Manifest_V3-4285F4?style=flat&logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/mv3/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

[🔴 Live Dashboard](https://nayanx0013-phishguard-extension.hf.space/dashboard) · [📦 Backend Repo](https://github.com/Nayanx0013/Phishguard-Backend-extension) · [🐛 Report a Bug](https://github.com/Nayanx0013/Phishguard-extension/issues)

</div>

---

## 📊 Stats at a Glance

| 🗄️ Threats in DB | 🔬 ML Features | 📡 Threat Feeds | 🎯 ML Confidence | 🤖 ML Models |
|:-:|:-:|:-:|:-:|:-:|
| **1.9M+** | **43** | **10+** | **99%** | **2** |

---

## ✨ What is PhishGuard?

PhishGuard is a **production-grade, AI-powered phishing detection system** that protects you from phishing attacks in real-time. It combines:

- 🤖 **Ensemble ML** — Random Forest + Neural Network (ONNX)
- 📡 **10+ Threat Intelligence Feeds** — VirusTotal, Google Safe Browsing, PhishTank & more
- 🧩 **Chrome Extension** — auto-scans every page you visit
- 📊 **Live Web Dashboard** — real-time stats, model health, and threat analytics
- 🔄 **Auto-Retrain Engine** — the model improves itself from user reports

> Built with AI-assisted development — focused on system architecture, ML pipeline design, and full-stack integration.

---

## 🚀 Install PhishGuard on Chrome (3 minutes, free)

> No Chrome Web Store needed. Works in Developer Mode — completely safe and standard practice for open-source extensions.

### Prerequisites

- Google Chrome browser
- Git installed → [Download Git](https://git-scm.com/downloads) *(or skip to the ZIP method below)*

---

### Step 1 — Download the extension

Open your **Terminal** (Mac/Linux) or **Command Prompt** (Windows) and run:

```bash
git clone https://github.com/Nayanx0013/Phishguard-extension.git
```

This creates a folder called `Phishguard-extension` on your computer.

> **Don't have Git?** Click the green **Code** button on this page → **Download ZIP** → unzip the folder.

---

### Step 2 — Open Chrome Extensions

Paste this into your Chrome address bar and press Enter:

```
chrome://extensions
```

---

### Step 3 — Enable Developer Mode

In the **top-right corner** of the Extensions page, toggle **Developer mode** ON.

> Three new buttons will appear at the top left of the page.

---

### Step 4 — Load the extension

1. Click **"Load unpacked"**
2. A file picker will open — navigate to the `Phishguard-extension` folder you downloaded
3. Select the **root folder** (the one that contains `manifest.json`) — don't go inside any subfolder
4. Click **Select Folder** (Windows) or **Open** (Mac)

✅ PhishGuard will now appear in your Extensions list.

---

### Step 5 — Pin it to your toolbar

Click the **puzzle piece icon 🧩** in Chrome's top-right → find PhishGuard → click the **pin icon 📌**

The PhishGuard shield will now appear in your toolbar on every tab. You're protected!

---

## 🧠 How It Works

### Ensemble ML Detection

PhishGuard engineers **43 features per URL** and runs them through two models whose verdicts are combined into a single confidence score:

| Model | Framework | Role |
|---|---|---|
| 🌲 Random Forest | scikit-learn | High recall, 43 features |
| 🧠 Neural Network | ONNX Runtime | Fast inference |
| 📊 Ensemble | Combined | Final confidence score 0–100% |

### Feature Engineering (43 features per URL)

- 🌐 **URL Entropy Analysis** — catches randomly-generated domains
- 🔤 **Homograph Attack Detection** — lookalike Unicode characters
- 🏷️ **Brand Impersonation Detection** — known brand names in suspicious positions
- 🔀 **Typosquatting Detection** — edit-distance similarity to known domains
- ⛓️ **Redirect Chain Analysis** — detects suspicious multi-hop redirects
- 📅 **Domain Age Verification** — flags newly registered domains
- 🔗 **URL Structure Features** — length, special chars, subdomains, TLD

---

## 📡 Threat Intelligence Feeds (10+)

| Feed | Description |
|---|---|
| VirusTotal | Cross-references 70+ AV engines |
| Google Safe Browsing | Google's phishing & malware database |
| PhishTank | Community phishing URL database |
| OpenPhish | Real-time phishing feed |
| URLhaus (abuse.ch) | Malware URL database |
| PhishingArmy | Community threat feed |
| mitchellkrogza | Phishing hosts list |
| VX Vault + 3 more | Additional real-time feeds |

> **Circuit Breakers:** If any API goes down, PhishGuard automatically falls back to remaining sources — no interruption to you.

---

## 🧩 Chrome Extension Features

### Auto-Scan & Popup
- 🟢 **LIVE Badge** — always shows real-time monitoring status
- ✅ **SITE IS SAFE** — large green checkmark when ML says safe
- 🚨 **PHISHING DETECTED** — bold red alert when threat is found
- 📊 **Ensemble Confidence Score** — 0–100% certainty with progress bar
- 🔬 **4-Model Breakdown** — RF · Neural Net · VirusTotal · GSB individual verdicts

### Gmail Link Scanner
Injects shield icons directly into Gmail next to every link:
- ✅ Green shield — link verified safe
- 🚨 Red shield — phishing link detected
- ⚠️ Yellow shield — suspicious, proceed with caution

### Full-Page Phishing Warning
When a high-confidence phishing site is detected, PhishGuard intercepts navigation and shows a full-page warning with the flagged URL, confidence score, and threat signals.

---

## 📊 Live Dashboard

🔗 **[nayanx0013-phishguard-extension.hf.space/dashboard](https://nayanx0013-phishguard-extension.hf.space/dashboard)**

- 📈 Total scans, threats blocked, clean sites, threat rate
- 📅 7-day daily activity graph
- 🍩 Safe vs phishing breakdown chart
- 🏆 Top phishing domains ranked list
- 🤖 Model health monitor (RF / NN / VirusTotal / GSB)
- 🔗 Batch URL scanner
- 🔄 Auto-retrain engine status

---

## 🔒 Security

| Feature | Details |
|---|---|
| 🔑 HMAC Admin Auth | Timing-safe comparison for admin key |
| 🚦 Rate Limiting | IP-based throttling on all endpoints |
| 🛡️ XSS Protection | Input validation & output encoding |
| 🗄️ Parameterized SQL | No SQL injection vectors |
| 📜 Restricted CSP | Content Security Policy on extension |
| 🔐 Minimal Permissions | Chrome extension least-privilege |
| 💾 Bounded Cache | 10K entries, SHA-256 keyed |
| 🔒 DB Leak Prevention | try/finally connection management |

---

## 🏗️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python · Flask · SQLite · Turso (cloud DB) |
| **ML Models** | scikit-learn · ONNX Runtime · XGBoost · PyTorch |
| **Extension** | JavaScript · Chrome Manifest V3 · Service Worker · Content Scripts |
| **Deployment** | Docker · HuggingFace Spaces |
| **Security** | HMAC · Rate Limiting · CSP · XSS Protection · Parameterized SQL |
| **APIs** | VirusTotal · Google Safe Browsing · PhishTank · OpenPhish · URLhaus · 5 more |

---

## 🎮 Security Score & Gamification

PhishGuard gamifies security behaviour — your score increases as you scan URLs and block threats:

| Level | Points |
|---|---|
| 🥉 Rookie | 0 pts |
| 🥈 Agent | 100 pts |
| 🥇 Hunter | 500 pts |
| 💎 Guardian | 1,000 pts |
| 🔥 Elite | 2,000 pts |
| 👑 Legend | 5,000 pts |

---

## 🔄 Auto-Retrain Engine

The ML models continuously improve from verified user reports:

- Users report URLs as Phishing or Safe directly from the popup
- Reports are queued until thresholds are met (5 combined reports)
- New model must beat **88% accuracy** before replacing the existing one
- Models self-improve over time — PhishGuard gets smarter as more people use it

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Nayanx0013/Phishguard-extension/issues).

---

## 👤 Built By

**Nayan Ghosh**

Built using AI-assisted development — focused on system design, architecture decisions, and integrating all the moving parts into a production-grade security tool.

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

🛡️ **Built for a safer internet — protecting users from phishing attacks, one scan at a time.**

[🔴 Live Demo](https://nayanx0013-phishguard-extension.hf.space/dashboard) · [⭐ Star this repo](https://github.com/Nayanx0013/Phishguard-extension) · [📦 Backend](https://github.com/Nayanx0013/Phishguard-Backend-extension)

</div>
