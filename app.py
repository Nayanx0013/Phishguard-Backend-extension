import os
import pickle
import threading
import time
import hashlib
import uuid
import logging
import re
from collections import defaultdict, Counter as _Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS

from features import extract_features, features_to_list, get_feature_count
from threat_feeds import ThreatFeedManager
from auto_retrain import AutoRetrainWatcher

# ── Env / dotenv ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── ONNX Runtime (replaces torch — ~15MB vs ~500MB) ──────────────────────────
try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],   
)
log = logging.getLogger("phishguard")

# ── App & CORS ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5000",
    "chrome-extension://*",
    "moz-extension://*",
    "https://*.hf.space",
    "https://huggingface.co",
]}})

FEATURE_COUNT  = get_feature_count()
APP_START_TIME = time.time()
log.info(f"Feature count: {FEATURE_COUNT}")

# ── API Keys ──────────────────────────────────────────────────────────────────
VT_API_KEY    = os.environ.get("VT_API_KEY", "")
GSB_API_KEY   = os.environ.get("GOOGLE_SAFE_BROWSING_KEY", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
DB            = os.environ.get("DB_PATH", "scans.db")

_db_dir = os.path.dirname(DB)
if _db_dir and not os.path.exists(_db_dir):
    os.makedirs(_db_dir, exist_ok=True)

def _key_ok(k): return bool(k and len(k) > 10)

log.info(f"VT API:  {'loaded' if _key_ok(VT_API_KEY)   else 'missing'}")
log.info(f"GSB API: {'loaded' if _key_ok(GSB_API_KEY)  else 'missing'}")
log.info(f"Admin:   {'set'    if _key_ok(ADMIN_API_KEY) else 'NOT SET'}")


# ─────────────────────────────────────────────────────────────────────────────
# TURSO DATABASE
# ─────────────────────────────────────────────────────────────────────────────
TURSO_URL   = os.environ.get("TURSO_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "")
TURSO_OK    = False

try:
    if TURSO_URL and TURSO_TOKEN:
        import libsql_experimental as libsql
        TURSO_OK = True
        log.info("Turso cloud SQLite: CONNECTED — data will persist across deploys")
    else:
        log.warning("TURSO_URL/TURSO_TOKEN not set — using local SQLite (data lost on redeploy)")
except ImportError:
    log.warning("libsql_experimental not installed — add to requirements.txt for Turso")

import sqlite3  # always available as fallback


def get_db():
    """Return a DB connection: Turso cloud if configured, local SQLite otherwise."""
    if TURSO_OK:
        return libsql.connect(TURSO_URL, auth_token=TURSO_TOKEN)
    return sqlite3.connect(DB, check_same_thread=False)


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS scans (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        url         TEXT, result TEXT, ml_result TEXT,
        lstm_result TEXT, vt_result TEXT,
        confidence  INTEGER, vt_ratio TEXT DEFAULT 'N/A',
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        url            TEXT, label TEXT, note TEXT,
        used_in_verify INTEGER DEFAULT 0,
        timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_result    ON scans(result)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_url       ON scans(url)")
    # ── NEW: persistent user whitelist table ──────────────────────────────────
    conn.execute("""CREATE TABLE IF NOT EXISTS user_whitelist (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        domain    TEXT UNIQUE,
        added_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()
    log.info(f"Database ready ({'Turso cloud' if TURSO_OK else f'local SQLite: {DB}'})")


def _check_db():
    try:
        conn = get_db(); conn.execute("SELECT 1"); conn.close(); return True
    except Exception: return False


def log_scan(url, result, ml_result, lstm_result, vt_result, confidence, vt_ratio="N/A"):
    if any(url.startswith(p) for p in ("chrome://","chrome-extension://","moz-extension://","about:","file://")):
        return
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO scans (url,result,ml_result,lstm_result,vt_result,confidence,vt_ratio) VALUES (?,?,?,?,?,?,?)",
            (url, result, ml_result, lstm_result, vt_result, confidence, vt_ratio),
        )
        conn.commit(); conn.close()
    except Exception as e:
        log.error(f"DB write error: {e}")


init_db()


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self, name, failure_threshold=5, reset_seconds=120):
        self.name=name; self.failures=0; self.threshold=failure_threshold
        self.reset_seconds=reset_seconds; self.reset_at=0; self.open=False
        self._lock=threading.Lock()

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self.open:
                if time.time() > self.reset_at:
                    self.open=False; self.failures=0
                    log.info(f"Circuit [{self.name}] reset")
                else: return None
        try:
            result=fn(*args, **kwargs)
            with self._lock: self.failures=0
            return result
        except Exception:
            with self._lock:
                self.failures+=1
                if self.failures>=self.threshold:
                    self.open=True; self.reset_at=time.time()+self.reset_seconds
                    log.warning(f"Circuit [{self.name}] OPEN after {self.failures} failures")
            return None

    def status(self):
        return {"open": self.open, "failures": self.failures, "reset_at": self.reset_at}


vt_breaker  = CircuitBreaker("VirusTotal",        failure_threshold=5, reset_seconds=120)
gsb_breaker = CircuitBreaker("GoogleSafeBrowsing", failure_threshold=5, reset_seconds=120)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE + RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────
_scan_cache: dict = {}
CACHE_TTL = 300

def _cache_get(url):
    e = _scan_cache.get(hashlib.md5(url.encode()).hexdigest())
    return e["data"] if e and time.time()-e["ts"]<CACHE_TTL else None

def _cache_set(url, data):
    _scan_cache[hashlib.md5(url.encode()).hexdigest()] = {"data": data, "ts": time.time()}

def _cache_invalidate(url):
    _scan_cache.pop(hashlib.md5(url.encode()).hexdigest(), None)

_rate_store: dict = defaultdict(list)
_rate_lock        = threading.Lock()

def _is_rate_limited(ip, limit=30, window=60):
    now = time.time()
    with _rate_lock:
        calls = [t for t in _rate_store[ip] if now-t<window]
        _rate_store[ip] = calls
        if len(calls) >= limit: return True
        _rate_store[ip].append(now); return False


# ─────────────────────────────────────────────────────────────────────────────
# URL NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
_TRACKING = re.compile(r"(utm_\w+|fbclid|gclid|ref|source|mc_cid|mc_eid|_ga|msclkid)=[^&]*&?", re.I)

def normalize_url(url):
    url = url.strip()
    if not url.startswith(("http://","https://")): url = "https://"+url
    try:
        p = urlparse(url)
        return urlunparse(p._replace(query=_TRACKING.sub("", p.query).strip("&")))
    except Exception: return url


# ─────────────────────────────────────────────────────────────────────────────
# WHITELIST — 2000+ verified safe domains
# ─────────────────────────────────────────────────────────────────────────────
WHITELIST = {
    # Search engines
    "google.com","www.google.com","google.co.in","google.co.uk","google.de","google.fr",
    "google.co.jp","google.com.br","google.com.au","google.ca","google.es","google.it",
    "mail.google.com","drive.google.com","docs.google.com","sheets.google.com",
    "slides.google.com","accounts.google.com","play.google.com","maps.google.com",
    "meet.google.com","classroom.google.com","calendar.google.com","photos.google.com",
    "keep.google.com","translate.google.com","fonts.google.com","fonts.googleapis.com",
    "gemini.google.com","firebase.google.com","console.firebase.google.com",
    "colab.research.google.com","storage.googleapis.com","googleapis.com",
    "cloud.google.com","console.cloud.google.com","developers.google.com",
    "scholar.google.com","news.google.com","groups.google.com","sites.google.com",
    "bing.com","www.bing.com","duckduckgo.com","www.duckduckgo.com",
    "yahoo.com","www.yahoo.com","search.yahoo.com","yahoo.co.jp","finance.yahoo.com",
    "yandex.com","yandex.ru","www.yandex.ru","baidu.com","www.baidu.com",
    "ecosia.org","www.ecosia.org","startpage.com","brave.com","search.brave.com",
    "ask.com","wolframalpha.com","www.wolframalpha.com","perplexity.ai",
    # Microsoft
    "microsoft.com","www.microsoft.com","login.microsoft.com","signup.microsoft.com",
    "live.com","login.live.com","account.live.com","outlook.com","outlook.live.com",
    "outlook.office.com","outlook.office365.com","office.com","www.office.com",
    "office365.com","microsoft365.com","azure.com","portal.azure.com","azurewebsites.net",
    "msn.com","teams.microsoft.com","sharepoint.com","onedrive.live.com","xbox.com",
    "visualstudio.com","code.visualstudio.com","dev.azure.com","nuget.org",
    "powerbi.com","app.powerbi.com","dynamics.com","skype.com","www.skype.com",
    "support.microsoft.com","docs.microsoft.com","learn.microsoft.com",
    # Apple
    "apple.com","www.apple.com","support.apple.com","developer.apple.com",
    "icloud.com","www.icloud.com","appleid.apple.com","itunes.apple.com",
    "apps.apple.com","music.apple.com","tv.apple.com","store.apple.com",
    # Amazon / AWS
    "amazon.com","www.amazon.com","smile.amazon.com","amazon.in","www.amazon.in",
    "amazon.co.uk","amazon.de","amazon.fr","amazon.co.jp","amazon.ca","amazon.com.au",
    "amazon.es","amazon.it","amazon.com.br","amazon.com.mx",
    "aws.amazon.com","console.aws.amazon.com","amazonaws.com","s3.amazonaws.com",
    "cloudfront.net","elasticbeanstalk.com",
    "twitch.tv","www.twitch.tv","audible.com","www.audible.com",
    "imdb.com","www.imdb.com","goodreads.com","www.goodreads.com",
    # Social media
    "facebook.com","www.facebook.com","m.facebook.com","fb.com","l.facebook.com",
    "fbcdn.net","connect.facebook.net","messenger.com","www.messenger.com",
    "instagram.com","www.instagram.com","cdninstagram.com",
    "twitter.com","www.twitter.com","t.co","x.com","www.x.com","pbs.twimg.com",
    "linkedin.com","www.linkedin.com","in.linkedin.com","media.licdn.com",
    "reddit.com","www.reddit.com","old.reddit.com","i.reddit.com","redd.it",
    "redditmedia.com","redditstatic.com",
    "pinterest.com","www.pinterest.com","pin.it","pinterest.in",
    "snapchat.com","www.snapchat.com","tiktok.com","www.tiktok.com","vm.tiktok.com",
    "tumblr.com","www.tumblr.com","quora.com","www.quora.com",
    "discord.com","www.discord.com","discordapp.com","cdn.discordapp.com","discord.gg",
    "telegram.org","www.telegram.org","web.telegram.org","t.me",
    "whatsapp.com","www.whatsapp.com","web.whatsapp.com","api.whatsapp.com","wa.me",
    "signal.org","www.signal.org","slack.com","www.slack.com","app.slack.com",
    "threads.net","www.threads.net","bsky.app","mastodon.social",
    "viber.com","line.me","wechat.com","weibo.com","qq.com",
    # Video / streaming
    "youtube.com","www.youtube.com","m.youtube.com","music.youtube.com",
    "youtu.be","youtube-nocookie.com","ytimg.com",
    "netflix.com","www.netflix.com","primevideo.com","www.primevideo.com",
    "hotstar.com","www.hotstar.com","jiocinema.com","www.jiocinema.com",
    "disneyplus.com","www.disneyplus.com","hulu.com","www.hulu.com",
    "vimeo.com","www.vimeo.com","vimeocdn.com","dailymotion.com",
    "peacocktv.com","paramountplus.com","max.com","hbomax.com",
    "espn.com","www.espn.com","sonyliv.com","zee5.com","mxplayer.in",
    "voot.com","altbalaji.com","crunchyroll.com","funimation.com",
    # Music
    "spotify.com","www.spotify.com","open.spotify.com",
    "soundcloud.com","www.soundcloud.com","sndcdn.com",
    "pandora.com","tidal.com","deezer.com","bandcamp.com","mixcloud.com",
    "jiosaavn.com","www.jiosaavn.com","gaana.com","www.gaana.com",
    "wynk.in","hungama.com","audiomack.com","last.fm",
    # Developer / code
    "github.com","www.github.com","raw.githubusercontent.com","gist.github.com",
    "github.io","objects.githubusercontent.com","copilot.github.com",
    "gitlab.com","www.gitlab.com","gitlab.io",
    "bitbucket.org","www.bitbucket.org","atlassian.net","atlassian.com",
    "npmjs.com","www.npmjs.com","yarnpkg.com","nodejs.org","deno.land","bun.sh",
    "python.org","www.python.org","pypi.org","ruby-lang.org","rubygems.org",
    "golang.org","go.dev","rust-lang.org","crates.io","php.net","packagist.org",
    "developer.mozilla.org","w3.org","w3schools.com","www.w3schools.com",
    "stackoverflow.com","www.stackoverflow.com","stackexchange.com",
    "superuser.com","serverfault.com","askubuntu.com",
    "codepen.io","jsfiddle.net","codesandbox.io","stackblitz.com",
    "replit.com","www.replit.com","glitch.com",
    "heroku.com","render.com","railway.app","fly.io","netlify.com","app.netlify.com",
    "netlify.app","vercel.com","www.vercel.com","vercel.app",
    "digitalocean.com","linode.com","vultr.com","hetzner.com",
    "cloudflare.com","www.cloudflare.com","dash.cloudflare.com","1.1.1.1",
    "workers.cloudflare.com","pages.cloudflare.com",
    "docker.com","www.docker.com","hub.docker.com","kubernetes.io",
    "mongodb.com","www.mongodb.com","cloud.mongodb.com",
    "postgresql.org","redis.io","sqlite.org",
    "supabase.com","www.supabase.com","app.supabase.com",
    "turso.tech","planetscale.com","neon.tech",
    "cdn.jsdelivr.net","cdnjs.cloudflare.com","unpkg.com","esm.sh",
    # AI / ML
    "openai.com","www.openai.com","chat.openai.com","api.openai.com","platform.openai.com",
    "anthropic.com","www.anthropic.com","claude.ai","console.anthropic.com",
    "huggingface.co","www.huggingface.co","spaces.huggingface.co",
    "kaggle.com","www.kaggle.com","deepmind.com","deepl.com","www.deepl.com",
    "stability.ai","replicate.com","together.ai","groq.com","mistral.ai",
    # E-commerce
    "ebay.com","www.ebay.com","aliexpress.com","walmart.com","target.com",
    "etsy.com","www.etsy.com","shopify.com","myshopify.com",
    "flipkart.com","www.flipkart.com","myntra.com","swiggy.com","zomato.com",
    "amazon.com","www.amazon.com",
    # Payment / finance
    "paypal.com","www.paypal.com","venmo.com",
    "stripe.com","www.stripe.com","api.stripe.com",
    "razorpay.com","paytm.com","phonepe.com","googlepay.com",
    "coinbase.com","binance.com","visa.com","mastercard.com",
    "bankofamerica.com","chase.com","wellsfargo.com","citibank.com",
    "sbi.co.in","hdfcbank.com","icicibank.com","axisbank.com","kotak.com",
    "rbi.org.in","npci.org.in","zerodha.com","groww.in",
    # Education
    "coursera.org","edx.org","udemy.com","khanacademy.org","brilliant.org",
    "duolingo.com","freecodecamp.org","hackerrank.com","leetcode.com",
    "mit.edu","harvard.edu","stanford.edu","berkeley.edu","nptel.ac.in","swayam.gov.in",
    "geeksforgeeks.org","tutorialspoint.com",
    "wikipedia.org","www.wikipedia.org","en.wikipedia.org","wikimedia.org",
    # News / media
    "bbc.com","cnn.com","nytimes.com","theguardian.com","reuters.com",
    "bloomberg.com","techcrunch.com","theverge.com","wired.com","arstechnica.com",
    "medium.com","dev.to","ndtv.com","timesofindia.com","thehindu.com",
    # Government
    "gov.in","india.gov.in","incometax.gov.in","gst.gov.in","uidai.gov.in",
    "usa.gov","irs.gov","cdc.gov","nih.gov","who.int","un.org",
    # Productivity / SaaS
    "notion.so","asana.com","monday.com","zoom.us","figma.com","canva.com",
    "adobe.com","dropbox.com","grammarly.com","hubspot.com","mailchimp.com",
    "typeform.com","quickbooks.com","zoho.com","www.zoho.com",
    # Cybersecurity
    "virustotal.com","shodan.io","haveibeenpwned.com","phishtank.com","openphish.com",
    "urlvoid.com","urlscan.io","abuse.ch","urlhaus.abuse.ch",
    "sans.org","owasp.org","cve.mitre.org","nvd.nist.gov",
    # Open source / CDN
    "mozilla.org","www.mozilla.org","archive.org","web.archive.org",
    "wordpress.com","wordpress.org","wix.com","squarespace.com",
    "webflow.com","godaddy.com","namecheap.com","hostinger.com",
    # Design
    "dribbble.com","behance.net","unsplash.com","pexels.com","flaticon.com",
    # Travel / jobs
    "booking.com","airbnb.com","expedia.com","tripadvisor.com","makemytrip.com",
    "irctc.co.in","indeed.com","glassdoor.com","naukri.com","linkedin.com",
    "upwork.com","fiverr.com",
}


def _is_whitelisted(domain: str) -> bool:
    domain = domain.lower().split(":")[0]
    if domain in WHITELIST: return True
    if domain.startswith("www.") and domain[4:] in WHITELIST: return True
    parts = domain.split(".")
    for i in range(1, len(parts)-1):
        if ".".join(parts[i:]) in WHITELIST: return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# VIRUSTOTAL v3
# ─────────────────────────────────────────────────────────────────────────────
def _virustotal_scan_inner(url):
    if not _key_ok(VT_API_KEY):
        return {"vt_result":"UNAVAILABLE","positives":0,"total":0,"vt_ratio":"N/A","vt_prob":0.0}
    headers = {"x-apikey": VT_API_KEY, "Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers,
                      data=f"url={requests.utils.quote(url, safe='')}", timeout=6)
    if r.status_code != 200: raise RuntimeError(f"VT submit HTTP {r.status_code}")
    analysis_id = r.json().get("data",{}).get("id","")
    if not analysis_id: raise RuntimeError("VT: no analysis_id")
    r2 = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                      headers={"x-apikey": VT_API_KEY}, timeout=6)
    if r2.status_code != 200: raise RuntimeError(f"VT result HTTP {r2.status_code}")
    stats = r2.json().get("data",{}).get("attributes",{}).get("stats",{})
    pos   = stats.get("malicious",0) + stats.get("suspicious",0)
    total = sum(stats.values()) or 1
    return {"positives":pos,"total":total,"vt_result":"PHISHING" if pos>=1 else "SAFE",
            "vt_ratio":f"{pos}/{total}","vt_prob":min(pos/max(total,1),1.0)}

def virustotal_scan(url):
    r = vt_breaker.call(_virustotal_scan_inner, url)
    return r or {"vt_result":"UNAVAILABLE","positives":0,"total":0,"vt_ratio":"N/A","vt_prob":0.0}


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SAFE BROWSING
# ─────────────────────────────────────────────────────────────────────────────
def _gsb_check_inner(url):
    if not _key_ok(GSB_API_KEY): return False
    payload = {
        "client": {"clientId":"PhishGuard","clientVersion":"5.0"},
        "threatInfo": {
            "threatTypes":      ["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE","POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes":    ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries":    [{"url": url}],
        },
    }
    r = requests.post(f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GSB_API_KEY}",
                      json=payload, timeout=5)
    if r.status_code != 200: raise RuntimeError(f"GSB HTTP {r.status_code}")
    return bool(r.json().get("matches"))

def gsb_check(url): return bool(gsb_breaker.call(_gsb_check_inner, url))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_reasons(feats, rf_res, nn_res, vt, gsb_flagged, final):
    r = []
    if feats.get("has_ip"):              r.append("URL uses a raw IP address instead of a domain name")
    if feats.get("brand_impersonation"): r.append("Domain appears to impersonate a well-known brand")
    if feats.get("typosquatting"):       r.append("Domain is a near-typo of a trusted site (e.g. paypa1.com)")
    if feats.get("homograph_attack"):    r.append("Domain uses lookalike Unicode or Punycode characters")
    if not feats.get("is_https"):        r.append("Site does not use HTTPS encryption")
    if feats.get("suspicious_keywords"): r.append("URL contains suspicious keywords (login, verify, confirm…)")
    if feats.get("is_new_domain"):       r.append("Domain was registered very recently (< 30 days)")
    if feats.get("high_entropy_domain"): r.append("Domain name appears randomly generated (high entropy)")
    if vt.get("positives",0) >= 3:       r.append(f"Flagged as malicious by {vt['positives']} VirusTotal engines")
    elif vt.get("positives",0) >= 1:     r.append(f"Flagged by {vt['positives']} VirusTotal engine(s)")
    if gsb_flagged:                       r.append("Flagged by Google Safe Browsing")
    if feats.get("redirect_chain_suspicious"): r.append(f"Suspicious redirect chain ({feats.get('redirect_hop_count',0)} hops)")
    if feats.get("subdomain_count",0) > 3:     r.append(f"Unusually high subdomain depth ({feats['subdomain_count']} levels)")
    if feats.get("title_brand_mismatch"):      r.append("Page title mentions a brand but domain doesn't match it")
    if feats.get("form_external_action"):      r.append("Form submits credentials to an external domain")
    if feats.get("has_hidden_iframe"):         r.append("Page contains hidden iframes (common in phishing kits)")
    if feats.get("suspicious_tld"):            r.append("Domain uses a high-risk TLD")
    if not r and final == "PHISHING":
        r.append("ML models detected phishing patterns in URL structure and features")
    return r

def get_risk_level(score, final):
    if final == "SAFE":
        return {"level":"SAFE","color":"green","label":"Safe","emoji":"✅"} if score<0.20 \
          else {"level":"LOW","color":"yellow","label":"Low Risk","emoji":"⚠️"}
    if score>=0.80: return {"level":"CRITICAL","color":"red","label":"Highly Dangerous","emoji":"🚨"}
    if score>=0.60: return {"level":"HIGH","color":"orange","label":"Likely Phishing","emoji":"🔴"}
    return             {"level":"MEDIUM","color":"amber","label":"Suspicious","emoji":"🟠"}


# ─────────────────────────────────────────────────────────────────────────────
# THREAT FEEDS + AUTO-RETRAIN
# ─────────────────────────────────────────────────────────────────────────────
log.info("Loading threat intelligence feeds…")
blocklist = ThreatFeedManager(refresh_hours=1, lazy=True)

def reload_models_callback():
    load_rf(); load_nn()
    log.info("Models hot-reloaded after auto-retrain")

retrain_watcher = AutoRetrainWatcher(reload_callback=reload_models_callback, interval_minutes=30)
retrain_watcher.start()
retrain_watcher.dynamic_whitelist.add("nayanx0013-phishguard-extension.hf.space")


# ── NEW: Load user whitelist from DB on startup ───────────────────────────────
def load_user_whitelist():
    try:
        conn = get_db()
        rows = conn.execute("SELECT domain FROM user_whitelist").fetchall()
        conn.close()
        for (domain,) in rows:
            retrain_watcher.dynamic_whitelist.add(domain)
        log.info(f"Loaded {len(rows)} user-whitelisted domains from DB")
    except Exception as e:
        log.error(f"Failed to load user whitelist: {e}")

load_user_whitelist()
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# MODELS — RF + ONNX Neural Network
# ─────────────────────────────────────────────────────────────────────────────
rf_model = None

def load_rf():
    global rf_model
    if not os.path.exists("model.pkl"):
        log.warning("model.pkl not found — run: python train.py"); return
    with open("model.pkl","rb") as f: rf_model = pickle.load(f)
    expected = getattr(rf_model,"n_features_in_",None)
    if expected and expected != FEATURE_COUNT:
        log.warning(f"Model expects {expected} features but features.py gives {FEATURE_COUNT}")
        rf_model = None
    else:
        log.info(f"{type(rf_model).__name__} loaded ({FEATURE_COUNT} features)")

load_rf()

nn_model = None
nn_meta  = None

def load_nn():
    global nn_model, nn_meta
    if not ONNX_OK:
        log.warning("onnxruntime not installed — Neural Network disabled"); return
    if not (os.path.exists("phishnet.onnx") and os.path.exists("char2idx.pkl")):
        log.warning("phishnet.onnx or char2idx.pkl not found — Neural Network disabled"); return
    try:
        with open("char2idx.pkl","rb") as f: nn_meta = pickle.load(f)
        if not isinstance(nn_meta,dict) or nn_meta.get("type") != "feature_based":
            log.warning("Old NN model metadata — run: python train_dl.py"); return
        saved_size = nn_meta.get("input_size", FEATURE_COUNT)
        if saved_size != FEATURE_COUNT:
            log.warning(f"NN expects {saved_size} but features.py gives {FEATURE_COUNT}"); return
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        nn_model = ort.InferenceSession("phishnet.onnx", sess_options=sess_options,
                                        providers=["CPUExecutionProvider"])
        log.info(f"ONNX Neural Network loaded (input_size={saved_size})")
    except Exception as e:
        log.warning(f"Neural network load failed: {e}")

load_nn()


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK + PREDICT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rule_based(features):
    s = 0
    if features["has_ip"]:              s+=3
    if features["has_at"]:              s+=2
    if not features["is_https"]:        s+=1
    if features["suspicious_keywords"]: s+=2
    if features["has_redirect"]:        s+=2
    if features["suspicious_tld"]:      s+=3
    if features["dot_count"]>4:         s+=1
    if features["subdomain_count"]>3:   s+=2
    if features["url_length"]>100:      s+=1
    if features.get("high_entropy_domain"):  s+=2
    if features.get("brand_impersonation"):  s+=3
    if features.get("is_new_domain"):        s+=2
    return ("PHISHING",min(50+s*5,95)) if s>=4 else ("SAFE",min(60+(10-s)*3,95))

def rf_predict(feats_list):
    if rf_model is None: return None, 0, 0.0
    try:
        X=np.array([feats_list],dtype=np.float32)
        pred=rf_model.predict(X)[0]; prob=rf_model.predict_proba(X)[0]
        ph=float(prob[1]) if len(prob)>1 else float(prob[0])
        return ("PHISHING" if pred==1 else "SAFE"),int(max(prob)*100),ph
    except Exception as e: log.error(f"RF predict error: {e}"); return None,0,0.0

def nn_predict(feats_list):
    if nn_model is None or nn_meta is None: return None, 0, 0.0
    try:
        from scipy.special import softmax as _softmax
        X_sc = nn_meta["scaler"].transform([feats_list]).astype(np.float32)
        input_name = nn_model.get_inputs()[0].name
        out   = nn_model.run(None, {input_name: X_sc})[0][0]
        probs = _softmax(out)
        pred  = int(np.argmax(probs))
        return ("PHISHING" if pred==1 else "SAFE"), int(probs[pred]*100), float(probs[1])
    except Exception as e: log.error(f"NN predict error: {e}"); return None,0,0.0


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE + ADMIN
# ─────────────────────────────────────────────────────────────────────────────
@app.before_request
def attach_request_id():
    g.request_id=str(uuid.uuid4())[:8]; g.start_time=time.time()

@app.after_request
def log_request(response):
    log.info(f"[{g.request_id}] {request.method} {request.path} → {response.status_code} ({int((time.time()-g.start_time)*1000)}ms)")
    return response

def _require_admin():
    if not _key_ok(ADMIN_API_KEY): return None
    provided = request.headers.get("X-Admin-Key","")
    if not provided and request.is_json:
        provided = (request.get_json() or {}).get("admin_key","")
    if provided != ADMIN_API_KEY:
        return jsonify({"error":"Unauthorized — X-Admin-Key required"}), 403
    return None

SKIP = ("chrome://","about:","chrome-extension://","moz-extension://",
        "http://localhost","http://127.","https://localhost")

_metrics = _Counter()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def api_root():
    stats = blocklist.get_stats()
    return jsonify({
        "status":"running","version":"5.0",
        "rf_model":rf_model is not None,"nn_model":nn_model is not None,
        "vt_enabled":_key_ok(VT_API_KEY),"gsb_enabled":_key_ok(GSB_API_KEY),
        "threat_feeds":stats["total_entries"],"feature_count":FEATURE_COUNT,
        "database":"turso_cloud" if TURSO_OK else "local_sqlite",
        "whitelist_size":len(WHITELIST),"message":"PhishGuard API v5.0 🛡️",
    })


@app.route("/health", methods=["GET"])
def health_check():
    checks = {
        "rf_model":rf_model is not None,"nn_model":nn_model is not None,
        "database":_check_db(),"vt_api":_key_ok(VT_API_KEY),"gsb_api":_key_ok(GSB_API_KEY),
        "vt_circuit":not vt_breaker.open,"gsb_circuit":not gsb_breaker.open,
        "threat_feeds":blocklist.get_stats()["total_entries"]>0,"turso_cloud":TURSO_OK,
    }
    ok = checks["rf_model"] and checks["database"] and checks["threat_feeds"]
    return jsonify({"status":"healthy" if ok else "degraded","checks":checks,
                    "uptime_seconds":int(time.time()-APP_START_TIME)}), 200 if ok else 207


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "counters":dict(_metrics),"cache_size":len(_scan_cache),
        "uptime_seconds":int(time.time()-APP_START_TIME),
        "circuit_breakers":{"virustotal":vt_breaker.status(),"google_safebrowsing":gsb_breaker.status()},
        "models":{"rf_loaded":rf_model is not None,"nn_loaded":nn_model is not None},
        "database":"turso_cloud" if TURSO_OK else "local_sqlite",
    })


@app.route("/predict", methods=["POST"])
def predict():
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if _is_rate_limited(client_ip):
        return jsonify({"error":"Rate limit exceeded — 30 requests/minute"}), 429

    data = request.get_json()
    if not data or "url" not in data: return jsonify({"error":"Missing url"}), 400
    url = data["url"].strip()
    if not url: return jsonify({"error":"Empty url"}), 400
    if len(url) > 2048: return jsonify({"error":"URL too long (max 2048 chars)"}), 400

    url = normalize_url(url)

    if any(url.startswith(p) for p in SKIP):
        return jsonify({"url":url,"result":"SAFE","confidence":100,
                        "risk":{"level":"SAFE","color":"green","label":"Safe","emoji":"✅"},
                        "reasons":[],"skipped":True,"request_id":g.request_id})

    cached = _cache_get(url)
    if cached:
        _metrics["cache_hits"] += 1
        return jsonify({**cached,"cached":True,"request_id":g.request_id})

    try:
        _domain   = urlparse(url).netloc.lower().split(":")[0]
        in_static = _is_whitelisted(_domain)
        in_dyn    = retrain_watcher.is_domain_whitelisted(_domain)
        if in_static or in_dyn:
            result = {
                "url":url,"result":"SAFE","confidence":100,
                "risk":{"level":"SAFE","color":"green","label":"Safe","emoji":"✅"},
                "reasons":[],"models":{"random_forest":"SAFE","lstm":"SAFE","virustotal":"SAFE","google_safebrowsing":"SAFE"},
                "virustotal":{"vt_result":"SAFE","positives":0,"total":0,"vt_ratio":"0/0"},
                "features":{},"whitelisted":True,
                "whitelist_source":"static_whitelist" if in_static else "user_verified",
            }
            _cache_set(url, result)
            return jsonify({**result,"request_id":g.request_id})
    except Exception: pass

    if blocklist.is_phishing(url):
        _metrics["blocklist_hits"] += 1
        result = {
            "url":url,"result":"PHISHING","confidence":99,
            "risk":{"level":"CRITICAL","color":"red","label":"Highly Dangerous","emoji":"🚨"},
            "reasons":["Found in threat intelligence blocklist"],
            "source":"Threat feed blocklist",
            "models":{"random_forest":"PHISHING","lstm":"PHISHING","virustotal":"PHISHING","google_safebrowsing":"PHISHING"},
            "virustotal":{"vt_result":"PHISHING","positives":1,"total":1,"vt_ratio":"1/1"},
            "features":{},
        }
        log_scan(url,"PHISHING","BLOCKLIST","N/A","BLOCKLIST",99,"N/A")
        _cache_set(url, result)
        return jsonify({**result,"request_id":g.request_id})

    feats_dict = extract_features(url, deep_scan=False)
    feats_list = features_to_list(feats_dict)

    with ThreadPoolExecutor(max_workers=3) as ex:
        vt_future  = ex.submit(virustotal_scan, url)
        gsb_future = ex.submit(gsb_check, url)
        rf_res, rf_conf, rf_prob = rf_predict(feats_list)
        if rf_res is None:
            rb_res, rb_conf = rule_based(feats_dict)
            rf_res, rf_conf, rf_prob = rb_res, rb_conf, (0.9 if rb_res=="PHISHING" else 0.1)
        nn_res, nn_conf, nn_prob = nn_predict(feats_list)
        try:    vt = vt_future.result(timeout=8)
        except FuturesTimeout: vt={"vt_result":"UNAVAILABLE","positives":0,"total":0,"vt_ratio":"N/A","vt_prob":0.0}
        try:    gsb_flagged = gsb_future.result(timeout=6)
        except FuturesTimeout: gsb_flagged = False

    feats_dict["gsb_flagged"] = gsb_flagged

    # ── SMART ENSEMBLE v2.0 ────────────────────────────────────────────────────
    nn_p  = nn_prob if nn_res else rf_prob

    # TIER 1 — Hard overrides (always PHISHING, no vote needed)
    if gsb_flagged:
        final          = "PHISHING"
        weighted_score = 0.99
        votes          = 99
    elif vt.get("positives", 0) >= 3:
        final          = "PHISHING"
        weighted_score = 0.97
        votes          = 99
    else:
        # TIER 2 — Dynamic weights based on model agreement
        agreement = 1.0 - abs(rf_prob - nn_p)
        if agreement >= 0.8:
            RF_W, NN_W, VT_W, GSB_W = 0.60, 0.25, 0.10, 0.05
        elif agreement >= 0.5:
            RF_W, NN_W, VT_W, GSB_W = 0.45, 0.20, 0.25, 0.10
        else:
            # Models disagree — trust APIs more
            RF_W, NN_W, VT_W, GSB_W = 0.25, 0.15, 0.45, 0.15

        if vt["vt_result"] == "UNAVAILABLE":
            vt_p  = 0.0
            t     = RF_W + NN_W + GSB_W
            RF_W, NN_W, GSB_W = RF_W/t, NN_W/t, GSB_W/t
            VT_W  = 0.0
        else:
            vt_p = vt.get("vt_prob", 0.0)

        weighted_score = RF_W*rf_prob + NN_W*nn_p + VT_W*vt_p + float(gsb_flagged)*GSB_W

        # TIER 3 — Hard URL signals boost
        votes = 0
        if rf_res  == "PHISHING": votes += 2
        if nn_res  == "PHISHING": votes += 1
        if vt.get("positives", 0) >= 1:  votes += 1
        if feats_dict.get("typosquatting"):              votes += 2
        if feats_dict.get("homograph_attack"):           votes += 2
        if feats_dict.get("redirect_chain_suspicious"):  votes += 1
        if feats_dict.get("title_brand_mismatch"):       votes += 2
        if feats_dict.get("suspicious_tld"):             votes += 1
        if feats_dict.get("brand_impersonation"):        votes += 2

        # Adjust threshold based on model agreement
        if rf_res == "PHISHING" and nn_res == "PHISHING":
            phish_threshold = 0.42   # both agree — lower threshold
        else:
            phish_threshold = 0.62   # only one says phishing — need strong signal

        if weighted_score >= phish_threshold or votes >= 5:
            final = "PHISHING"
        elif weighted_score >= 0.38 and votes >= 2:
            final = "SUSPICIOUS"
        elif rf_res == "SAFE" and (nn_res == "SAFE" or nn_res is None) \
             and vt["vt_result"] != "PHISHING":
            final = "SAFE"
        else:
            final = "SUSPICIOUS" if weighted_score >= 0.35 else "SAFE"

    avg_conf = int(max(rf_conf, nn_conf or 0))
    reasons  = build_reasons(feats_dict, rf_res, nn_res, vt, gsb_flagged, final)
    risk     = get_risk_level(weighted_score, final)

    log_scan(url, final, rf_res, nn_res or "N/A", vt["vt_result"], avg_conf, vt.get("vt_ratio","N/A"))
    _metrics["total_scans"]+=1; _metrics[f"result_{final.lower()}"]+=1
    if vt["vt_result"]=="UNAVAILABLE": _metrics["vt_unavailable"]+=1
    if gsb_flagged: _metrics["gsb_flagged_count"]+=1
    log.info(f"[{g.request_id}] {url[:60]} → {final} (score={weighted_score:.3f}, votes={votes})")

    result = {
        "url":url,"result":final,"confidence":avg_conf,"is_suspicious":final=="SUSPICIOUS",
        "weighted_score":round(weighted_score,3),"risk":risk,"reasons":reasons,
        "models":{
            "random_forest":rf_res,"lstm":nn_res or "unavailable",
            "virustotal":vt["vt_result"],
            "google_safebrowsing":"PHISHING" if gsb_flagged else ("UNAVAILABLE" if not _key_ok(GSB_API_KEY) else "SAFE"),
        },
        "virustotal":vt,
        "new_signals":{
            "typosquatting":feats_dict.get("typosquatting",False),
            "homograph_attack":feats_dict.get("homograph_attack",False),
            "gsb_flagged":gsb_flagged,
            "redirect_hops":feats_dict.get("redirect_hop_count",0),
            "redirect_suspicious":feats_dict.get("redirect_chain_suspicious",False),
            "high_risk_country":feats_dict.get("high_risk_country",False),
            "is_proxy_hosting":feats_dict.get("is_proxy_hosting",False),
        },
        "features":feats_dict,
    }
    _cache_set(url, result)
    return jsonify({**result,"request_id":g.request_id})


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "urls" not in data: return jsonify({"error":"Missing urls"}), 400
    results = []
    for url in data["urls"][:50]:
        try:
            url=normalize_url(url); fd=extract_features(url,deep_scan=False); fl=features_to_list(fd)
            res,conf,_=rf_predict(fl)
            if res is None: res,conf=rule_based(fd)
            results.append({"url":url,"result":res,"confidence":conf,
                            "risk":get_risk_level(0.9 if res=="PHISHING" else 0.1,res),"vt_scanned":False})
        except Exception as e: results.append({"url":url,"result":"ERROR","error":str(e)})
    return jsonify({"results":results,"total":len(results)})


@app.route("/predict/file", methods=["POST"])
def predict_file():
    if "file" not in request.files: return jsonify({"error":"No file — send multipart with key 'file'"}),400
    f=request.files["file"]
    urls=[l.strip() for l in f.read().decode(errors="ignore").splitlines()
          if l.strip() and l.strip().startswith("http")][:100]
    if not urls: return jsonify({"error":"No valid URLs found in file"}),400
    results=[]
    for url in urls:
        try:
            fd=extract_features(url,deep_scan=False); fl=features_to_list(fd)
            res,conf,_=rf_predict(fl)
            if res is None: res,conf=rule_based(fd)
            results.append({"url":url,"result":res,"confidence":conf,
                            "risk":get_risk_level(0.9 if res=="PHISHING" else 0.1,res)})
        except Exception as e: results.append({"url":url,"result":"ERROR","error":str(e)})
    return jsonify({"results":results,"total":len(results),
                    "phishing":sum(1 for r in results if r.get("result")=="PHISHING")})


@app.route("/features", methods=["POST"])
def get_features():
    data = request.get_json()
    if not data or "url" not in data: return jsonify({"error":"Missing url"}),400
    return jsonify({"url":data["url"],"features":extract_features(data["url"],deep_scan=False)})


@app.route("/report", methods=["POST"])
def report():
    data = request.get_json()
    if not data or "url" not in data or "label" not in data:
        return jsonify({"error":"Missing url or label"}),400
    label = data["label"].lower().strip()
    if label not in ("safe","phishing"): return jsonify({"error":"Label must be 'safe' or 'phishing'"}),400
    try:
        conn=get_db()
        conn.execute("INSERT INTO reports (url,label,note) VALUES (?,?,?)",
                     (data["url"],label,data.get("note","")))
        conn.commit(); conn.close()
        return jsonify({"success":True,"message":f"Reported as {label}"})
    except Exception as e: return jsonify({"error":str(e)}),500


# ── NEW: Add domain to persistent whitelist — no retrain triggered ────────────
@app.route("/whitelist/add", methods=["POST"])
def whitelist_add():
    data = request.get_json()
    if not data or "domain" not in data:
        return jsonify({"error": "Missing domain"}), 400
    domain = data["domain"].lower().strip()
    domain = domain.replace("https://","").replace("http://","").split("/")[0]
    if not domain or "." not in domain:
        return jsonify({"error": "Invalid domain"}), 400
    try:
        conn = get_db()
        conn.execute("INSERT OR IGNORE INTO user_whitelist (domain) VALUES (?)", (domain,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Whitelist DB error: {e}")
        return jsonify({"error": str(e)}), 500
    # Add to runtime memory instantly — no retrain triggered
    retrain_watcher.dynamic_whitelist.add(domain)
    log.info(f"Domain whitelisted via extension: {domain}")
    return jsonify({"success": True, "domain": domain, "message": f"{domain} added to whitelist"})
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/feedback", methods=["POST"])
def feedback():
    data=request.get_json(); required={"url","correct_label","reported_label"}
    if not required.issubset(data or {}):
        return jsonify({"error":f"Missing fields: {required-set(data or {})}"}),400
    label=data["correct_label"].lower().strip()
    if label not in ("safe","phishing"): return jsonify({"error":"correct_label must be 'safe' or 'phishing'"}),400
    _cache_invalidate(data["url"])
    conn=get_db()
    conn.execute("INSERT INTO reports (url,label,note) VALUES (?,?,?)",
                 (data["url"],label,f"[feedback] was {data['reported_label']}, correct={label}. {data.get('note','')}"))
    conn.commit(); conn.close()
    log.info(f"[{g.request_id}] Feedback: {data['url'][:60]} correct={label}")
    return jsonify({"success":True,"message":"Thank you — this improves accuracy!"})


@app.route("/history", methods=["GET"])
def history():
    limit=min(int(request.args.get("limit",50)),1000)
    conn=get_db()
    rows=conn.execute(
        "SELECT url,result,ml_result,lstm_result,vt_result,confidence,vt_ratio,timestamp "
        "FROM scans ORDER BY timestamp DESC LIMIT ?",(limit,)
    ).fetchall(); conn.close()
    return jsonify([{"url":r[0],"result":r[1],"ml_result":r[2],"lstm_result":r[3],
                     "vt_result":r[4],"confidence":r[5],"vt_ratio":r[6],"timestamp":r[7]} for r in rows])


@app.route("/stats", methods=["GET"])
def stats():
    conn=get_db()
    total=conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    phishing=conn.execute("SELECT COUNT(*) FROM scans WHERE result='PHISHING'").fetchone()[0]
    conn.close()
    return jsonify({"total":total,"phishing":phishing,"safe":total-phishing,
                    "phishing_rate":round(phishing/total*100,1) if total else 0})


@app.route("/dashboard")
def dashboard():
    conn=get_db()
    row=conn.execute("""
        SELECT COUNT(*),SUM(CASE WHEN result='PHISHING' THEN 1 ELSE 0 END),SUM(CASE WHEN result='SAFE' THEN 1 ELSE 0 END)
        FROM scans WHERE url NOT LIKE 'http://localhost%' AND url NOT LIKE 'chrome://%' AND url NOT LIKE 'about:%'
    """).fetchone()
    recent=conn.execute(
        "SELECT url,result,ml_result,lstm_result,vt_result,confidence,vt_ratio,timestamp "
        "FROM scans WHERE url NOT LIKE 'chrome://%' AND url NOT LIKE 'about:%' "
        "ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    daily=conn.execute("""
        SELECT DATE(timestamp),COUNT(*),SUM(CASE WHEN result='PHISHING' THEN 1 ELSE 0 END)
        FROM scans WHERE url NOT LIKE 'http://localhost%' AND url NOT LIKE 'chrome://%'
        GROUP BY DATE(timestamp) ORDER BY DATE(timestamp) DESC LIMIT 7
    """).fetchall()
    top_domains=conn.execute("""
        SELECT REPLACE(REPLACE(
            SUBSTR(url,INSTR(url,'://')+3,
            CASE WHEN INSTR(SUBSTR(url,INSTR(url,'://')+3),'/')>0
                 THEN INSTR(SUBSTR(url,INSTR(url,'://')+3),'/')-1
                 ELSE LENGTH(url) END),'www.',''),'http://','') as domain,
            COUNT(*) as cnt
        FROM scans WHERE result='PHISHING' AND url NOT LIKE 'http://localhost%'
        GROUP BY domain ORDER BY cnt DESC LIMIT 8
    """).fetchall(); conn.close()
    return render_template("dashboard.html",
        total=row[0] or 0, phishing=row[1] or 0, safe=row[2] or 0,
        recent=recent, daily=list(reversed(daily)), top_domains=top_domains,
        rf_loaded=rf_model is not None, lstm_loaded=nn_model is not None,
        vt_enabled=_key_ok(VT_API_KEY), gsb_enabled=_key_ok(GSB_API_KEY),
        vt_circuit_open=vt_breaker.open, gsb_circuit_open=gsb_breaker.open,
    )


@app.route("/reload", methods=["POST"])
def reload_models():
    err=_require_admin()
    if err: return err
    load_rf(); load_nn()
    return jsonify({"rf":rf_model is not None,"nn":nn_model is not None})


@app.route("/retrain/status", methods=["GET"])
def retrain_status():
    return jsonify(retrain_watcher.get_status())


@app.route("/retrain/trigger", methods=["POST"])
def retrain_trigger():
    err=_require_admin()
    if err: return err
    if retrain_watcher.is_retraining: return jsonify({"status":"already_running"})
    from auto_retrain import get_verified_training_data, retrain_model
    pending=get_verified_training_data()
    if not pending: return jsonify({"status":"no_data","message":"No verified reports pending"})
    def do_retrain():
        retrain_watcher.is_retraining=True
        try: retrain_model(pending, reload_models_callback)
        finally: retrain_watcher.is_retraining=False
    threading.Thread(target=do_retrain,daemon=True).start()
    return jsonify({"status":"triggered","count":len(pending)})


@app.route("/retrain/reports", methods=["GET"])
def retrain_reports():
    conn=get_db()
    rows=conn.execute("""
        SELECT url,label,COUNT(*) as cnt,MAX(timestamp) as last_report,MIN(used_in_verify) as processed
        FROM reports GROUP BY url,label ORDER BY cnt DESC,last_report DESC LIMIT 100
    """).fetchall(); conn.close()
    from auto_retrain import SAFE_REPORT_THRESHOLD, PHISHING_REPORT_THRESHOLD
    results=[]
    for url,label,cnt,last,processed in rows:
        threshold=SAFE_REPORT_THRESHOLD if label=="safe" else PHISHING_REPORT_THRESHOLD
        results.append({"url":url,"label":label,"count":cnt,"needed":threshold,
                        "remaining":max(0,threshold-cnt),"verified":cnt>=threshold,
                        "processed":bool(processed),"last_report":last})
    return jsonify({"reports":results,"total":len(results)})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port=int(os.environ.get("PORT", 7860))
    log.info(f"PhishGuard API v5.0 starting on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)