

import re
import math
import os
import socket
import ssl
import requests
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime

try:
    import whois
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

try:
    import dns.resolver
    DNS_OK = True
except ImportError:
    DNS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# BRAND / TYPOSQUATTING DATA
# ─────────────────────────────────────────────────────────────────────────────
BRANDS = [
    "paypal", "amazon", "google", "microsoft", "apple", "facebook", "netflix",
    "instagram", "twitter", "linkedin", "bankofamerica", "chase", "wellsfargo",
    "citibank", "hsbc", "barclays", "dropbox", "adobe", "steam", "ebay",
    "alibaba", "yahoo", "outlook", "office365", "coinbase", "binance",
    "blockchain", "metamask", "dhl", "fedex", "ups", "usps", "royalmail",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".click", ".link", ".live", ".online", ".site", ".website", ".info", ".biz",
    ".cfd", ".cyou", ".icu", ".sbs", ".vip", ".bar", ".pw", ".rest", ".fin",
    ".bond", ".hair", ".monster", ".gdn", ".bid", ".loan", ".locker", ".xin",
    ".work", ".party", ".racing", ".win", ".download", ".stream", ".trade",
    ".accountant", ".cricket", ".science", ".review", ".country", ".faith",
    ".date", ".men", ".zip", ".mov", ".bot",
    ".ru", ".cn", ".su", ".to", ".cc", ".ws",
    ".li", ".digital", ".network", ".center", ".space", ".store", ".shop",
    ".tech", ".media", ".world", ".today", ".support", ".solutions",
    ".services", ".systems", ".cloud", ".pro", ".uno",
    ".buzz", ".vin", ".surf", ".run", ".fun", ".life", ".wtf", ".fail",
    ".ninja", ".rocks", ".guru", ".global", ".group", ".email", ".chat",
    ".money", ".cash", ".tax", ".investments", ".capital", ".insurance",
    ".claims", ".lawyer", ".legal", ".help", ".care", ".health", ".clinic",
    ".diet", ".fit", ".surgery", ".university", ".college", ".degree",
    ".study", ".how", ".now", ".new", ".free", ".gift", ".promo", ".discount",
    ".sale", ".deals", ".bargains", ".cheap", ".express", ".delivery",
    ".shipping", ".tracking", ".post", ".mail", ".secure", ".safe",
    ".protect", ".verify", ".validation", ".update", ".upgrade",
    ".login", ".signin", ".account", ".billing", ".payment", ".invoice",
    ".bank", ".credit", ".debit", ".wallet", ".crypto", ".token", ".nft",
    ".exchange", ".trading",
]

HIGH_RISK_COUNTRIES = {
    "RU", "CN", "KP", "IR", "NG", "UA", "RO", "BY", "MD", "AM",
    "GE", "AZ", "KZ", "UZ", "TJ", "TM", "KG", "MN", "VN", "ID",
}

# ─────────────────────────────────────────────────────────────────────────────
# SUSPICIOUS KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX: These words are now checked against the DOMAIN ONLY, not domain+path.
# Previously "login" in /vforcesite/LMS_Login would fire for awsacademy.com.
DOMAIN_SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "confirm", "password", "credential",
    "webscr", "cmd=_login", "submit-form", "validate",
    "authenticate", "activation", "recover", "unlock", "suspended",
]

# Words checked against PATH only — separate weaker signal (feature #43)
PATH_SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "webscr", "password", "credential",
    "authenticate",
]

# Domains exempt from path-keyword scoring
_PATH_KEYWORD_EXEMPT = {
    "awsacademy.com", "awseducate.com", "amazonaws.com",
    "canvas.instructure.com", "blackboard.com", "moodle.org",
    "d2l.com", "brightspace.com", "coursera.org", "edx.org", "udemy.com",
    "salesforce.com", "trailhead.salesforce.com",
    "atlassian.com", "auth0.com", "okta.com", "onelogin.com",
    "login.gov", "id.me", "zoom.us", "slack.com", "notion.so",
    "github.com", "gitlab.com", "google.com", "microsoft.com",
    "apple.com", "amazon.com",
}


def _domain_is_path_exempt(domain: str) -> bool:
    d = domain.lower()
    for trusted in _PATH_KEYWORD_EXEMPT:
        if d == trusted or d.endswith("." + trusted):
            return True
    parts = d.split(".")
    if len(parts) >= 2:
        tld  = parts[-1]
        tld2 = ".".join(parts[-2:])
        if tld in {"edu", "gov", "mil"}:
            return True
        if tld2 in {"ac.in", "ac.uk", "ac.jp", "gov.in", "gov.uk", "gov.au", "edu.au"}:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────
def _domain_entropy(domain):
    s = domain.replace("www.", "").split(".")[0]
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return round(-sum((c / length) * math.log2(c / length) for c in counts.values()), 3)


def _brand_impersonation(domain):
    dc = domain.lower().replace("www.", "")
    for brand in BRANDS:
        if brand in dc:
            parts = dc.split(".")
            if len(parts) >= 2 and parts[-2] != brand:
                return True
    return False


def _domain_age(domain):
    if not WHOIS_OK:
        return -1
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            if hasattr(created, "tzinfo") and created.tzinfo:
                from datetime import timezone
                return max(0, (datetime.now(timezone.utc) - created).days)
            return max(0, (datetime.now() - created).days)
    except Exception:
        pass
    return -1


def _ssl_cert_age(domain):
    try:
        ctx  = ssl.create_default_context()
        conn = ctx.wrap_socket(socket.socket(), server_hostname=domain)
        conn.settimeout(3)
        conn.connect((domain, 443))
        cert   = conn.getpeercert()
        conn.close()
        issued = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z")
        return max(0, (datetime.now() - issued).days)
    except Exception:
        return -1


def _dns_check(domain):
    r = {"dns_resolves": False, "has_mx": False, "has_spf": False}
    if not DNS_OK:
        return r
    try:
        dns.resolver.resolve(domain, "A",  lifetime=3)
        r["dns_resolves"] = True
    except Exception:
        pass
    try:
        dns.resolver.resolve(domain, "MX", lifetime=3)
        r["has_mx"] = True
    except Exception:
        pass
    try:
        txts = dns.resolver.resolve(domain, "TXT", lifetime=3)
        r["has_spf"] = any("spf" in str(x).lower() for x in txts)
    except Exception:
        pass
    return r


def _levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), curr[j] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def _typosquatting(domain):
    dc    = domain.lower().replace("www.", "")
    parts = dc.split(".")
    sld   = parts[-2] if len(parts) >= 2 else dc
    for brand in BRANDS:
        if sld == brand:
            return False
        dist = _levenshtein(sld, brand)
        if dist <= 2 and len(sld) >= 4:
            return True
    return False


def _homograph_attack(domain):
    dc = domain.lower().replace("www.", "")
    if "xn--" in dc:
        return True
    try:
        dc.encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _redirect_chain(url):
    try:
        session = requests.Session()
        session.max_redirects = 5
        resp      = session.head(url, allow_redirects=True, timeout=5,
                                 headers={"User-Agent": "Mozilla/5.0"})
        history   = resp.history
        hop_count = len(history)
        if hop_count == 0:
            return 0, False, False

        def etld1(d):
            p = d.split(".")
            return ".".join(p[-2:]) if len(p) >= 2 else d

        orig_d     = etld1(urlparse(url).netloc.lower().replace("www.", ""))
        final_d    = etld1(urlparse(resp.url).netloc.lower().replace("www.", ""))
        changed    = orig_d != final_d
        suspicious = hop_count >= 3 or changed
        return hop_count, changed, suspicious
    except Exception:
        return 0, False, False


def _cert_transparency(domain):
    try:
        parts = domain.replace("www.", "").split(".")
        root  = ".".join(parts[-2:]) if len(parts) >= 2 else domain
        r = requests.get(
            f"https://crt.sh/?q={root}&output=json",
            timeout=5, headers={"User-Agent": "PhishGuard/5.0"},
        )
        if r.status_code == 200:
            return len(r.json())
    except Exception:
        pass
    return -1


def _ip_geo_risk(domain):
    try:
        ip = socket.gethostbyname(domain)
        if ip.startswith(("10.", "192.168.", "127.")):
            return "", False, False
        r = requests.get(
            f"http://ip-api.com/json/{ip}?fields=countryCode,proxy,hosting",
            timeout=4,
        )
        if r.status_code == 200:
            d        = r.json()
            cc       = d.get("countryCode", "")
            is_risky = cc in HIGH_RISK_COUNTRIES
            is_proxy = d.get("proxy", False) or d.get("hosting", False)
            return cc, is_risky, is_proxy
    except Exception:
        pass
    return "", False, False


def _page_content_analysis(url, domain):
    result = {
        "has_password_field":   False,
        "title_brand_mismatch": False,
        "has_hidden_iframe":    False,
        "form_external_action": False,
    }
    try:
        r    = requests.get(url, timeout=6, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        html = r.text.lower()
        result["has_password_field"] = ('type="password"' in html or "type='password'" in html)
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
        if title_m:
            title = title_m.group(1).lower()
            for brand in BRANDS:
                if brand in title and brand not in domain.lower():
                    result["title_brand_mismatch"] = True
                    break
        result["has_hidden_iframe"] = bool(re.search(
            r'<iframe[^>]*(width\s*=\s*["\']?\s*0|height\s*=\s*["\']?\s*0|display\s*:\s*none)', html
        ))
        for action in re.findall(r'<form[^>]*action\s*=\s*["\']([^"\']+)["\']', html):
            if action.startswith("http") and domain not in action:
                result["form_external_action"] = True
                break
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE KEYS  — 42 original + 1 new = 43 total
# ─────────────────────────────────────────────────────────────────────────────
# The first 42 entries are IDENTICAL to the original features.py so that
# any existing model.pkl (trained on 42 features) still loads cleanly.
# Feature #43 (path_login_keyword) is appended at the end — retrain to use it.
FEATURE_KEYS = [
    # ── Original 42 (DO NOT reorder — model.pkl depends on this order) ────────
    "url_length",
    "has_ip",
    "has_at",
    "is_https",
    "suspicious_keywords",       # ← domain-only after fix (was domain+path)
    "dot_count",
    "subdomain_count",
    "has_redirect",
    "has_hex",
    "domain_length",
    "path_depth",
    "has_dash_in_domain",
    "has_query",
    "query_param_count",
    "domain_has_digits",
    "suspicious_tld",
    "has_double_slash",
    "longest_word_length",
    "domain_entropy",
    "high_entropy_domain",
    "brand_impersonation",
    "domain_age_days",
    "is_new_domain",
    "ssl_cert_age_days",
    "new_ssl_cert",
    "dns_resolves",
    "has_mx_record",
    "has_spf_record",
    "typosquatting",
    "homograph_attack",
    "gsb_flagged",
    "redirect_hop_count",
    "redirect_domain_changed",
    "redirect_chain_suspicious",
    "high_risk_country",
    "is_proxy_hosting",
    "cert_transparency_count",
    "low_cert_history",
    "has_password_field",
    "title_brand_mismatch",
    "has_hidden_iframe",
    "form_external_action",
    # ── New feature #43 ──────────────────────────────────────────────────────
    "path_login_keyword",        # login/signin in PATH on non-exempt domain
]


def get_feature_count() -> int:
    return len(FEATURE_KEYS)


def features_to_list(features: dict) -> list:
    return [
        int(features[k]) if isinstance(features[k], bool) else features.get(k, 0)
        for k in FEATURE_KEYS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(url: str, deep_scan: bool = False) -> dict:
    features = {}
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if ":" in domain:
        domain = domain.split(":")[0]
    path = parsed.path.lower()

    # ── URL structure ─────────────────────────────────────────────────────────
    features["url_length"]        = len(url)
    features["has_ip"]            = bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", domain))
    features["has_at"]            = "@" in url
    features["is_https"]          = parsed.scheme == "https"
    features["dot_count"]         = domain.count(".")
    features["has_redirect"]      = url.count("//") > 1
    features["has_hex"]           = "%" in url
    features["domain_length"]     = len(domain)
    features["path_depth"]        = path.count("/")
    features["has_dash_in_domain"]= "-" in domain
    features["has_query"]         = len(parsed.query) > 0
    features["query_param_count"] = len(parsed.query.split("&")) if parsed.query else 0
    features["domain_has_digits"] = bool(re.search(r"\d", domain))
    features["suspicious_tld"]    = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
    features["has_double_slash"]  = "//" in path

    words = re.split(r"[/.\-_?=&]", url.lower())
    features["longest_word_length"] = max((len(w) for w in words), default=0)

    # ── Domain analysis ───────────────────────────────────────────────────────
    parts = domain.split(".")
    features["subdomain_count"] = max(0, len(parts) - 2)

    entropy = _domain_entropy(domain)
    features["domain_entropy"]      = entropy
    features["high_entropy_domain"] = entropy > 3.5
    features["brand_impersonation"] = _brand_impersonation(domain)
    features["typosquatting"]       = _typosquatting(domain)
    features["homograph_attack"]    = _homograph_attack(domain)

    # ── BUG FIX: suspicious_keywords — domain ONLY, not domain+path ──────────
    # Original code:  check_target = (domain + path).lower()
    # Fixed code:     check_target = domain.lower()
    # Why: "login" in awsacademy.com/vforcesite/LMS_Login was triggering
    # because the path was included.  Now only domain-level keywords fire.
    # paypal-login.ru → still fires ✓  |  awsacademy.com/LMS_Login → clean ✓
    domain_lower = domain.lower()
    features["suspicious_keywords"] = any(w in domain_lower for w in DOMAIN_SUSPICIOUS_WORDS)

    # ── NEW feature #43: path keyword on non-exempt domain ───────────────────
    path_has_kw = any(w in path.lower() for w in PATH_SUSPICIOUS_WORDS)
    features["path_login_keyword"] = path_has_kw and not _domain_is_path_exempt(domain)

    # ── GSB placeholder (overwritten in app.py after real GSB call) ──────────
    features["gsb_flagged"] = False

    # ── Redirect chain ────────────────────────────────────────────────────────
    if deep_scan:
        hop_count, domain_changed, chain_suspicious = _redirect_chain(url)
    else:
        hop_count, domain_changed, chain_suspicious = 0, False, False
    features["redirect_hop_count"]        = hop_count
    features["redirect_domain_changed"]   = domain_changed
    features["redirect_chain_suspicious"] = chain_suspicious

    # ── IP geo risk ───────────────────────────────────────────────────────────
    if deep_scan:
        _cc, is_risky_country, is_proxy = _ip_geo_risk(domain)
    else:
        is_risky_country, is_proxy = False, False
    features["high_risk_country"] = is_risky_country
    features["is_proxy_hosting"]  = is_proxy

    # ── Deep scan features ────────────────────────────────────────────────────
    if deep_scan:
        age = _domain_age(domain)
        features["domain_age_days"]   = age
        features["is_new_domain"]     = (age != -1 and age < 30)
        cert_age = _ssl_cert_age(domain)
        features["ssl_cert_age_days"] = cert_age
        features["new_ssl_cert"]      = (cert_age != -1 and cert_age < 30)
        dns_r = _dns_check(domain)
        features["dns_resolves"]   = dns_r["dns_resolves"]
        features["has_mx_record"]  = dns_r["has_mx"]
        features["has_spf_record"] = dns_r["has_spf"]
        cert_count = _cert_transparency(domain)
        features["cert_transparency_count"] = cert_count
        features["low_cert_history"]        = (0 <= cert_count <= 2)
        page = _page_content_analysis(url, domain)
        features["has_password_field"]   = page["has_password_field"]
        features["title_brand_mismatch"] = page["title_brand_mismatch"]
        features["has_hidden_iframe"]    = page["has_hidden_iframe"]
        features["form_external_action"] = page["form_external_action"]
    else:
        features["domain_age_days"]         = -1
        features["is_new_domain"]           = False
        features["ssl_cert_age_days"]       = -1
        features["new_ssl_cert"]            = False
        features["dns_resolves"]            = True
        features["has_mx_record"]           = True
        features["has_spf_record"]          = False
        features["cert_transparency_count"] = -1
        features["low_cert_history"]        = False
        features["has_password_field"]      = False
        features["title_brand_mismatch"]    = False
        features["has_hidden_iframe"]       = False
        features["form_external_action"]    = False

    return features