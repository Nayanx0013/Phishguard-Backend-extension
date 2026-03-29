

import threading
import time
import requests
from urllib.parse import urlparse



_NEVER_PHISHING = {
    "google.com", "youtube.com", "facebook.com", "amazon.com", "microsoft.com",
    "apple.com", "twitter.com", "x.com", "instagram.com", "linkedin.com",
    "github.com", "wikipedia.org", "reddit.com", "netflix.com", "paypal.com",
    "cloudflare.com", "amazonaws.com", "azure.com", "stripe.com",
}


class ThreatFeedManager:
    SOURCES = [
        {
            "name":   "OpenPhish",
            "url":    "https://openphish.com/feed.txt",
            "format": "url_per_line",
            "info":   "Community phishing feed, updated every 12 hours"
        },
        {
            "name":   "URLhaus (abuse.ch)",
            "url":    "https://urlhaus.abuse.ch/downloads/text/",
            "format": "url_per_line_skip_hash",
            "info":   "Malware distribution URLs, updated in real-time"
        },
        {
            "name":   "Phishing Army Extended",
            "url":    "https://phishing.army/download/phishing_extendedlist.txt",
            "format": "url_per_line_skip_hash",
            "info":   "Aggregates PhishTank+OpenPhish+Cert.pl, updated every 6h"
        },
        {
            "name":   "mitchellkrogza Active",
            "url":    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
            "format": "url_per_line_skip_hash",
            "info":   "GitHub-hosted phishing database, updated hourly"
        },
        {
            "name":   "mitchellkrogza New Today",
            "url":    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt",
            "format": "url_per_line_skip_hash",
            "info":   "Brand new phishing links from today only"
        },
        {
            "name":   "Jarelllama Scam Blocklist",
            "url":    "https://raw.githubusercontent.com/jarelllama/Scam-Blocklist/main/lists/wildcard_domains/scams.txt",
            "format": "domain_per_line_skip_hash",
            "info":   "Scam + phishing domains, uses DGA detection"
        },
        {
            "name":   "PeterDaveHello Phishing",
            "url":    "https://raw.githubusercontent.com/PeterDaveHello/threat-hostlist/master/phishing/hosts",
            "format": "hosts_file",
            "info":   "Curated phishing domain blocklist"
        },
        {
            "name":   "CyberHost Malware",
            "url":    "https://lists.cyberhost.uk/malware.txt",
            "format": "domain_per_line_skip_hash",
            "info":   "Verified malicious domains"
        },
        {
            "name":   "VX Vault URLs",
            "url":    "http://vxvault.net/URL_List.php",
            "format": "url_per_line",
            "info":   "Malware hosting URLs"
        },
        {
            "name":   "Phishing.Database Domains",
            "url":    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
            "format": "domain_per_line_skip_hash",
            "info":   "Active phishing domains only"
        },
    ]

    def __init__(self, refresh_hours: int = 1, lazy: bool = False):
        self.urls          = set()
        self.domains       = set()
        self.lock          = threading.Lock()
        self.stats         = {}
        self.last_refresh  = None
        self.refresh_hours = refresh_hours

        load_thread    = threading.Thread(target=self._load_all, daemon=True)
        load_thread.start()
        refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        refresh_thread.start()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_all(self):
        new_urls    = set()
        new_domains = set()
        new_stats   = {}

        for source in self.SOURCES:
            name  = source["name"]
            count = 0
            try:
                # Retry with backoff on transient failures
                r = None
                for attempt in range(3):
                    try:
                        r = requests.get(
                            source["url"], timeout=20,
                            headers={"User-Agent": "PhishGuard/5.0 Security Research"},
                        )
                        if r.status_code == 200:
                            break
                    except Exception:
                        time.sleep(2 ** attempt)

                if r is None or r.status_code != 200:
                    new_stats[name] = {"count": 0, "status": f"HTTP {getattr(r,'status_code','ERR')}"}
                    continue

                fmt   = source["format"]
                lines = r.text.strip().split("\n")

                for line in lines:
                    line = line.strip().rstrip(".")  # strip trailing dots
                    if not line:
                        continue
                    if fmt in ("url_per_line_skip_hash", "domain_per_line_skip_hash", "hosts_file"):
                        if line.startswith("#"):
                            continue

                    if fmt == "hosts_file":
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] in ("0.0.0.0", "127.0.0.1"):
                            d = parts[1].lower().strip()
                            if d and "." in d:
                                new_domains.add(d)
                                count += 1

                    elif fmt in ("domain_per_line_skip_hash", "domain_per_line"):
                        d = line.lower().strip()
                        if d.startswith("*."):
                            d = d[2:]
                        if d and "." in d and not d.startswith("http"):
                            new_domains.add(d)
                            count += 1

                    else:  # url_per_line / url_per_line_skip_hash
                        if line.startswith("http"):
                            new_urls.add(line)
                            try:
                                netloc = urlparse(line).netloc.lower()
                                if ":" in netloc:
                                    netloc = netloc.split(":")[0]
                                if netloc:
                                    new_domains.add(netloc)
                            except Exception:
                                pass
                            count += 1

                new_stats[name] = {"count": count, "status": "OK"}
                print(f"  ✅ {name}: {count:,} entries")

            except Exception as e:
                new_stats[name] = {"count": 0, "status": str(e)[:60]}
                print(f"  ⚠️  {name}: {e}")

        # Remove any entries that are major whitelisted brands (false positive guard)
        for safe in _NEVER_PHISHING:
            new_domains.discard(safe)
            new_domains.discard(f"www.{safe}")

        with self.lock:
            self.urls         = new_urls
            self.domains      = new_domains
            self.stats        = new_stats
            self.last_refresh = time.time()

        print(f"\n✅ Threat feeds loaded: {len(new_urls):,} URLs + {len(new_domains):,} domains")

    def _refresh_loop(self):
        time.sleep(self.refresh_hours * 3600)
        while True:
            print(f"\n🔄 Refreshing threat feeds ({self.refresh_hours}h interval)…")
            self._load_all()
            time.sleep(self.refresh_hours * 3600)

    # ── Lookup ────────────────────────────────────────────────────────────────
    def is_phishing(self, url: str) -> bool:
        """
        Returns True if url or any of its parent domains is in a threat feed.
        Walks the full subdomain tree: a.b.evil.com → checks b.evil.com → evil.com
        Never returns True for domains in _NEVER_PHISHING (false positive guard).
        """
        with self.lock:
            # Exact URL match
            if url in self.urls or url.rstrip("/") in self.urls:
                return True

            try:
                netloc = urlparse(url).netloc.lower()
                if ":" in netloc:
                    netloc = netloc.split(":")[0]
                netloc = netloc.rstrip(".")

                # Safety: never flag major brands even if in a feed
                base = netloc.replace("www.", "")
                if base in _NEVER_PHISHING:
                    return False

                # Walk subdomain tree
                parts = netloc.split(".")
                for i in range(len(parts) - 1):
                    candidate = ".".join(parts[i:])
                    if candidate in self.domains:
                        return True
            except Exception:
                pass

        return False

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        with self.lock:
            return {
                "total_urls":    len(self.urls),
                "total_domains": len(self.domains),
                "total_entries": len(self.urls) + len(self.domains),
                "sources":       self.stats,
                "last_refresh":  self.last_refresh,
                "refresh_hours": self.refresh_hours,
            }

    def get_summary(self) -> str:
        s = self.get_stats()
        return (f"{s['total_entries']:,} threat indicators "
                f"({s['total_urls']:,} URLs + {s['total_domains']:,} domains) "
                f"from {len(self.SOURCES)} sources")