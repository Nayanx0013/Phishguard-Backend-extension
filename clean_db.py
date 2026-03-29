

import sqlite3
from urllib.parse import urlparse

WHITELIST_DOMAINS = [
    "google.com", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "wikipedia.org", "youtube.com", "stackoverflow.com",
    "reddit.com", "linkedin.com", "twitter.com", "x.com",
    "instagram.com", "facebook.com", "netflix.com", "cloudflare.com",
    "python.org", "kaggle.com", "gitlab.com", "discord.com",
    "slack.com", "notion.so", "zoom.us", "paypal.com",
]

conn          = sqlite3.connect("scans.db")
total_deleted = 0

# FIX: Fetch all phishing scans and parse the domain from each URL properly
# instead of using LIKE '%domain%' which was too broad
phishing_rows = conn.execute(
    "SELECT id, url FROM scans WHERE result='PHISHING'"
).fetchall()

ids_to_delete = []
for row_id, url in phishing_rows:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        domain_clean = domain.replace("www.", "")

        for wl_domain in WHITELIST_DOMAINS:
            if domain == wl_domain or domain_clean == wl_domain:
                ids_to_delete.append(row_id)
                break
    except Exception:
        pass

if ids_to_delete:
    placeholders = ",".join("?" * len(ids_to_delete))
    conn.execute(f"DELETE FROM scans WHERE id IN ({placeholders})", ids_to_delete)
    total_deleted += len(ids_to_delete)
    print(f"  Removed {len(ids_to_delete)} false positive(s) for whitelisted domains")

# Delete localhost/internal scans (prefix match is safe here)
for pattern in ["http://localhost%", "chrome://%", "about:%", "chrome-extension://%"]:
    cur = conn.execute("DELETE FROM scans WHERE url LIKE ?", (pattern,))
    if cur.rowcount > 0:
        print(f"  Removed {cur.rowcount} internal scan(s) matching {pattern}")
        total_deleted += cur.rowcount

conn.commit()
conn.close()
print(f"\n✅ Total removed: {total_deleted} false positive scans")
print("✅ Database cleaned — restart python app.py")