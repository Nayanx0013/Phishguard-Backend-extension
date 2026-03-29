# gunicorn.conf.py - PhishGuard for Hugging Face Spaces
workers = 2
threads = 4
timeout = 60
bind = "0.0.0.0:7860"   # HF Spaces requires port 7860
worker_class = "gthread"
accesslog = "-"
errorlog  = "-"
loglevel  = "info"