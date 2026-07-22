# Deploying the Meridian web dashboard on a Linux VPS

This runs `web_server.py` as an always-on systemd service that restarts on crash
and starts on boot. The server binds to **127.0.0.1:8787** (localhost only), so by
default nothing is exposed to the internet — you reach it over an SSH tunnel, or
put a reverse proxy with authentication in front of it.

> **Security note:** the dashboard has **no authentication**. Never expose port
> 8787 directly to the public internet. Use the SSH-tunnel or the
> nginx + basic-auth options below.

---

## 1. Install

```bash
# System deps (Debian/Ubuntu)
sudo apt update && sudo apt install -y python3 python3-venv git

# Dedicated unprivileged service user
sudo useradd -r -m -d /opt/meridian -s /usr/sbin/nologin meridian

# Clone into /opt/meridian
sudo -u meridian git clone https://github.com/alexanderlopez4997-sketch/al.git /opt/meridian
cd /opt/meridian

# Virtualenv + dependencies
sudo -u meridian python3 -m venv .venv
sudo -u meridian .venv/bin/pip install --upgrade pip
sudo -u meridian .venv/bin/pip install -r requirements.txt
```

## 2. Configure API keys

Create `/opt/meridian/.env` (git-ignored; `web_server.py` auto-loads it on start).
Copy `.env.example` for the full list. Minimum for live data:

```bash
sudo -u meridian tee /opt/meridian/.env >/dev/null <<'EOF'
FINNHUB_KEY=your_finnhub_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
QUIVER_API_TOKEN=your_quiver_token
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
EOF
sudo chmod 600 /opt/meridian/.env
```

Any key you omit just leaves that feed off; SEC·EDGAR needs no key.

## 3. Install and start the service

```bash
sudo cp /opt/meridian/deploy/meridian.service /etc/systemd/system/meridian.service
# If you cloned somewhere other than /opt/meridian, edit WorkingDirectory,
# ExecStart, User, and ReadWritePaths in the unit first.
sudo systemctl daemon-reload
sudo systemctl enable --now meridian
sudo systemctl status meridian          # should show "active (running)"
journalctl -u meridian -f               # follow logs
```

## 4. Reach the dashboard

**Option A — SSH tunnel (simplest, safest, no extra setup):**

```bash
# from your laptop
ssh -L 8787:localhost:8787 youruser@your-vps
# then open http://localhost:8787 in your browser
```

**Option B — public via nginx + HTTP basic auth (if you want a real URL):**

```bash
sudo apt install -y nginx apache2-utils
sudo htpasswd -c /etc/nginx/.meridian_htpasswd youruser   # sets a password
```

```nginx
# /etc/nginx/sites-available/meridian  (symlink into sites-enabled, then reload)
server {
    listen 80;
    server_name dashboard.example.com;      # your domain

    location / {
        auth_basic "Meridian";
        auth_basic_user_file /etc/nginx/.meridian_htpasswd;
        proxy_pass http://127.0.0.1:8787;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/meridian /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Then add TLS with `sudo certbot --nginx -d dashboard.example.com` (Let's Encrypt).

## 5. Updating

```bash
cd /opt/meridian
sudo -u meridian git pull
sudo -u meridian .venv/bin/pip install -r requirements.txt   # if deps changed
sudo systemctl restart meridian
```

## Troubleshooting

- **Feeds show off:** the `.env` is missing keys, or you changed it without
  restarting — `web_server.py` reads keys once at startup, so
  `sudo systemctl restart meridian` after editing `.env`.
- **Service won't start:** `journalctl -u meridian -e` for the traceback. Most
  often a missing dependency (`pip install -r requirements.txt`) or a wrong path
  in the unit file.
- **Live prices missing but demo works:** the box can't reach the data provider
  (network egress / firewall). Demo mode never needs the network.
