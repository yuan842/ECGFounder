# First-Timer's Guide

**Read this first if you've never deployed something to AWS before, or if "EC2", "VPC", and "security group" don't yet mean anything concrete to you.**

This document does three things:
1. Tells you to **skip AWS for now** and run the demo on your laptop first.
2. Gives you a **plain-English mental model** of what each piece does and why.
3. Provides a **gotcha-by-gotcha** survival guide for the AWS upgrade later.

If you're already comfortable with AWS, skip straight to [SETUP.md](SETUP.md).

---

## 1. Honest pre-recommendation: do it locally first

**You probably do not need AWS for your first demo.**

Streamlit can run on your laptop in 5 minutes. If your demo audience is in the same room — or on a screen-share — pointing them at `http://localhost:8000` from your laptop is a perfectly legitimate, perfectly impressive demo. Many product launches at much bigger companies happen exactly this way.

Reasons to do **AWS** instead of local:
- ✅ The demo needs to be live 24/7 with people accessing it without you
- ✅ You need multiple people in different time zones to hit it
- ✅ You want a stable URL that survives your laptop sleeping
- ✅ The demo is part of an automated test pipeline

Reasons the **local path** is fine for now:
- ✅ One-shot demos with you in the room (or on Zoom)
- ✅ Learning what the app does before committing to infra
- ✅ Iterating on the app code with instant feedback (`streamlit run` auto-reloads)
- ✅ Zero cost, zero AWS-account confusion, zero VPN config, zero IAM permissions

**Recommendation: do Phase 0 below first. Don't touch AWS until Phase 0 is working and you've shown it to at least one person.**

---

## 2. Phase 0 — Local Streamlit demo (~30 min total)

You'll need: a Mac, Windows, or Linux laptop with internet.

### 2.1 Install Python 3 (if you don't have it)

```bash
# Check what you have:
python3 --version

# Mac (recommended path: install via Homebrew):
brew install python@3.11

# Windows:
# Download installer from https://www.python.org/downloads/
# IMPORTANT: tick "Add Python to PATH" during install.

# Linux (Ubuntu/Debian):
sudo apt update && sudo apt install -y python3 python3-pip
```

Aim for **Python 3.10 or newer**. If `python3 --version` prints `3.10.x` or higher, you're good.

### 2.2 Clone the ECGFounder repo

```bash
# Pick a folder you can find later:
cd ~/Documents     # or wherever you want
git clone https://github.com/<your-org>/ECGFounder.git
cd ECGFounder
```

If you don't have `git`, install Git first: [git-scm.com/downloads](https://git-scm.com/downloads).

### 2.3 Install Python dependencies

```bash
pip3 install -r requirements.txt streamlit
```

This downloads ~2 GB (PyTorch is the biggest piece). Takes ~5 min on a normal internet connection. **It is normal for `pip` to print a lot of warnings.** As long as the last line says something like "Successfully installed", you're fine.

### 2.4 First model warm-up

```bash
python3 -c "from checkpoints import load_ecgfounder; from device_utils import resolve_device; load_ecgfounder(resolve_device())"
```

This downloads the model checkpoint (~350 MB, one-time). Takes ~1 min.

### 2.5 Run the Streamlit app

> ⚠️ **Note: the app file (`app/app.py`) does not exist yet** — it gets created in phase 1 of the implementation plan. Once it exists, this command launches it:

```bash
streamlit run app/app.py
```

Streamlit prints something like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8000
```

Open that URL in your browser. **That's the demo.** You can:
- Drag an ECG file from `data/ecg_tp_fzark/Atrial Fibrillation/` onto the page
- Click the sidebar sample buttons
- Watch the model run, see the findings list, the waveform, the bar chart

**Stop it with `Ctrl+C` in the terminal.**

### 2.6 Show it to someone

Best demo UX from a laptop:

| Audience location | Best way to show |
|---|---|
| In the same room | Plug your laptop into the projector/TV, open the browser at `http://localhost:8000` |
| On a Zoom / Google Meet call | Share your screen showing the browser at `http://localhost:8000` |
| Remote but you don't have video call set up | Use [ngrok](https://ngrok.com/) to tunnel `localhost:8000` to a public HTTPS URL — `ngrok http 8000`, give them the printed URL. (See §"Going semi-public without AWS" below.) |

**This is enough for most internal demos.** Only escalate to AWS if Phase 0 isn't enough.

---

## 3. Plain-English mental model

Before you touch AWS, here's what each piece is doing and why it exists. No jargon.

### 3.1 The four moving parts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Your laptop          A "server" on AWS         The model file         │
│   ┌─────────┐          ┌─────────────────┐       ┌───────────────┐      │
│   │ Chrome  │  ◄──────►│  Streamlit app  │ ◄────►│ 350 MB .pth   │      │
│   │ browser │   HTTP   │  (Python code)  │  reads│ (the AI brain)│      │
│   └─────────┘          └─────────────────┘       └───────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Your laptop's browser** is the UI. It does nothing smart — it just shows what the server sends.
- **The "server" on AWS** is a regular Linux computer running 24/7. It runs the Streamlit app (which is itself Python code that we wrote).
- **The model file** is just a 350 MB file sitting on the server's disk. It holds the numbers (weights) that make AI predictions possible. Downloaded once.
- **HTTP** is the protocol the browser and the server use to talk. Same protocol any website uses.

In Phase 0 (local), your laptop plays all three roles. In Phase A–F (AWS), the laptop is just the browser; the model and the Python code live on a rented Linux computer.

### 3.2 What AWS actually provides

Think of AWS like a hotel that rents you a computer by the hour:

| AWS service | Plain English | Cost when not in use |
|---|---|---|
| **EC2** (Elastic Compute Cloud) | The actual rented Linux computer | Stopped = ~$0/hr |
| **EBS** (Elastic Block Store) | The hard disk attached to the rented computer | ~$1.50/month even when EC2 stopped |
| **VPC** (Virtual Private Cloud) | The network the computer is on | $0 |
| **Security Group** | A firewall that says who can connect | $0 |
| **IAM** (Identity & Access Mgmt) | The user accounts and permissions for *you* to manage AWS | $0 |

You'll touch **EC2** the most (~95% of the work) and the rest only once during initial setup.

### 3.3 What a "region" is

AWS has data centers around the world. A region is a geographic location:
- `us-east-1` — Northern Virginia (cheapest, most services available, latency may be high from West Coast)
- `us-west-2` — Oregon (similar to us-east-1, lower latency from West Coast)
- `eu-west-1` — Ireland
- etc.

**Pick the region closest to where you'll be running the demo from.** For most US-based teams, `us-east-1` is the default; it's the cheapest and has every feature.

### 3.4 What "the VPN" is

If your organization has a VPN (Cisco AnyConnect, Tailscale, OpenVPN, Pulse), it puts your laptop on the **internal** network. From the internal network, you can reach internal-only things — like the EC2 instance we're going to set up.

If your org **doesn't have a VPN**, you have three options:
- (a) Use [Tailscale](https://tailscale.com/) personally — free for ≤100 devices, takes ~10 min to set up.
- (b) Let the EC2 instance accept connections from your home IP only (less secure but functional).
- (c) Stay on the local Phase 0 path and skip AWS.

Most "I'm new" demos use option (c) until they outgrow it.

---

## 4. Pre-flight checklist for the AWS path

Don't start Phase A in [SETUP.md](SETUP.md) until **all** of these are true.

| ✅ | Item | How to confirm |
|---|---|---|
| ☐ | I have a working credit/debit card | (AWS will charge ~$60/month if you leave it running) |
| ☐ | I have an email address only I control | (AWS account-creation email goes here) |
| ☐ | I know what region I'll use | typically `us-east-1` — pick once, stick with it |
| ☐ | I have either: a corporate VPN OR I'm OK with Tailscale OR home-IP-only | (see §3.4) |
| ☐ | I have a terminal app I'm comfortable typing into | macOS: Terminal.app / iTerm2; Windows: PowerShell or WSL; Linux: any |
| ☐ | I have **completed Phase 0 above** | Streamlit running on my laptop, I've used it on a real ECG file |

The last one is the most important. **Do not skip Phase 0.** Debugging "is it the model? is it Streamlit? is it AWS?" simultaneously is a nightmare. Get the model + Streamlit working on your laptop first, then you know AWS is the only variable.

---

## 5. Cost panic prevention (read this BEFORE creating the AWS account)

The single most stressful thing about cloud is "I forgot to turn it off and now I have a $400 bill". Set these up the moment your AWS account exists.

### 5.1 Set a billing alarm

In the AWS console:
1. Click your username top-right → **Billing and Cost Management**.
2. **Billing preferences** → enable "Receive Free Tier Usage Alerts" and "Receive Billing Alerts".
3. **Budgets** → **Create budget** → "Cost budget" → set to **$10/month** with email alert at 50%, 80%, 100%.

If the budget alert fires, log in immediately and check what's running.

### 5.2 Recognize the only thing that costs money

For this demo, the only charge is the **EC2 instance running**. As long as you `stop` the instance (not just close your browser), the meter stops.

```bash
# When you're done demoing, on your laptop:
aws ec2 stop-instances --instance-ids i-<your-instance-id>

# Before the next demo:
aws ec2 start-instances --instance-ids i-<your-instance-id>
```

Both commands are **free** to run, take ~1 minute, and the data on the EBS disk is preserved.

### 5.3 Worst-case scenarios and their cost

| Scenario | What you'll be charged |
|---|---|
| You leave a `t3.large` running 24/7 for the month | ~$62 |
| You leave a `t3.large` running 24/7 for a year | ~$750 |
| You accidentally launched a `p3.8xlarge` (8 GPUs) and left it on overnight | ~$100 in 8 hours 😱 |
| You stopped the instance properly | ~$1.50/month (just the disk) |

**Defensive rule for newbies**: when launching the instance in Phase B of SETUP.md, the dropdown shows hundreds of instance types. **Only ever pick from the `t2.*` or `t3.*` family until you know what you're doing.** Those are all cheap (cents per hour). The expensive ones (`p3.*`, `g5.*`, `x1e.*`) are GPU / huge-RAM types that cost dollars-per-hour.

### 5.4 Nuclear option if something goes wrong

If your billing alarm fires and you can't figure out what's running:

1. **Stop everything** — go to EC2 console, select every running instance, click `Instance state → Stop`.
2. Email AWS Support — they're very forgiving about first-time-user surprise bills, especially if you stopped things promptly.
3. Don't panic — you're not the first person to do this.

---

## 6. Glossary

Skim this once. You don't need to memorize anything; come back when an acronym confuses you in SETUP.md.

| Term | What it actually is |
|---|---|
| **AWS** | Amazon Web Services — the cloud platform |
| **EC2** | Elastic Compute Cloud — rented Linux/Windows servers |
| **EBS** | Elastic Block Store — the disk attached to an EC2 instance |
| **VPC** | Virtual Private Cloud — the private network your EC2 lives in |
| **IAM** | Identity & Access Mgmt — users and permissions for AWS |
| **AMI** | Amazon Machine Image — the OS template your EC2 boots from (we use Ubuntu 22.04) |
| **Security Group** | A firewall attached to your EC2; controls who can connect on which port |
| **Region** | A geographic data-center location like `us-east-1` (Virginia) |
| **CIDR** | Network notation like `10.0.0.0/8` — a range of IP addresses. For VPN whitelisting. |
| **SSH** | Secure Shell — protocol for remote terminal access to a Linux server |
| **systemd** | The Linux service manager that auto-starts our Streamlit app on boot |
| **journalctl** | The systemd log viewer — `journalctl -u ecg-demo -f` tails the demo's logs |
| **Streamlit** | The Python web-app framework we use for the UI |
| **t3.large** | A specific EC2 instance size (2 CPU cores, 8 GB RAM, no GPU) — ~$0.08/hour |
| **`pip`** | The Python package installer — `pip install streamlit` |
| **systemctl** | The CLI command to start/stop/restart systemd services |
| **`apt`** | Ubuntu's package installer — `apt install python3` |

---

## 7. Going semi-public without AWS (ngrok)

If you want a temporary public URL to share with one remote viewer for a 30-minute demo, you don't need AWS. Use ngrok:

```bash
# One-time install:
brew install ngrok          # macOS
# or download from https://ngrok.com/download

# In one terminal, run the app:
streamlit run app/app.py

# In another terminal, expose port 8000:
ngrok http 8000
```

ngrok prints a URL like `https://random-words-1234.ngrok-free.app`. Share that link; it works from anywhere in the world. When you `Ctrl+C` ngrok, the URL stops working.

**Pros**: 5-minute setup, no AWS, free for occasional use, HTTPS for free.
**Cons**: URL changes every restart, only works while your laptop is on, ngrok's free tier may show an interstitial page.

**Do NOT use this with real patient data** — your traffic goes through ngrok's servers. Synthetic / public datasets only.

---

## 8. Common first-timer gotchas (with first-aid)

### "I can't `pip install` — it says permission denied"

Two fixes:
```bash
pip3 install --user streamlit       # installs to your user dir, no sudo needed
# OR
python3 -m venv ~/.venv-ecg && source ~/.venv-ecg/bin/activate
pip install streamlit              # inside virtualenv, no permission issues
```

The `--user` flag is fine for a one-time setup; virtualenv is the long-term-correct way.

### "The Streamlit page won't open in my browser"

In order:
1. Is the `streamlit run` command still running in your terminal? It must stay running. `Ctrl+C` to stop intentionally.
2. Did you go to the URL Streamlit printed (`http://localhost:8000` typically — but sometimes `http://localhost:8501`)?
3. Did your firewall pop up a permission request? Allow Python to accept incoming connections.
4. Try `http://127.0.0.1:8000` instead of `http://localhost:8000`.

### "I get `ModuleNotFoundError: No module named 'streamlit'`"

`pip` and `python3` are out of sync. On macOS especially. Use:
```bash
python3 -m pip install streamlit
python3 -m streamlit run app/app.py
```

The `-m` form guarantees you're using the same Python both times.

### "I created an AWS account but the EC2 dashboard is empty / confused"

Did you pick the right **region** at the top-right of the AWS console? AWS shows you what's in *one region at a time*. If your EC2 is in `us-east-1` but you're looking at `us-west-2`, you'll see nothing.

### "The EC2 instance is launched but I can't SSH in"

Check in order:
1. Are you on the VPN (or did you allow your home IP in the security group)?
2. Did the security group's inbound rule allow port 22 from your current IP?
3. Is the EC2 instance status `running` (not `pending` or `stopped`)?
4. Are you using the right key file? `ssh -i ~/.ssh/ecg-demo ubuntu@<ip>` — the `-i` flag is mandatory.
5. Does the key have the right permissions? `chmod 600 ~/.ssh/ecg-demo`.

### "SSH works but I get connection refused on port 8000"

Most likely the security group has port 22 open but not port 8000. Add an inbound rule for TCP port 8000 from your IP/CIDR.

---

## 9. Where to ask for help

When you're stuck, gather these before asking anyone:

1. **What you ran** — the exact command, copy-paste.
2. **What you saw** — the exact error message, copy-paste.
3. **What you expected** — one sentence.
4. **What you've already tried** — saves the helper's time.

Channels (in order of preference for this project):
- **Internal Slack `#ecg-demo`** (or wherever the project chat lives) — best for project-specific questions.
- **AWS Re:Post** ([repost.aws](https://repost.aws/)) — official Q&A for AWS questions.
- **Stack Overflow** with the `streamlit` or `aws` tag — for framework/AWS questions.
- **`asksomeone-near-you` IRL** — for "this menu doesn't match the doc" type confusion. AWS UI changes monthly; sometimes screenshots in docs go stale.

---

## 10. Realistic timeline expectations

The SETUP.md doc says "~45 minutes". That assumes you've done AWS before. For a true first-timer, here's a more honest estimate:

| Activity | First time | Second time |
|---|---|---|
| Phase 0 — local Streamlit on your laptop | 30 min | 5 min |
| AWS account creation + billing setup | 30 min | 0 (already done) |
| Phase A — AWS CLI + SSH key setup | 30 min | 5 min |
| Phase B — provision EC2 in console | 30 min | 5 min |
| Phase C–D — install on EC2 + systemd | 30 min | 10 min |
| Phase E — verification + browser test | 20 min | 5 min |
| **Total first-time, AWS path** | **~3 hours** | ~30 min |
| Phase 0 only (laptop demo) | 30 min | — |

If you have 30 minutes today, do Phase 0. If you have an afternoon, do the AWS path.

---

## 11. The TL;DR

If you only read one section, read this one:

1. **Don't start with AWS.** Run the Streamlit app on your laptop (Phase 0 above). That's enough for 90% of internal demos.
2. **Set a $10 budget alarm before doing anything in AWS.** This protects you from surprise bills.
3. **Only ever launch `t2.*` or `t3.*` EC2 instance types.** The expensive ones cost $10+ per hour.
4. **Stop the instance when you're done demoing.** `aws ec2 stop-instances --instance-ids i-...` — free, takes 1 minute, drops cost to ~$1.50/month.
5. **Read [SETUP.md](SETUP.md) once Phase 0 is working** and you actually need cloud hosting.

Everything else is detail.
