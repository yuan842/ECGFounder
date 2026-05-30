# Setup Guide — ECG Cloud Demo

End-to-end installation and configuration to take the [Streamlit plan](STREAMLIT_APP.md) from "fresh AWS account" to "demo running at `http://<ec2>:8000`".

Audience: one operator following this once per environment. Estimated time: **~45 minutes** the first time; ~5 minutes for subsequent environments after Phase A is done.

---

## 0. Tools-and-where matrix

The whole stack uses **eight** tools across two machines. Nothing else.

### Local machine (your laptop)

| Tool | Why | Install method | Version |
|---|---|---|---|
| **AWS account** | EC2 hosting | https://aws.amazon.com/ — corporate SSO or root account | n/a |
| **AWS CLI v2** | provisioning, optional ops | `brew install awscli` (macOS) / `apt install awscli` (Linux) / [installer](https://aws.amazon.com/cli/) (Windows) | ≥ 2.13 |
| **SSH client** | connect to EC2 | built into macOS/Linux; `ssh.exe` on Windows 10+ | any |
| **VPN client** | reach EC2 from your laptop | per your org's standard (Tailscale, Cisco AnyConnect, OpenVPN, etc.) | n/a |
| **Modern web browser** | use the demo | Chrome / Safari / Edge / Firefox | recent |

### EC2 instance (Ubuntu 22.04 t3.large)

| Tool | Why | Install method (run on EC2) | Version |
|---|---|---|---|
| **Python 3** | runs Streamlit + model | `apt install python3 python3-pip` | ≥ 3.10 (Ubuntu 22.04 ships 3.10.12) |
| **git** | clone the repo | `apt install git` | any recent |
| **systemd** | run the demo as a service | preinstalled on Ubuntu | n/a |
| **Streamlit** | web framework | `pip install streamlit` | ≥ 1.28 (need `st.rerun`, `st.bar_chart(horizontal=True)`) |
| **Repo Python deps** | torch, numpy, pandas, wfdb, scipy | `pip install -r requirements.txt` | per repo's pinning |

That's it. **No Docker, no S3, no Lambda, no API Gateway, no CDN, no DB, no Redis, no nginx, no TLS cert manager** — those are all "out of scope" per [README.md §"Out of scope"](README.md#out-of-scope--when-to-not-use-this-stack).

---

## Phase A — One-time local-machine setup

Skip steps you've already done for other AWS work.

### A.1 AWS account access

Either:
- **Corporate SSO**: your engineering-platform team should have already enrolled you. Confirm you can sign in to `https://<your-org>.awsapps.com/start` and see the account that will host the demo (typically a "sandbox" or "research" account).
- **Personal account**: create at https://aws.amazon.com/, complete billing setup, create an IAM user with `AmazonEC2FullAccess` (not the root account for daily use).

### A.2 AWS CLI install + configure

```bash
# macOS
brew install awscli
# Linux
sudo apt update && sudo apt install -y awscli   # or use the official .pkg from AWS for v2

# Verify
aws --version
# expected: aws-cli/2.x ...

# Configure with credentials (one-time)
aws configure
#   AWS Access Key ID:     <your key>
#   AWS Secret Access Key: <your secret>
#   Default region name:   us-east-1   (or wherever the VPN exit is)
#   Default output format: json
```

For SSO-based access use `aws configure sso` instead; follow the prompts. Verify with:

```bash
aws sts get-caller-identity
# Should print your IAM user/role ARN.
```

### A.3 SSH key pair

```bash
# Create a dedicated key for the demo (don't reuse personal keys)
ssh-keygen -t ed25519 -f ~/.ssh/ecg-demo -C "ecg-demo $(date +%Y-%m-%d)"
# Press enter at the passphrase prompts (or set one if you prefer).

# Upload the public key to AWS as a key pair named "ecg-demo"
aws ec2 import-key-pair \
    --key-name ecg-demo \
    --public-key-material "fileb://~/.ssh/ecg-demo.pub"
```

### A.4 VPN access

Confirm you can connect to your org's VPN and reach an internal IP range (typically `10.0.0.0/8`). The EC2 instance in Phase B will be placed in a security group that accepts inbound port 8000 only from that range.

If your org doesn't have a VPN: a hardened EC2 demo box on the public internet requires TLS + auth (out of scope here — read the [README §"Out of scope"](README.md#out-of-scope--when-to-not-use-this-stack) before considering this).

---

## Phase B — Provision the EC2 instance

This is a one-time setup per environment (e.g. "dev" and "staging" demo boxes are separate).

### B.1 Launch the instance (AWS console route, easiest)

In the AWS console:

1. **EC2 → Launch Instance**.
2. **Name**: `ecg-demo`.
3. **AMI**: `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type` (64-bit x86) — current Free Tier eligible AMI.
4. **Instance type**: `t3.large` (2 vCPU, 8 GB RAM). Per [STREAMLIT_APP.md §12](STREAMLIT_APP.md#12-sizing) — **NOT** `t3.medium`, which has OOM risk.
5. **Key pair**: select `ecg-demo` (created in A.3).
6. **Network settings**:
   - VPC: your default or the VPC that the VPN reaches.
   - Subnet: a private subnet if the VPC has one; otherwise the default subnet.
   - Auto-assign public IP: **disabled** (private demo, accessed via VPN).
   - Security group: **create new**, named `ecg-demo-sg`, with rules:
     - **Inbound**: SSH (22) from `<your-VPN-CIDR>` (e.g. `10.0.0.0/8`), HTTP-alt (8000) from `<your-VPN-CIDR>`. **Block 0.0.0.0/0 entirely.**
     - **Outbound**: all (default).
7. **Storage**: 15 GB gp3, encrypted (default).
8. **Advanced → IAM instance profile**: leave **None** (the demo doesn't need any AWS perms). If your org mandates a baseline role (CloudWatch agent, SSM), attach that.
9. **Launch**.

Wait ~30 s for the instance to reach `running`. Note its **private IPv4** (e.g. `10.0.0.42`) — this is the `<ec2-host>` you'll use everywhere below.

### B.2 Equivalent CLI route (skip if you used the console)

```bash
# Find the latest Ubuntu 22.04 AMI in your region
AMI=$(aws ec2 describe-images --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)

# Create the security group
SG=$(aws ec2 create-security-group --group-name ecg-demo-sg \
    --description "ECG demo internal" --output text --query 'GroupId')
aws ec2 authorize-security-group-ingress --group-id $SG \
    --protocol tcp --port 22 --cidr 10.0.0.0/8       # adjust to your VPN CIDR
aws ec2 authorize-security-group-ingress --group-id $SG \
    --protocol tcp --port 8000 --cidr 10.0.0.0/8

# Launch
aws ec2 run-instances \
    --image-id $AMI --instance-type t3.large \
    --key-name ecg-demo --security-group-ids $SG \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=15,VolumeType=gp3,Encrypted=true}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ecg-demo}]'
```

### B.3 First SSH login (smoke test)

```bash
# Connect via VPN first, then:
ssh -i ~/.ssh/ecg-demo ubuntu@<ec2-host>
# (Type "yes" at the host-key prompt.)
```

If that connects, Phase B is done.

---

## Phase C — First-time EC2 setup

All commands below run **on the EC2 instance** (after SSH-ing in).

### C.1 System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
python3 --version    # expect 3.10.x
pip3 --version
```

### C.2 Clone the ECGFounder repo

```bash
cd ~
git clone https://github.com/<your-org>/ECGFounder.git
cd ECGFounder
git log -1 --oneline    # confirm you have the version you expect
```

### C.3 Install Python dependencies

```bash
# Repo deps (torch, numpy, pandas, scipy, wfdb, ...) + Streamlit
pip3 install -r requirements.txt streamlit

# Verify Streamlit version
python3 -c "import streamlit; print(streamlit.__version__)"
# expect 1.28+ — needed for st.rerun and st.bar_chart(horizontal=True)
```

If `pip` warns "running pip as the root user is not recommended", that's safe to ignore on EC2 (the `ubuntu` user pip-installs into `~/.local/`).

### C.4 Warm the model checkpoint cache

The first-time `load_ecgfounder()` call downloads `1_lead_ECGFounder.pth` (~353 MB) from Hugging Face. Do this once now so the first Streamlit request isn't slow.

```bash
python3 -c "from checkpoints import load_ecgfounder; from device_utils import resolve_device; load_ecgfounder(resolve_device())"
# expected: "[checkpoints] downloaded ./checkpoint/1_lead_ECGFounder.pth (352.7 MB)"
# subsequent runs: "[checkpoints] using existing ./checkpoint/1_lead_ECGFounder.pth"
```

Note: t3.large has no GPU — `resolve_device()` returns `cpu`. Inference is ~500 ms–1 s per request, still fast enough for the demo.

### C.5 Sanity-check the algorithm code works on this box

```bash
python3 -c "
import torch
from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from label_config import FZARK_ONTOLOGY
device = resolve_device()
model = DualHeadECGFounder(device, routing='ptbxl_specific')
x = torch.randn(1, 1, 5000).to(device)
out = torch.sigmoid(model(x)).cpu().numpy().ravel()
print(f'OK — output shape {out.shape}, sample: {[round(out[i], 3) for i in [4,5,6,9,16,19,93,98,142]]}')
"
# expect: "OK — output shape (150,), sample: [...]"
```

If this prints `OK`, the algorithm side is wired up correctly.

---

## Phase D — Install the demo service

The Streamlit app and its systemd unit. **Both files will be added in phase 1 of the implementation plan** — this section documents the install steps for once they exist.

### D.1 Drop the systemd unit

```bash
sudo cp ~/ECGFounder/deploy/ecg-demo.service /etc/systemd/system/
# Inspect the unit file once:
cat /etc/systemd/system/ecg-demo.service
```

Expected content (per [STREAMLIT_APP.md §11](STREAMLIT_APP.md#11-deployment-runbook)):

```ini
[Unit]
Description=ECG demo Streamlit app
After=network-online.target

[Service]
WorkingDirectory=/home/ubuntu/ECGFounder
Environment=PYTHONPATH=/home/ubuntu/ECGFounder
ExecStart=/usr/bin/python3 -m streamlit run app/app.py \
    --server.address 0.0.0.0 \
    --server.port 8000 \
    --server.maxUploadSize 25 \
    --browser.gatherUsageStats false \
    --server.headless true
Restart=always
RestartSec=5
User=ubuntu
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### D.2 Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ecg-demo
sudo systemctl status ecg-demo --no-pager
# expect: "active (running)"
```

If it says `failed`, see [§Troubleshooting](#troubleshooting) below.

### D.3 First-request warm-up

```bash
curl -sI http://localhost:8000/ | head -1
# expect: HTTP/1.1 200 OK
```

The first browser visit will trigger the `@st.cache_resource`-cached model load (~10 s on CPU since the file is already on disk). Subsequent visits are instant.

---

## Phase E — Verification checklist

Run through these in order; each builds on the previous.

| # | Check | Command / Action | Expected |
|---|---|---|---|
| 1 | systemd unit running | `sudo systemctl is-active ecg-demo` | `active` |
| 2 | port 8000 listening | `ss -tlnp 'sport = :8000'` | `LISTEN ... 0.0.0.0:8000` (uvicorn/python) |
| 3 | local HTTP responds | `curl -sI http://localhost:8000/` | `HTTP/1.1 200 OK` |
| 4 | VPN reach from laptop | `curl -sI http://<ec2-host>:8000/` (on laptop, VPN on) | `HTTP/1.1 200 OK` |
| 5 | browser loads page | open `http://<ec2-host>:8000/` in Chrome | Streamlit "ECG Founder — internal demo" UI |
| 6 | sample button works | click "AFib TP" in the sidebar | waveform + findings render in ~1 s |
| 7 | upload works | drag an `.json` from `data/ecg_tp_fzark/Atrial Fibrillation/` | same successful render |
| 8 | logs look clean | `sudo journalctl -u ecg-demo -n 50 --no-pager` | no `Traceback` / `ERROR` |

If all eight pass, the demo is live.

---

## Phase F — Day-2 operations

### F.1 Updating the code

```bash
# On the EC2 instance
cd ~/ECGFounder
git fetch && git pull
# If new pip deps were added in this update:
pip3 install -r requirements.txt streamlit
# Restart the service
sudo systemctl restart ecg-demo
sudo journalctl -u ecg-demo -f   # tail logs to confirm startup
```

### F.2 Editing the frontend / Streamlit app

Any edit to `app/app.py` requires a service restart (Streamlit's auto-reload is disabled by `--server.headless true`):

```bash
sudo systemctl restart ecg-demo
```

Hard-refresh the browser (Cmd+Shift+R / Ctrl+F5) to bust any cached assets.

### F.3 Tailing logs

```bash
sudo journalctl -u ecg-demo -f             # live tail
sudo journalctl -u ecg-demo --since "10 minutes ago"
sudo journalctl -u ecg-demo --grep "Traceback"   # error hunting
```

### F.4 Stop / start the instance (cost savings)

When you're not demoing, stop the EC2 to drop hourly charges. The EBS volume keeps the repo + model checkpoint on disk.

```bash
# On laptop:
aws ec2 stop-instances --instance-ids i-<your-instance-id>
# Later, to bring it back:
aws ec2 start-instances --instance-ids i-<your-instance-id>
# Note: the private IP may change after a stop/start unless an Elastic IP is attached.
```

The systemd unit auto-starts on boot, so the demo is back online within ~1 minute of `start-instances`.

### F.5 Rollback to a previous commit

```bash
cd ~/ECGFounder
git log --oneline -10            # find the SHA you want
git checkout <sha>
sudo systemctl restart ecg-demo
```

To return to mainline:

```bash
git checkout main && git pull
sudo systemctl restart ecg-demo
```

---

## Troubleshooting

### `systemctl status ecg-demo` shows `failed`

```bash
sudo journalctl -u ecg-demo -n 100 --no-pager | tail -60
```

Common causes:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` | pip install didn't run on this box | `pip3 install streamlit` |
| `ModuleNotFoundError: dual_head_ecgfounder` | systemd's `WorkingDirectory` or `PYTHONPATH` wrong | check the unit file matches D.1 |
| `OSError: address already in use` | port 8000 already held by another process | `sudo lsof -iTCP:8000`; kill the squatter |
| `FileNotFoundError: checkpoint/1_lead_ECGFounder.pth` | model not downloaded | re-run C.4 |
| `[Errno 28] No space left on device` | EBS full | check `df -h`; expand the EBS volume or prune logs |

### Browser shows "site can't be reached"

In order, check:

1. Are you on the VPN? (`ping <ec2-host>` should succeed)
2. Is the security group inbound rule allowing port 8000 from your VPN CIDR? Re-check in EC2 console.
3. Is the EC2 instance running? (`aws ec2 describe-instances --instance-ids i-... --query 'Reservations[].Instances[].State.Name'`)
4. Is systemd up? (SSH in and re-run E.1–E.3 above)

### Inference is slow (>3 s per request)

t3.large is CPU-only; ~500 ms–1 s is normal. If you see >3 s consistently:

- Check `top` while a request is in flight — Python should be ~100% CPU on one core; if it's lower, something is starving (e.g. parallel pip install? something else on the box?).
- Confirm the model is using `cpu` and not falling back to a slow path: `journalctl -u ecg-demo | grep device:`
- If you need consistent sub-500 ms inference, upgrade to `g4dn.xlarge` (Nvidia T4 GPU, ~$0.53/hr). Update Phase B.1 instance type and re-do Phase C from scratch (Nvidia driver + CUDA libs are extra steps — not covered here, see AWS deep-learning AMI for that path).

### Model-load OOM mid-request

Should not happen on t3.large per the budget in [STREAMLIT_APP.md §12](STREAMLIT_APP.md#12-sizing) (~3.6 GB worst case vs 8 GB available). If it does:

```bash
sudo journalctl -k --grep "killed process"   # check the dmesg / kernel log
```

If a process named `python3` was OOM-killed: someone else is using the box, OR a memory leak landed in a recent code change. Restart the service (`systemctl restart ecg-demo`) as a temporary fix; bisect the offending commit.

---

## Cost summary

Steady-state running cost for one always-on demo box in `us-east-1`:

| Item | Cost |
|---|---|
| `t3.large` compute, 24×7 | ~$0.083/hr × 730 hr = **$60.59/month** |
| 15 GB gp3 EBS | ~$1.50/month |
| Data transfer (internal VPN) | $0 (private subnet) |
| **Total, always-on** | **~$62/month** |

Cost-saving lever: stop the instance between demos (F.4). Storage-only when stopped = ~$1.50/month. **Stopping and starting via `aws ec2` CLI is free.**

For ad-hoc demo schedules (e.g. 4 hours/week): ~$2.50/month with discipline around `stop-instances` after each demo.

---

## What's NOT in this guide (and where to read about it)

| Concern | Where it's covered |
|---|---|
| Application code structure / API contract / etc. | [STREAMLIT_APP.md](STREAMLIT_APP.md) — the design doc |
| Why Streamlit instead of FastAPI | [README.md "Why Streamlit"](README.md) + [_archived/README.md](_archived/README.md) |
| What to do if external/customer demos become a requirement | [README.md "Out of scope"](README.md#out-of-scope--when-to-not-use-this-stack) |
| Real PHI/HIPAA path | not here — schedule a security review |
| HTTPS / TLS termination | not here — internal HTTP only |
| Auto-scaling, multi-AZ | not here — single-instance MVP |
| Model retraining pipeline | [scripts/finetune_scope_linprobe.py](../../scripts/finetune_scope_linprobe.py) and surrounding fine-tune docs |

---

## Checklist for the impatient

If you've done this once before and just need a checklist:

```bash
# Local
ssh-keygen -t ed25519 -f ~/.ssh/ecg-demo
aws ec2 import-key-pair --key-name ecg-demo --public-key-material fileb://~/.ssh/ecg-demo.pub
# Launch t3.large Ubuntu 22.04 with security group allowing port 8000 from VPN CIDR

# EC2 (one-time)
sudo apt update && sudo apt install -y python3-pip git
git clone <repo> && cd ECGFounder
pip3 install -r requirements.txt streamlit
python3 -c "from checkpoints import load_ecgfounder; from device_utils import resolve_device; load_ecgfounder(resolve_device())"
sudo cp deploy/ecg-demo.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ecg-demo

# Verify
curl -sI http://localhost:8000/
```

If all of that produces `HTTP/1.1 200 OK`, the demo is live at `http://<ec2-host>:8000/` from your VPN.
