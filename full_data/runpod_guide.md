# Running SPD decomposition (`config_full_ha.yaml`) on RunPod

Target: 8× H100, ~48h.

## 1. Generate an SSH key (once, on your laptop)

RunPod uses SSH public keys for pod access. If you don't already have one dedicated to RunPod:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/runpod_key -C "runpod"
# enter a passphrase — disk read alone won't yield a usable key
```

This gives you two files:

- `~/.ssh/runpod_key` — **private** key, stays on your laptop. Passed to `ssh -i`.
- `~/.ssh/runpod_key.pub` — **public** key, paste into RunPod.

In the RunPod dashboard: **Settings → SSH Public Keys → + Add Key** and paste the contents of `runpod_key.pub`.

Security notes:
- Keep full-disk encryption on (LUKS / FileVault / BitLocker). A passphrase protects the key file itself; FDE protects the whole machine when stolen or powered off.
- Use `ssh-agent` so you only type the passphrase once per session:
  ```bash
  eval "$(ssh-agent)"
  ssh-add ~/.ssh/runpod_key
  ```

## 2. Launch the pod

1. **Deploy → Pods → GPU Pods**.
2. Filter by `H100 SXM` (or `H100 PCIe` — SXM has NVLink so DDP all-reduce is much faster; PCIe works but adds ~20% to step time). Pick a host with **8 GPUs on one node**.
3. Pricing: **On-Demand** (Spot is ~40% cheaper but can be evicted mid-run; for a 48h single-training-job, not worth the risk).
4. Template: a recent **PyTorch** template (e.g. "RunPod PyTorch 2.4 / CUDA 12.4").
5. Customize deployment:
   - **Container disk**: 100 GB (disposable, used for OS + caches).
   - **Volume disk**: 300 GB, mounted at `/workspace`. Persists across pod stop/start on the same pod.
   - Expose TCP port 22 for SSH.
6. Deploy. Wait for status `Running`.

Trade-off of a pod volume (vs a network volume): if the pod is **terminated** (not stopped), the volume is deleted. Use **Stop**, not **Terminate**, if you need to pause. Checkpoints are synced to WandB by default (see step 11), so WandB is the off-pod safety net.

## 3. Connect

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519
```

Open a `tmux` session immediately so the job survives disconnects:

```
apt-get install -y tmux                                                                         `````

```bash
tmux new -s spd
```

How to disconnect:
Ctrl+B  D            # detach, close browser, go to bed

# next morning, on a fresh browser:
tmux attach -t spd   # see current loss, GPU utilization
Ctrl+B  D            # detach again

Verify:

```bash
df -h /workspace     # should show ~300 GB available
nvidia-smi           # should list 8 H100s
```

## 4. Clone the repo and install

```bash
cd /workspace
git clone -b targeted_decomposition https://github.com/Antovigo/spd.git
cd spd
# Repo uses uv; install it if not present:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync        # creates .venv with all deps inside /workspace/spd/.venv
```

## 5. Point caches and outputs at `/workspace`

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export SPD_OUT_DIR=/workspace/spd_out
EOF
source ~/.bashrc
mkdir -p /workspace/hf_cache /workspace/spd_out
```

Now streamed HF shards, the tokenizer, the pretrained model download, and SPD checkpoints all land on the pod volume instead of the container disk.

## 6. Set up credentials

Create `/workspace/spd/.env`:

```
WANDB_API_KEY=<your key>
WANDB_ENTITY=<your entity>
```

For HuggingFace (public datasets don't strictly need it, but login avoids rate-limit surprises):

```bash
huggingface-cli login     # paste your HF token
```

Log in to wandb (from the /workspace/spd folder):

```
uv run python -c "
import wandb
api = wandb.Api()
run = api.run('goodfire/spd/t-9d2b8f02')
print(run.name, run.state)
for f in run.files():
print(f.name, f.size)
"
```

## 7. Copy the configs

```bash
mkdir -p /workspace/configs
# From your laptop:
# scp -P <port> -i ~/.ssh/runpod_key \
#   config_full_ha.yaml config_full_ha_smoke.yaml config_full_ha_10x.yaml config_full_ha_10x_smoke.yaml \
#   root@<pod-ip>:/workspace/configs/
```

The pretrained target `goodfire/spd/runs/t-9d2b8f02` is fetched from WandB on first use — nothing to upload.

## 8. Sanity check (smoke run, 1 GPU)

Before committing 48h, run the smoke config on 1 GPU to catch config / auth / dataset issues:

```bash
cd /workspace/spd
uv run python spd/experiments/lm/lm_decomposition.py /workspace/configs/config_full_ha_smoke.yaml
```

Verify: WandB run appears, training loss decreases, `slow_eval` fires at step 500, a checkpoint lands in `/workspace/spd_out/runs/<run_id>/`, and the same checkpoint shows up as a WandB artifact. No OOM.

## 9. Launch the full run (8-GPU DDP)

```bash
cd /workspace/spd
uv run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node 8 \
  spd/experiments/lm/lm_decomposition.py \
  /workspace/configs/config_full_ha_10x.yaml \
  2>&1 | tee /workspace/run.log
```

Note: the config has `batch_size: 64`. With DDP, torchrun launches 8 processes and the effective batch becomes `8 × 64 = 512` unless the script divides by world size — confirm in early WandB steps that the effective batch / step-count matches expectations. To hold the effective batch at 64, set `batch_size: 8` in a copy of the config.

## 10. Monitor

- WandB dashboard for loss curves.
- `nvidia-smi dmon -s u` — all 8 GPUs should stay near 100%. If they don't, streaming bandwidth is likely the bottleneck.
- `du -sh /workspace/spd_out/` — checkpoints land there, saved every 40 000 steps (≈ 10 checkpoints over the full run).
- `df -h /workspace` — keep an eye on remaining space; 400 GB dataset streams ~50 GB of cached shards, checkpoints add more.

## 11. Off-pod backup of checkpoints

Since there's no network volume, a pod termination would wipe everything in `/workspace`. Two safety nets:

1. **WandB artifact sync (primary).** The config's `sync_checkpoints_to_wandb` defaults to `True`, so every saved checkpoint is uploaded to WandB. Confirm this after the first `save_freq` step (40 000) — you should see a `.pth` artifact appear in the WandB run.

2. **Laptop rsync (secondary).** From your laptop, pull checkpoints as they land:
   ```bash
   rsync -avz --partial -e "ssh -p <port> -i ~/.ssh/runpod_key" \
     root@<pod-ip>:/workspace/spd_out/runs/<run_id>/ \
     ~/spd_backup/<run_id>/
   ```
   Re-run every few hours (or throw it in a loop).

If needed, install rsync on the pod:
```
apt-get install -y rsync
```

## 12. Recovery if the pod dies

If the pod is terminated mid-run:

1. Launch a new pod (repeat steps 2–7).
2. Download the latest checkpoint from WandB:
   ```python
   # on the new pod
   from spd.models.component_model import ComponentModel
   model = ComponentModel.from_pretrained("wandb:<entity>/spd/runs/<run_id>")
   ```
   Or fetch the raw `.pth` via `wandb artifact get`.
3. Relaunch pointing at the checkpoint. Add to the config:
   ```yaml
   init_spd_checkpoint: /path/to/ckpt_step_<N>.pth
   ```
   Check `spd/run_spd.py` for whether it auto-skips completed steps; otherwise reduce `steps` manually.

At worst you lose ~5h of progress (one `save_freq` window).

## 13. When finished

- Final model + config are uploaded to WandB automatically.
- Optional post-processing: `uv run spd-harvest <wandb_path> --n_gpus 8` then `spd-attributions`.
- To pull local checkpoints to your laptop:
  ```bash
  scp -P <port> -i ~/.ssh/runpod_key -r \
    root@<pod-ip>:/workspace/spd_out/runs/<run_id> ./
  ```
- **Terminate the pod** from the RunPod UI (this deletes the pod volume too — do the scp first!).

## Cost sanity check

8× H100 SXM on RunPod is roughly $20–30/hr → **~$1k–$1.5k for 48h**. Make sure credit covers it plus a buffer for smoke runs and post-processing.
