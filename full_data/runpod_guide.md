# Running SPD decomposition (`config_full_ha.yaml`) on RunPod

Target: 8× H100, ~48h.

## 1. Launch the RunPod instance

1. Sign in at runpod.io → **Deploy** → **Pods** → **GPU Pods**.
2. Filter by `H100 SXM` (or `H100 PCIe`) and pick a host with **8 GPUs available on one node**. For ~48h use **On-Demand** (Spot risks losing the pod mid-run).
3. Template: pick a recent **PyTorch** template (e.g. "RunPod PyTorch 2.4 / CUDA 12.4"). Includes CUDA + drivers + SSH.
4. Disk: **container disk ≥ 100 GB** and **volume ≥ 200 GB** mounted at `/workspace` (checkpoints + HF tokenized dataset cache are sizeable).
5. Expose SSH (TCP 22) — RunPod does this via its public TCP port mapping.
6. Deploy and wait until status is `Running`.

## 2. Connect

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/your_runpod_key
```

(Or use `runpodctl` / the web terminal.) Open a `tmux` session immediately so the job survives disconnects:

```bash
tmux new -s spd
```

## 3. Clone the repo and install

```bash
cd /workspace
git clone -b targeted_decomposition https://github.com/Antovigo/spd.git
cd spd
# Repo uses uv; install it if not present:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync        # creates .venv with all deps
```

## 4. Set up credentials

Create `/workspace/spd/.env`:

```
WANDB_API_KEY=<your key>
WANDB_ENTITY=<your entity>
```

Also `huggingface-cli login` if needed (the dataset `danbraunai/pile-uncopyrighted-tok-shuffled` is public, but login avoids rate-limit surprises).

## 5. Copy the config

```bash
mkdir -p /workspace/configs
# From your laptop:
# scp -P <port> config_full_ha.yaml root@<pod-ip>:/workspace/configs/
```

The pretrained target `goodfire/spd/runs/t-9d2b8f02` is fetched from WandB on first use — nothing to upload.

## 6. Sanity check (5-min dry run)

Before committing 48h, make a copy with `steps: 1000` and run on 1 GPU to catch config / auth / dataset issues:

```bash
cd /workspace/spd
uv run python spd/experiments/lm/lm_decomposition.py /workspace/configs/config_full_ha_smoke.yaml
```

Verify: WandB run appears, training loss decreases, no OOM.

## 7. Launch the full run (8-GPU DDP)

```bash
cd /workspace/spd
uv run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node 8 \
  spd/experiments/lm/lm_decomposition.py \
  /workspace/configs/config_full_ha.yaml \
  2>&1 | tee /workspace/run.log
```

Note: the config has `batch_size: 64`. With DDP, torchrun launches 8 processes and the effective batch becomes `8 × 64 = 512` unless the script divides by world size — confirm in early WandB steps that the effective batch / step-count matches expectations. To hold the effective batch at 64, set `batch_size: 8` in a copy of the config.

## 8. Monitor

- WandB dashboard for loss curves.
- `nvidia-smi dmon -s u` on the pod to confirm all 8 GPUs stay near 100%.
- `du -sh ~/spd_out/` occasionally — checkpoints land there.

## 9. When finished

- Final model + config are uploaded to WandB automatically.
- Optional post-processing: `uv run spd-harvest <wandb_path> --n_gpus 8` then `spd-attributions`.
- **Stop the pod** from the RunPod UI (you're billed while it runs, even idle).
- To pull local checkpoints: `scp -P <port> -r root@<pod-ip>:~/spd_out/runs/<run_id> ./`

## Cost sanity check

8× H100 SXM on RunPod is roughly $20–30/hr → **~$1k–$1.5k for 48h**. Make sure credit covers it plus a buffer.
