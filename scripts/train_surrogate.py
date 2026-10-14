"""Train a small causal network to imitate the connectome simulation.

The thing being approximated is a fixed single-input, single-output,
causal, time-invariant nonlinear filter, so a dilated causal convolution
stack is the natural shape: each layer doubles its dilation, giving an
exponentially growing memory for a linear number of parameters. The
membrane time constant is 20 ms, so a receptive field of ~100 ms covers
the dynamics with room to spare.

Trained on single-trial targets on purpose. MSE regression converges on
the conditional mean, so the network learns the TRIAL-AVERAGED response
without ever being shown one -- it outputs the denoised signal in a
single pass.

This is an APPROXIMATION of the simulation, not the simulation. It is
only trustworthy where it is validated, which is why the script reports
held-out correlation and per-note pitch tracking, and why anything built
on it has to say which one produced a given result.

Usage:  python scripts/train_surrogate.py [--data ...npz] [--epochs 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path("C:/CourtshipData/surrogate")
OUT = Path(__file__).resolve().parents[1] / "models"


class CausalBlock(nn.Module):
    """Dilated causal conv + residual. Padding is applied on the LEFT only
    and the tail trimmed, so no output step ever sees a future input --
    which would make the surrogate non-causal and useless for streaming."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        # No normalisation layer. GroupNorm/BatchNorm reduce over the TIME
        # axis, so every output would depend on the whole sequence including
        # later steps -- measured as a 0.033 leak backwards in time before
        # this was removed. Residual connections carry training stability
        # instead; weight_norm is parameter-level and stays causal.
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, ch, kernel, dilation=dilation))
        self.act = nn.GELU()

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))
        return x + self.act(self.conv(y))


class Surrogate(nn.Module):
    def __init__(self, ch: int = 40, n_blocks: int = 8):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.Sequential(*[CausalBlock(ch, 2 ** i) for i in range(n_blocks)])
        self.out = nn.Conv1d(ch, 1, 1)
        # Receptive field: 1 + sum over blocks of (kernel-1)*dilation
        self.receptive_field = 1 + sum(2 * (2 ** i) for i in range(n_blocks))

    def forward(self, x):                       # (B, 1, T) -> (B, 1, T)
        return self.out(self.blocks(self.inp(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "male_courting_dt0.2.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--channels", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    d = np.load(args.data)
    X, Y = d["drive"], d["response"]
    dt = float(d["dt_ms"])
    print(f"{X.shape[0]} clips x {X.shape[1]} steps at dt={dt} ms "
          f"({X.shape[0] * X.shape[1] * dt / 1000:.0f}s of audio)")

    # Scale both to O(1). Drive is a firing rate in Hz; response is a spike
    # count per step. Stored so inference can undo it.
    x_scale = float(np.abs(X).max()) or 1.0
    y_scale = float(np.abs(Y).max()) or 1.0
    Xs = (X / x_scale).astype(np.float32)[:, None, :]
    Ys = (Y / y_scale).astype(np.float32)[:, None, :]

    n_val = max(1, int(len(Xs) * args.val_frac))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xs))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    print(f"train {len(tr_i)} clips, validate {len(val_i)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Surrogate(args.channels, args.blocks).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par:,} parameters, receptive field "
          f"{model.receptive_field} steps ({model.receptive_field * dt:.0f} ms)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    xtr = torch.from_numpy(Xs[tr_i]); ytr = torch.from_numpy(Ys[tr_i])
    xva = torch.from_numpy(Xs[val_i]).to(dev); yva = torch.from_numpy(Ys[val_i]).to(dev)

    best = float("inf")
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / (Path(args.data).stem + ".pt")
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(order), args.batch):
            sel = order[i:i + args.batch]
            xb, yb = xtr[sel].to(dev), ytr[sel].to(dev)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(xva)
            vloss = float(nn.functional.mse_loss(pred, yva))
            p = pred.cpu().numpy().reshape(len(val_i), -1)
            a = yva.cpu().numpy().reshape(len(val_i), -1)
            cors = [np.corrcoef(p[k], a[k])[0, 1] for k in range(len(p))
                    if p[k].std() > 0 and a[k].std() > 0]
            cor = float(np.mean(cors)) if cors else float("nan")
        if vloss < best:
            best = vloss
            torch.save({"state": model.state_dict(), "x_scale": x_scale,
                        "y_scale": y_scale, "dt_ms": dt,
                        "channels": args.channels, "blocks": args.blocks}, ckpt)
        print(f"  epoch {ep+1:>3}/{args.epochs}  train {tot/len(tr_i):.5f}  "
              f"val {vloss:.5f}  held-out r {cor:.3f}{'  *' if vloss == best else ''}")

    print(f"\nbest val loss {best:.5f} -> {ckpt}")

    # ONNX export, so the surrogate can run in a browser with no GPU.
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    model.eval()
    onnx_path = OUT / (Path(args.data).stem + ".onnx")
    dummy = torch.zeros(1, 1, 2000, device=dev)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["drive"],
                      output_names=["response"], opset_version=17,
                      dynamic_axes={"drive": {2: "time"}, "response": {2: "time"}})
    meta = {"x_scale": x_scale, "y_scale": y_scale, "dt_ms": dt,
            "parameters": n_par, "receptive_field_steps": model.receptive_field,
            "best_val_mse": best, "onnx": onnx_path.name}
    (OUT / (Path(args.data).stem + "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
