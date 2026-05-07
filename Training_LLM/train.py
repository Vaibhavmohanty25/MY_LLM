from __future__ import annotations
import argparse, math, pathlib, time
import torch
from torch.utils.data import DataLoader

from tokenizer import ByteTokenizer
from model_gpt import GPT
from dataset import TextDataset, make_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny GPT on a text file.")
    p.add_argument("--data",          type=str,   required=True,  help="Path to .txt training file")
    p.add_argument("--steps",         type=int,   default=500,    help="Total training steps")
    p.add_argument("--eval_interval", type=int,   default=100,    help="Evaluate val loss every N steps")
    p.add_argument("--sample_every",  type=int,   default=100,    help="Print sample text every N steps")
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--block_size",    type=int,   default=128)
    p.add_argument("--n_layer",       type=int,   default=2)
    p.add_argument("--n_head",        type=int,   default=2)
    p.add_argument("--n_embd",        type=int,   default=128)
    p.add_argument("--lr",            type=float, default=3e-4,   help="Learning rate")
    p.add_argument("--out_dir",       type=str,   default="runs/min-gpt")
    p.add_argument("--cpu",           action="store_true")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=20):
    model.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def sample_text(model, tok, device, prompt="", max_new_tokens=80):
    model.eval()
    if prompt:
        ids = tok.encode(prompt).unsqueeze(0).to(device)
    else:
        ids = torch.tensor([[10]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
    model.train()
    return tok.decode(out[0].cpu())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Resolve data path relative to this script's directory
    script_dir = pathlib.Path(__file__).resolve().parent
    data_path  = script_dir / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = script_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteTokenizer()
    train_loader, val_loader = make_dataloaders(
        str(data_path), args.block_size, args.batch_size
    )

    model_config = dict(
        vocab_size  = tok.vocab_size,
        block_size  = args.block_size,
        n_layer     = args.n_layer,
        n_head      = args.n_head,
        n_embd      = args.n_embd,
    )
    model = GPT(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Infinite cycling iterator over the train loader
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- Periodic evaluation ----
        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_loss(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"step {step:4d}/{args.steps} | train_loss {loss.item():.4f} | "
                  f"val_loss {val_loss:.4f} | {elapsed:.1f}s")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {"model": model.state_dict(), "config": model_config, "step": step}
                torch.save(ckpt, out_dir / "model_best.pt")
                print(f"  ✔ saved best model (val_loss={val_loss:.4f})")

        # ---- Periodic sampling ----
        if step % args.sample_every == 0 or step == args.steps:
            sample = sample_text(model, tok, device, prompt="", max_new_tokens=80)
            print(f"\n--- Sample at step {step} ---\n{sample}\n{'---'*10}")

    # Always save the final checkpoint too
    ckpt = {"model": model.state_dict(), "config": model_config, "step": args.steps}
    torch.save(ckpt, out_dir / "model_final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
