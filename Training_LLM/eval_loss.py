from __future__ import annotations
import argparse, torch
from dataset import TextDataset, make_dataloaders
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Prepare validation DataLoader
    val_loader, _ = make_dataloaders(args.data, args.block_size, args.batch_size)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    losses = []
    with torch.no_grad():
        it = iter(val_loader)
        for _ in range(args.iters):
            try:
                xb, yb = next(it)
            except StopIteration:
                # Restart iterator if exhausted before reaching iters
                it = iter(val_loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()