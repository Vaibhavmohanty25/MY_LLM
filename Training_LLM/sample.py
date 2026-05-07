from __future__ import annotations
import argparse, torch
from tokenizer import ByteTokenizer
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', nargs='*', default=[], help='Prompt string (optional).')
    p.add_argument('--tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    # Determine device (CPU or CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    # Initialize tokenizer
    tok = ByteTokenizer()
    # Join prompt parts into a single string (handles unquoted spaces on Windows)
    prompt_str = ' '.join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    # Tokenise the prompt
    prompt_ids = tok.encode(prompt_str).unsqueeze(0).to(device)
    if prompt_ids.numel() == 0:
        # If no prompt provided, seed with newline byte (10)
        prompt_ids = torch.tensor([[10]], dtype=torch.long, device=device)


    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt.get('config', None)

    if config is None:
        # fall back to defaults
        model = GPT(tok.vocab_size, block_size=256).to(device)
        model.load_state_dict(ckpt['model'])
    else:
        model = GPT(**config).to(device)
        model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(tok.decode(out[0].cpu()))


if __name__ == '__main__':
    main()