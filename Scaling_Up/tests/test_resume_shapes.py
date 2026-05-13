import torch
import torch.nn as nn
from pathlib import Path
import sys
import pytest

# Add paths for Scaling_Up and modernizing_architecture
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'Scaling_Up'))
sys.path.append(str(ROOT / 'modernizing_architecture'))

from checkpointing import save_checkpoint, load_checkpoint
from model_modern import GPTModern

def test_resume_architecture_mismatch(tmp_path):
    """Verifies that loading a checkpoint with a different architecture fails."""
    cfg = dict(
        vocab_size=100, n_layer=2, n_head=4, n_embd=64, 
        block_size=128, dropout=0.0
    )
    model = GPTModern(**cfg)
    
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_checkpoint(model, None, None, None, step=10, out_dir=str(ckpt_dir), config=cfg)
    
    # Try loading into different architecture (mismatched n_layer)
    cfg_bad = cfg.copy()
    cfg_bad['n_layer'] = 3
    model_bad = GPTModern(**cfg_bad)
    
    with pytest.raises(RuntimeError, match="Architecture mismatch"):
        load_checkpoint(model_bad, str(ckpt_dir / "model_last.pt"))

def test_resume_success(tmp_path):
    """Verifies that a checkpoint can be successfully reloaded with matching config."""
    cfg = dict(
        vocab_size=100, n_layer=2, n_head=4, n_embd=64, 
        block_size=128, dropout=0.0
    )
    model = GPTModern(**cfg)
    
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_checkpoint(model, None, None, None, step=42, out_dir=str(ckpt_dir), config=cfg)
    
    # Create a fresh model with the same config
    model2 = GPTModern(**cfg)
    
    # Ensure weights are different before load
    with torch.no_grad():
        model2.tok_emb.weight.fill_(0.0)
    
    step = load_checkpoint(model2, str(ckpt_dir / "model_last.pt"))
    
    assert step == 42
    # Verify weights were loaded
    assert torch.allclose(model.tok_emb.weight, model2.tok_emb.weight)

def test_resume_optimizer_state(tmp_path):
    """Verifies that optimizer state is also restored."""
    cfg = dict(
        vocab_size=100, n_layer=1, n_head=2, n_embd=32, 
        block_size=64, dropout=0.0
    )
    model = GPTModern(**cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Fake some optimizer state
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state[p] = {'step': torch.tensor(5.0)}
            
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_checkpoint(model, optimizer, None, None, step=5, out_dir=str(ckpt_dir), config=cfg)
    
    # Reload
    model2 = GPTModern(**cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    load_checkpoint(model2, str(ckpt_dir / "model_last.pt"), optimizer=optimizer2)
    
    # Check if state was restored
    for group in optimizer2.param_groups:
        for p in group['params']:
            assert optimizer2.state[p]['step'] == 5
