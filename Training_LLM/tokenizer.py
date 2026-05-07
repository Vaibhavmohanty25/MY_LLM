import torch

class ByteTokenizer:
    """
    A simple byte-level tokenizer that maps characters to their UTF-8 byte values (0-255).
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a string into a tensor of byte values.
        """
        # Convert string to bytes, then to list of integers
        byte_list = list(text.encode('utf-8'))
        return torch.tensor(byte_list, dtype=torch.long)

    def decode(self, token_ids) -> str:
        """
        Decodes a list or tensor of byte values back into a string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Convert list of integers back to bytes, then decode to string
        return bytes(token_ids).decode('utf-8', errors='replace')
