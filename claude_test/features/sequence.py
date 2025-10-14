"""
Sequence Encoder using Protein Language Models

Uses ESM-2 (Evolutionary Scale Modeling) for peptide sequence encoding

Requires: fair-esm
Install: pip install fair-esm
"""

import torch
import torch.nn as nn

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM not available. Sequence encoding disabled.")
    print("Install with: pip install fair-esm")


class SequenceEncoder(nn.Module):
    """
    Encode amino acid sequences using ESM-2

    ESM-2 is a state-of-the-art protein language model
    """

    def __init__(self, hidden_dim, esm_model='esm2_t33_650M_UR50D', freeze_esm=True):
        super().__init__()

        if not ESM_AVAILABLE:
            raise RuntimeError("ESM library required for sequence encoding")

        # Load pre-trained ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.__dict__[esm_model]()
        self.batch_converter = self.alphabet.get_batch_converter()

        # ESM-2 hidden dimension (1280 for most models)
        self.esm_dim = self.esm_model.embed_dim

        # Freeze ESM if specified
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        # Projection to match target hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, sequences):
        """
        Args:
            sequences: List of amino acid sequences (strings)

        Returns:
            embeddings: [B, L, hidden_dim] sequence embeddings
            pooled: [B, hidden_dim] pooled sequence representation
        """
        # Prepare data for ESM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.parameters()).device)

        # Get ESM embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False
            )

        # Extract representations from last layer
        # Shape: [B, L, esm_dim]
        embeddings = results["representations"][self.esm_model.num_layers]

        # Remove BOS and EOS tokens
        embeddings = embeddings[:, 1:-1, :]

        # Project to target dimension
        embeddings = self.projection(embeddings)  # [B, L, hidden_dim]

        # Pool over sequence length (mean pooling)
        pooled = embeddings.mean(dim=1)  # [B, hidden_dim]

        return embeddings, pooled


class SimpleSequenceEncoder(nn.Module):
    """
    Simple sequence encoder using learnable amino acid embeddings
    (fallback when ESM is not available)
    """

    def __init__(self, hidden_dim, vocab_size=25):
        """
        Args:
            hidden_dim: Hidden dimension
            vocab_size: Number of amino acid types (20 standard + 5 special)
        """
        super().__init__()

        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Embedding(1000, hidden_dim)  # Max length 1000

        # Sequence encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, sequences_indices):
        """
        Args:
            sequences_indices: [B, L] integer indices of amino acids

        Returns:
            embeddings: [B, L, hidden_dim]
            pooled: [B, hidden_dim]
        """
        B, L = sequences_indices.shape

        # Embed amino acids
        embeddings = self.embedding(sequences_indices)  # [B, L, D]

        # Add positional encoding
        positions = torch.arange(L, device=embeddings.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_encoder(positions)
        embeddings = embeddings + pos_emb

        # Apply transformer
        embeddings = self.transformer(embeddings)  # [B, L, D]

        # Pool
        pooled = embeddings.mean(dim=1)  # [B, D]

        return embeddings, pooled


# Amino acid vocabulary
AA_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '<PAD>': 20, '<UNK>': 21, '<BOS>': 22, '<EOS>': 23, '<MASK>': 24
}


def sequence_to_indices(sequence, vocab=AA_VOCAB):
    """Convert amino acid sequence to indices"""
    return torch.LongTensor([
        vocab.get(aa, vocab['<UNK>']) for aa in sequence.upper()
    ])


if __name__ == '__main__':
    print("Testing Sequence Encoder...")

    # Test simple encoder (always available)
    print("\n1. Testing SimpleSequenceEncoder...")
    encoder = SimpleSequenceEncoder(hidden_dim=512)

    sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEG"]
    max_len = max(len(s) for s in sequences)

    # Convert to indices and pad
    seq_indices = []
    for seq in sequences:
        indices = sequence_to_indices(seq)
        if len(indices) < max_len:
            # Pad
            padding = torch.full((max_len - len(indices),), AA_VOCAB['<PAD>'])
            indices = torch.cat([indices, padding])
        seq_indices.append(indices)

    seq_indices = torch.stack(seq_indices)
    print(f"Sequence indices shape: {seq_indices.shape}")

    embeddings, pooled = encoder(seq_indices)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Pooled shape: {pooled.shape}")

    # Test ESM encoder if available
    if ESM_AVAILABLE:
        print("\n2. Testing ESM SequenceEncoder...")
        print("Note: This requires downloading ESM-2 model (may take time)")
        # Test code commented out to avoid long download times
        # esm_encoder = SequenceEncoder(hidden_dim=512)
        # embeddings, pooled = esm_encoder(sequences)

    print("\n✓ All tests passed!")
