import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """
    Class for input embedding.

    Args:
        d_model (int): Dimensionality of the model.
        source_vocab_size (int): Size of the source vocabulary.
    """

    def __init__(self, d_model: int, source_vocab_size: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.source_vocab_size = source_vocab_size
        self.embedding = nn.Embedding(source_vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)


class PosEmbedding(nn.Module):
    """
    Class for positional embedding.

    Args:
        d_model (int): Dimensionality of the model.
        max_len (int): Maximum length of sequences.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        pos_enc = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1) #! (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc.unsqueeze(0) #! (1, max_len, d_model)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return self.dropout(x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False))


class LayerNorm(nn.Module):
    """
    Class for layer normalization.
    """

    def __init__(self, epsilon=1e-6) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class FeedForward(nn.Module):
    """
    Class for the feed forward neural network.

    Args:
        d_model (int): Dimensionality of the model.
        hidden_size_ff (int): Hidden size of the feed forward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, hidden_size_ff: int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, hidden_size_ff)
        self.linear2 = nn.Linear(hidden_size_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """
    Class for multi-head attention mechanism.

    Args:
        d_model (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0, 'd_model must be divisible by heads'

        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout = None, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            dropout (nn.Dropout): Dropout layer.
            mask (torch.Tensor): Mask tensor for attention.

        Returns:
            torch.Tensor: Attention scores and weighted sum of values.
        """

        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) #! (batch, h, max_len, max_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores


    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        b, s, _ = query.shape #! (batch, max_len, d_model)

        query = query.view(b, s, self.heads, self.d_k).transpose(1, 2) #! (batch, heads, max_len, d_k)
        key = key.view(b, s, self.heads, self.d_k).transpose(1, 2) #! (batch, heads, max_len, d_k)
        value = value.view(b, s, self.heads, self.d_k).transpose(1, 2) #! (batch, heads, max_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.dropout, mask) #! x -> (batch, h, max_len, d_k)
        
        #! (batch, h, max_len, d_k) --> (batch, max_len, h, d_k) --> (batch, max_len, d_model)
        return self.w_o(x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model))


class SkipConnection(nn.Module):
    """
    Class for skip connection with layer normalization.

    Args:
        dropout (float): Dropout rate.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Class for an encoder block in the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        hidden_size_ff (int): Hidden size of the feed forward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, heads: int, hidden_size_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feedforward = FeedForward(d_model, hidden_size_ff, dropout)
        self.skip_connection1 = SkipConnection(dropout)
        self.skip_connection2 = SkipConnection(dropout)

    def forward(self, x, enc_mask):
        x = self.skip_connection1(x, lambda x: self.self_attention(x, x, x, mask=enc_mask))
        return self.skip_connection2(x, self.feedforward)


class Encoder(nn.Module):
    """
    Class for the encoder in the Transformer architecture.

    Args:
        num_layers (int): Number of encoder blocks.
        d_model (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        hidden_size_ff (int): Hidden size of the feed forward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, num_layers: int, d_model: int, heads: int, hidden_size_ff: int, dropout: float) -> None:
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock(d_model, heads, hidden_size_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm()

    def forward(self, x, enc_mask=None):
        for layer in range(self.layers):
            x = layer(x, enc_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Class for a decoder block in the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        hidden_size_ff (int): Hidden size of the feed forward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, heads: int, hidden_size_ff: int, dropout: float) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feedforward = FeedForward(d_model, hidden_size_ff, dropout)
        self.skip_connection1 = SkipConnection(dropout)
        self.skip_connection2 = SkipConnection(dropout)
        self.skip_connection3 = SkipConnection(dropout)

    def forward(self, x, bottleneck, enc_mask=None, dec_mask=None):
        x = self.skip_connection1(x, lambda x: self.self_attention(x, x, x, mask=dec_mask))
        x = self.skip_connection2(x, lambda x: self.cross_attention(x, bottleneck, bottleneck, mask=enc_mask))
        return self.skip_connection3(x, self.feedforward)


class Decoder(nn.Module):
    """
    Class for the decoder in the Transformer architecture.

    Args:
        num_layers (int): Number of decoder blocks.
        d_model (int): Dimensionality of the model.
        heads (int): Number of attention heads.
        hidden_size_ff (int): Hidden size of the feed forward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, num_layers: int, d_model: int, heads: int, hidden_size_ff: int, dropout: float) -> None:
        super().__init__()

        self.layers = nn.ModuleList([DecoderBlock(d_model, heads, hidden_size_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm()

    def forward(self, x, bottleneck, enc_mask=None, dec_mask=None):
        for layer in range(self.layers):
            x = layer(x, bottleneck, enc_mask, dec_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Class for the projection layer in the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        target_vocab_size (int): Size of the target vocabulary.
    """

    def __init__(self, d_model: int, target_vocab_size: int) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, target_vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Seq2SeqTransformer(nn.Module):
    """
    Sequence-to-Sequence Transformer model for machine translation.

    Args:
        source_vocab_size (int): Size of the source vocabulary.
        target_vocab_size (int): Size of the target vocabulary.
        source_max_len (int): Maximum length of the source sequence.
        target_max_len (int): Maximum length of the target sequence.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        num_layers (int, optional): Number of encoder and decoder layers. Defaults to 6.
        heads (int, optional): Number of attention heads. Defaults to 8.
        hidden_size_ff (int, optional): Hidden size of the feed forward network. Defaults to 2048.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self, 
        source_vocab_size: int,
        target_vocab_size: int,
        source_max_len: int,
        target_max_len: int,
        d_model: int = 512,
        num_layers: int = 6,
        heads:int = 8,
        hidden_size_ff: int = 2048,
        dropout: float = 0.1
        ) -> None:
        super().__init__()

        self.source_embed = Embedding(d_model, source_vocab_size)
        self.target_embed = Embedding(d_model, target_vocab_size)

        self.source_pos_embed = PosEmbedding(d_model, source_max_len, dropout)
        self.target_pos_embed = PosEmbedding(d_model, target_max_len, dropout)

        self.encoder = Encoder(num_layers, d_model, heads, hidden_size_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, heads, hidden_size_ff, dropout)

        self.projection = ProjectionLayer(d_model, target_vocab_size)

        self.param_init()

    def param_init(self):
        """
        Initialize model parameters.
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def encode(self, source, source_mask):
        """
        Encode the source sequence.

        Args:
            source (torch.Tensor): Source input tensor.
            source_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Encoded source tensor.
        """

        source = self.source_embed(source)
        source = self.source_pos_embed(source)
        return self.encoder(source, source_mask)        

    def decode(self, bottleneck, source_mask, target, target_mask):
        """
        Decode the target sequence.

        Args:
            bottleneck (torch.Tensor): Bottleneck tensor from encoder.
            source_mask (torch.Tensor): Source mask tensor.
            target (torch.Tensor): Target input tensor.
            target_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Decoded target tensor.
        """

        target = self.target_embed(target)
        target = self.target_pos_embed(target)
        return self.decoder(target, bottleneck, source_mask, target_mask)

    def project(self, x):
        """
        Project the tensor to target vocabulary space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected tensor.
        """

        return self.projection(x)