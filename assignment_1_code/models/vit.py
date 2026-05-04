


## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code


"""
Vision Transformer for CIFAR-10 classification.

================================================================================
Source attribution
================================================================================
The transformer machinery used below comes from the lucidrains/vit-pytorch
package, specifically the ``SimpleViT`` variant.

    Repository: https://github.com/lucidrains/vit-pytorch
    File:       vit_pytorch/simple_vit.py
    Paper:      Beyer, L., Zhai, X., & Kolesnikov, A. (2022).
                "Better plain ViT baselines for ImageNet-1k."
                https://arxiv.org/abs/2205.01580
    License:    MIT

Compared to the canonical ViT (Dosovitskiy et al. 2021), ``SimpleViT`` makes
three small changes that empirically improve from-scratch training on small
datasets -- exactly our setup (no pretraining, ~30 epochs, CIFAR-10):

    1. Fixed 2-D sin-cos positional embedding (instead of a learnable one)
    2. Mean pooling over patch tokens (instead of a [CLS] token)
    3. No dropout

================================================================================
Where the key ViT concepts live (for the report)
================================================================================
The four defining concepts of a Vision Transformer are all visible in
``vit_pytorch/simple_vit.py`` (file you can install via ``pip install
vit-pytorch`` and inspect locally):

    (a) PATCH EMBEDDING
        The input image is split into non-overlapping patches and each patch
        is linearly projected to a token embedding.
        See ``SimpleViT.__init__``: the ``self.to_patch_embedding`` block
        with ``Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)")`` followed
        by ``LayerNorm + Linear(patch_dim, dim) + LayerNorm``.

    (b) POSITIONAL ENCODING
        Patches have no inherent order, so we add a position embedding to
        every token. SimpleViT uses fixed 2-D sin-cos embeddings.
        See ``posemb_sincos_2d`` in simple_vit.py and the call site in
        ``SimpleViT.forward`` where it is added to ``x``.

    (c) MULTI-HEAD SELF-ATTENTION
        The core of the transformer encoder: each token attends to every
        other token via scaled dot-product attention, run in parallel across
        multiple heads.
        See the ``Attention`` class (``q, k, v = ...``;
        ``attn = (q @ k.transpose(-1, -2)) * scale``;
        ``attn.softmax(dim=-1)``; ``out = attn @ v``) and the ``Transformer``
        class which stacks ``Attention + FeedForward`` blocks.

    (d) CLASSIFICATION HEAD
        After the encoder, tokens are mean-pooled into a single vector and
        passed through a linear classifier.
        See the last lines of ``SimpleViT.forward``:
        ``x = x.mean(dim=1); x = self.to_latent(x); return self.linear_head(x)``.

================================================================================
Required dependency
================================================================================
    pip install vit-pytorch
"""

import torch
import torch.nn as nn

from vit_pytorch import SimpleViT


class YourViT(nn.Module):
    """
    Thin wrapper around lucidrains' ``SimpleViT``, configured for CIFAR-10.

    Configuration rationale (32x32 inputs, 10 classes, train from scratch):

        image_size = 32     CIFAR-10 native resolution.
        patch_size = 4      4x4 patches -> 8x8 = 64 patch tokens. Patch sizes
                            of 8 or 16 leave too few tokens to learn from on
                            32x32 images.
        dim        = 256    Token embedding dimension. Try 192 for a smaller
                            and faster model, or 384 for a bigger/slower one.
        depth      = 6      Number of stacked transformer encoder layers.
        heads      = 8      Multi-head attention heads (head dim = 32).
        mlp_dim    = 512    Hidden dim of the per-block MLP. The original
                            ViT uses 4*dim; on CIFAR-10 from scratch 2*dim
                            is sufficient and trains faster.

    Total parameter count: roughly 3.7 M -- comparable to a small ResNet,
    about 25x smaller than ViT-Base (86 M).
    """

    def __init__(
            self,
            num_classes: int = 10,
            image_size: int = 32,
            patch_size: int = 4,
            dim: int = 256,
            depth: int = 6,
            heads: int = 8,
            mlp_dim: int = 512,
            channels: int = 3,
    ) -> None:
        super().__init__()
        self.net = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input batch of shape (B, 3, 32, 32), already normalized.
        Returns:
            Logits of shape (B, num_classes). CrossEntropyLoss applies the
            log-softmax internally, so we do NOT softmax here.
        """
        return self.net(x)
