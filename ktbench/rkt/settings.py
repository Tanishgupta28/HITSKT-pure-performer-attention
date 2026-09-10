"""Approved RKT training constants shared by smoke and production runs."""

from __future__ import annotations


RKT_EPOCH_CEILING = 300
RKT_BATCH_SIZE = 128
RKT_LEARNING_RATE = 0.001
RKT_WEIGHT_DECAY = 0.00001

# The paper does not state a clipping threshold. This is the authors'
# released trainer default, retained with the user's explicit approval.
RKT_GRADIENT_CLIP = 10.0
