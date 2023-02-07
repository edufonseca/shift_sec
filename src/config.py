# Basic configurations for the models (VGG41 and VGG42) and pooling mechanisms (TLPF, APS, BlurPool).
# See paper for more details https://arxiv.org/pdf/2107.00623.pdf

# ===CAVEATS===
# 1) There might be some additional hparams called within the model architecture that need to be defined externally,
# but they should be intuitive by taking a look at the paper.
#
# 2) Some hparams are hardcoded within the model architecture, with multiple options to select.
# Typically the currently selected one is the best one in our experiments. And the others, commented, were also explored.
# Often these multiple-choice hparams are accompanied by a TODO.

width_factors = [1]     # VGG41
# width_factors = [2]     # VGG42

global_poolings = ['maxmeanfreq']     # best case for global pooling
bb_pools_1 = ['mp_invar']             # Intra-block pooling (IBP): partial translation invariance w/o dim reduction.
trans_invs = [3]

# pooling mechanisms
# bb_pools_2 = ['mp_size_same_stride']  # baseline is standard max pooling
# bb_pools_2 = ['blur_pool_2D']         # BlurPool
bb_pools_2 = ['blur_pool_2D_learn']     # TLPF
# bb_pools_2 = ['ApsPool']              # APS
# bb_pools_2 = ['ApsPool_learn']        # APS & TLPF (best case)
