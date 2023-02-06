# These are the basic configurations for the model and pooling architectures.
# There might be some additional hparams called within the model architecture that need to be defined externally,
# but they should be intuitive.
# Also, there are some hparams that are hardcoded within the model (with several options commented out);
# they are usually accompanied by a TODO.

width_factors = [1]     # VGG41
# width_factors = [2]     # VGG42

global_poolings = ['maxmeanfreq']     # best case for global pooling
bb_pools_1 = ['mp_invar']             # best case for bb1
trans_invs = [3]

# pooling mechanisms
# bb_pools_2 = ['mp_size_same_stride']  # baseline is standard max pooling
# bb_pools_2 = ['blur_pool_2D']         # BlurPool
bb_pools_2 = ['blur_pool_2D_learn']   # TLPF
# bb_pools_2 = ['ApsPool']              # APS
# bb_pools_2 = ['ApsPool_learn']        # APS & TLPF (best case)




