global_poolings = ['maxmeanfreq']     # best case for global pooling
bb_pools_1 = ['mp_invar']             # best case for bb1
trans_invs = [3]


# pooling mechanisms
bb_pools_2 = ['mp_size_same_stride']  # baseline is standard max pooling
bb_pools_2 = ['blur_pool_2D']         # BlurPool
bb_pools_2 = ['blur_pool_2D_learn']   # TLPF
bb_pools_2 = ['ApsPool']              # APS
bb_pools_2 = ['ApsPool_learn']        # APS & TLPF (best case)




