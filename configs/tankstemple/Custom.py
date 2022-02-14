_base_ = '../default.py'

expname = ''
basedir = ''

data = dict(
    datadir='',
    dataset_type='tankstemple',
    inverse_y=True,
    # load2gpu_on_the_fly=True,
    white_bkgd=False,
)

coarse_train = dict(
    N_iters=5000 // 4 * 3,
    pervoxel_lr_downrate=2,
)

fine_train = dict(
    N_iters=20000 // 4 * 3,
    pg_scale=[1000, 2000, 3000, 4000, 6000, 8000, 10000],
)

coarse_model_and_render = dict(
    num_voxels=1024000 // 2,
    num_voxels_base=1024000 // 2,
)

fine_model_and_render = dict(
    num_voxels=160**3 // 2,
    num_voxels_base=160**3 // 2,
)
