_base_ = '../default.py'

expname = 'exp0'
basedir = ''

data = dict(
    datadir='',
    dataset_type='tankstemple',
    inverse_y=True,
    # load2gpu_on_the_fly=True,
    white_bkgd=False,
)

coarse_train = dict(
    N_iters=10000 // 4 * 3,
)

fine_train = dict(
    N_iters=20000 // 4 * 3,
    pg_scale=[1000, 2000, 3000, 5000, 7000, 9000],
)

coarse_model_and_render = dict(
    num_voxels=1024000 // 2,
    num_voxels_base=1024000 // 2,
)

fine_model_and_render = dict(
    num_voxels=160**3 // 2,
    num_voxels_base=160**3 // 2,
)
