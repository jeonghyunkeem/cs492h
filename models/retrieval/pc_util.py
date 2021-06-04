import numpy as np

# ----------------------------------------------------------------------------------------------  Point Cloud Sampling /
def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is (N x C), output is (num_sample x C)
    """
    if replace is None:
        replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]