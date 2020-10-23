from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos, arcsin
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
from numpy.linalg import inv, pinv, det, eig, eigh, eigvals
from matplotlib.patches import Ellipse

def generate_cov(off_diag):
    cov = ary([[1.0,off_diag],[off_diag,1.0]])
    return cov*2

PLOT_CIRCLE = True
if not PLOT_CIRCLE:
    determinant_cov, determinant_inv_cov, size = [], [], []

if __name__=='__main__':
    if PLOT_CIRCLE:
        i_range = np.linspace(-.98, .98, 11)
    else:
        i_range = np.linspace(-0, .98, 30)
    for i in i_range:
    # create a scatter of dots for plotting
        left, right, resolution = -10, 10, 300
        pt_list = np.linspace(left,right, resolution)
        unit_square_size = ((right-(left))/resolution)**2
        points = ary(np.meshgrid(pt_list, pt_list)).T.flatten().reshape([-1,2])

        # generate the covarince matrix, and evaluate the sizes of it needed
        cov = generate_cov(i)
        print(cov)
        (major, minor), eig_mat = eig(inv(cov)*det(cov))
        # (major, minor), eig_mat = eig(inv(cov))
        # print(major, minor)
        # print(eigvals(cov))
        mat = eig_mat.T
        # ignore the arccos becuase it will always return a non-negative value.
        orientation = np.mean([-arcsin(mat[0,1]), arcsin(mat[1,0])])

        chi2_level = ((points @ inv(cov)) * points).sum(axis=1)
        mask = chi2_level<=1

        if not PLOT_CIRCLE:
            determinant_cov.append(det(cov))
            determinant_inv_cov.append(det(inv(cov)))
            size.append(sum(mask)*unit_square_size)

        # plotting
        if PLOT_CIRCLE:
            fig, ax = plt.subplots()
            ellipse = Ellipse([0,0], # centered at the origin
                        2*sqrt(major),
                        2*sqrt(minor),
                        np.rad2deg(orientation)
                        )
            # DUUUUDE I got the width=2*sqrt(major)/sqrt(det(inv(cov))) equation by trial and error LMAO
            ax.add_patch(ellipse)
            ax.scatter(*points[mask].T, marker='+', alpha=0.4, color='C1', zorder=10) # scatter plot approach
            plt.show()
    if not PLOT_CIRCLE:
        plt.plot(sqrt(determinant_cov)*pi, label='cov determinant')
        plt.plot(determinant_inv_cov, label='inv_cov determinant')
        plt.plot(size, label='ellipse size')
        plt.legend()
        plt.show()
    """
    1. groud truth: we have this cloud of points of finite size.
    2. 
    """