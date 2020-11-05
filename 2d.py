# Boilerplate imports
from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos, arcsin
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
# Linear algebra functions
from numpy.linalg import inv, pinv, det, eig, eigh, eigvals
from matplotlib.patches import Ellipse # for plotting ellipse

# program to generate a covariance matrix where the variance values are fixed at [2,2].
def generate_cov(off_diag):
    cov = ary([[2.0, off_diag], [off_diag, 2.0]])
    return cov

PLOT_CIRCLE = True
# if PLOT_CIRCLE: plot the error ellipse using 11 differernt covariance values;
# else: plot plot the variation of the error ellipse area wrt. the covariance value.
if not PLOT_CIRCLE:
    determinant_cov, determinant_inv_cov, size = [], [], []


# Setup:
# 
# - We have a point cloud with a center at (0,0).
# - 1 sigma of the points (68% of them) lies within the error ellipse. (We won't be plotting the remaining 32%)
# - The top left element of the covariance matrix describes the width of this ellipse, and the bottom right describes the height of the ellipse.
# - Therefore, varying the covariance value (i.e. the symmetric off-diagonal terms) should only make the ellipse into a thinner ellipse that is leaning left/right.

# In[ ]:


if __name__=='__main__':
    if PLOT_CIRCLE:
        i_range = np.linspace(-1.98, 1.98, 11) # plotting the actual covariance circle
    else:
        i_range = np.linspace(-0, 1.98, 30) # plotting area variation as covariance varies across(0, 1)
    for i in i_range:
    # create a scatter of dots for plotting
        left, right, resolution = -10, 10, 300
        pt_list = np.linspace(left,right, resolution)
        unit_square_size = ((right-left)/resolution)**2
        points = ary(np.meshgrid(pt_list, pt_list)).T.flatten().reshape([-1,2])

        # generate the covariance matrix, and evaluate the shape of the error ellipse
        cov = generate_cov(i)
        print(cov)
        (minor, major), eig_mat = eig(inv(cov) * det(cov))
        print(major, minor)
        mat = inv(eig_mat)
        # ignore the arccos becuase it will always return a non-negative value.
        orientation = np.mean([arcsin(mat[0,1]), -arcsin(mat[1,0])])*np.sign(mat[0,0])

        chi2_level = ((points @ inv(cov)) * points).sum(axis=1)
        mask = chi2_level<=1

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
            
        else: # plotting the variation of the area
            determinant_cov.append(det(cov))
            determinant_inv_cov.append(det(inv(cov)))
            size.append(sum(mask)*unit_square_size)
            
    if not PLOT_CIRCLE:
        plt.plot(i_range, sqrt(determinant_cov)*pi, label='cov determinant')
        plt.xlabel('covariance (off-diagonal elements) value')
        plt.ylabel('area/area prediction/other quantities')
        plt.plot(i_range, determinant_inv_cov, label='inv_cov determinant')
        plt.plot(i_range, size, label='ellipse size')
        plt.legend()
        plt.show()


# Using ```PLOT_CIRCLE = True``` we can verify that my code has correctly plotted the error ellipse correctly: All points with $chi^2 \le 1$ are plotted within the error ellipse.
# 
# Meanhwile, using ```PLOT_CIRCLE = False``` we can show that the covariance ellipse area is well modelled by the equation det(cov), even when the covariance ellipse becomes very thin. The area can be empirically calculated by counting the number of points that the ellipse covers when spread over an evenly spaced grid.
