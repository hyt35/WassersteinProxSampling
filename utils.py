import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

class Plotter1D():
    def __init__(self, main_path, fun, plotgrid):
        self.main_path = main_path
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        self.iter = 0
        self.fun = fun
        self.plotgrid = plotgrid # eg linspace
        self.fun_plotgrid = fun(plotgrid)

    # Todo

    def do_plotting(self, X_list, name, title = None):
        fig, ax = plt.subplots()
        # fig.clf()
        # ax.set_xlim(self.plotgrid[0], self.plotgrid[-1])
        # plot the function
        if title is not None:
            fig.suptitle(title)
        ax.plot(self.plotgrid, self.fun_plotgrid)
        X_list = X_list.ravel() # should be a bunch of points
        fun_X = self.fun(X_list)
        # now plot
        ax.scatter(X_list, fun_X, color='b', alpha=0.1)
        fig.savefig(os.path.join(self.main_path, name))

        ax.set_xlim(self.plotgrid[0], self.plotgrid[-1])
        fig.savefig(os.path.join(self.main_path, name+"_zoom"))
        plt.close()

class Plotter2D():
    def __init__(self, main_path, fun, grid_x, grid_y):
        self.main_path = main_path
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        self.iter = 0
        self.fun = fun
        if self.fun is not None:
            meshgrid_X, meshgrid_Y = np.meshgrid(grid_x, grid_y)
            meshgrid = np.stack([meshgrid_X.ravel(), meshgrid_Y.ravel()], axis=1)
            
            self.fun_plotgrid = fun(meshgrid).reshape(len(grid_x), len(grid_y))
        self.grids = (grid_x, grid_y)
    # Todo

    def do_plotting(self, X_list, name, title = None):
        fig, ax = plt.subplots()
        # fig.clf()
        if title is not None:
            fig.suptitle(title)
        grid_x, grid_y = self.grids
        # ax.set_xlim(grid_x[0], grid_x[-1])
        # ax.set_ylim(grid_y[0], grid_y[-1])
        # plot the function
        # ax.plot(self.plotgrid, self.fun_plotgrid)
        if self.fun is not None:
            cf = ax.contour(self.grids[0], self.grids[1], self.fun_plotgrid)

        # X_list = X_list.ravel() # should be a bunch of points
        # fun_X = self.fun(X_list)
        # # now plot
        ax.scatter(X_list[:,0], X_list[:,1], color='b', alpha=0.1)
        if self.fun is not None:
            fig.colorbar(cf)
        fig.savefig(os.path.join(self.main_path, name))

        ax.set_xlim(grid_x[0], grid_x[-1])
        ax.set_ylim(grid_y[0], grid_y[-1])
        fig.savefig(os.path.join(self.main_path, name+"_zoom"))
        plt.close()

def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--beta', type=float, default=1.)
        parser.add_argument('--T', type=float, default=1.)
        parser.add_argument('--stepsize', type=float, default=0.1)
        parser.add_argument('--num_inits', type=int, default=100)
        # parser.add_argument('--test_fn', type=str, default="wavy")
        parser.add_argument('--n_iters', type=int, default=100)
        parser.add_argument('--sample_iters', type=int, default=50)


        return parser