#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#       base.py
#
#       Copyright 2010 Lionel Roubeyrie <lroubeyrie@limair.asso.fr>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

__author__ = "Lionel Roubeyrie"
__contact__ = "lionel.roubeyrie at gmail dot com"

import models
import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize


def randin(grid, pcent):
    """Returns pcent values of a grid where indices are randomly
    selected only once.
    """
    n = grid.shape[0]-1
    if n == 0:
        return None
        #FIXME : make an exception
    ind = list()
    m = int(n*pcent/100)
    while len(ind) != m:
        a = np.random.random_integers(0, n, 1)
        if a not in ind:
            ind.append(a)
    res = grid[ind].ravel()
    return res

def pprint(text):
    print("--------------------------")
    for l, t in text:
        print(" %s : %s"%(l, t))
    print("--------------------------\n")


class BasicGrid(object):
    def __init__(self, *args):
        if len(args) == 1: # U
            self.data = np.asarray(args[0])
        elif len(args) == 3: # X, Y, V
            self.data = np.array([args[0], args[1], args[2], np.zeros_like(args[0])]).T
        elif len(args)==4: # X, Y, V, E
            self.data = np.array(*args).T
        else:
            raise ValueError, "Bad input datas formats"
        names = ("x", "y", "v", "e")
        self.grid = np.rec.fromrecords(self.data, names=names)
        pprint([
        ["Number of point", self.grid.x.shape[0]],
        ["Standard deviation of the grid", np.std(self.grid.v)]
        ])

    def distance(self, a, b):
        """ Returns the distance between points

        """
        return np.sqrt( np.square(a.x - b.x) + np.square(a.y - b.y) )

    def semiVariance(self, a, b):
        """ Returns the semivariance between points

        """
        return 0.5*np.square(a.v - b.v)

    def angle(self, a, b):
        """ Returns the angle (in degrees) between points

        """
        return np.degrees( np.arctan((b.y - a.y) / (b.x - a.x)) )

    def compInfos(self, grid):
        """ Retrieves informations from a grid

        Parameters
        ----------
        grid : BasicGrid
               a complete grid with coordinates and values

        Returns
        ----------
        recarray
        id_i : integer
               origin point index in the grid
        id_j : integer
               target point index in the grid
        dist : float
               distance between the two points
        svar : float
               the empirical semivariance between the two points
        ang  : float
               the angle in radian between the two points

        """
        print("Analysing grid. Please wait...")
        n = grid.shape[0]
        id_i = np.array([], dtype=np.int)
        id_j = np.array([], dtype=np.int)
        dist = np.array([], dtype=np.float)
        svar = np.array([], dtype=np.float)
        ang = np.array([], dtype=np.float)
        for k in range(n-1):
            j = range(k+1, n)
            i = np.repeat(k, len(j))
            pi = grid[i]
            pj = grid[j]
            id_i = np.append(id_i, i)
            id_j = np.append(id_j, j)
            dist = np.append(dist, self.distance(pi, pj))
            svar = np.append(svar, self.semiVariance(pi, pj))
            ang = np.append(ang, self.angle(pi, pj))
        p = np.vstack((id_i, id_j, dist, svar, ang))
        names = ("id_i", "id_j", "dist", "svar", "ang")
        p = np.rec.fromrecords(p.T, names=names)
        return p

    def fitSermivariogramModel(self, modelname, nlag=15, tsill=None,
                               trange=None, tnugget=0.0):
        """Fit a semivariogram model to the datas
        Returns a model with the best parameters (sill, range, nugget) to
        fit the datas (semivariance against distance)

        Parameters
        ----------
        modelname : String
                    The desired model name to use. One of:
                    Spherical, Exponential, Gaussian, Pentaspherical, Nugget
        nlag :  Integer
                If not None, the semivariogram will be fitted using nlags
                bins (empirical semivariogram), between distance[0, max], else
                all the datas will be used (global semivariogram).
        tsill : Float
                Temporary initial sill, used by the fitted algorithm. If
                None it will be estimated from the datas.
        trange : Float
                 Temporary initial range, used by the fitted algorithm.
                 If None it will be estimated from the datas.
        tnugget : Float
                  Temporary initial nugget, used by the fitted algorithm.

        Returns
        ----------
        a model instance with sill, range and nugget computed to best fit
        the datas. Highly depending from the initial inputs, due to the used
        algorithm (scipy.optimize.leastsqr)

        """
        # the input datas
        dist = self.infos.dist
        svar = self.infos.svar

        # lag?
        if nlag is None:
            nlag = len(dist)
        if nlag < 1:
            raise ValueError, "nlag must be >=1"
        # Initials parameters estimations
        # TODO : find best solutions
        if tsill is None:
            tsill = 9/10*np.max(svar)
        if trange is None:
            trange = 0.5*np.max(dist)
        # Sort by distance
        sortind = np.argsort(dist)
        sortdist = dist[sortind]
        sortsvar = svar[sortind]
        # Select by bins
        index = sortdist.searchsorted(np.linspace(0, dist.max(), nlag+1))
        dist = [sortdist[index[i-1]:index[i]].mean() for i in range(1, len(index))]
        svar = [sortsvar[index[i-1]:index[i]].mean() for i in range(1, len(index))]

        # Retrieve the model class from his name
        model = models.getModels(sill=tsill, range=trange, nugget=tnugget)[modelname]

        # these are our inital guesses for the sill and range
        params = (model.sill, model.range, model.nugget)

        # perform the least squares
        lstsqResult = optimize.leastsq(model.residual, params, args=(dist, svar), full_output=0)

        if(lstsqResult != 1):
            p = lstsqResult[0]
            model.sill, model.range, model.nugget = p
            if model.range > 0:
                # work out the square deviation too
                squareDeviates = model.residual(p, dist, svar)**2
                # we divide by number of points minus the degrees of freedom
                denom = 1 / float(len(svar)-len(p))
                model.variance = denom*sum(squareDeviates)
                pprint([["Model Type", model.type], ["Sill", model.sill],
                ["Range", model.range], ["Nugget", model.nugget]])
                return model
            else:
                pprint([["Error", "Computed range <=0 :("],])
                return None
        else:
            pprint([["Error", "Bad fitting computation :("],])
            return None

    def tofile(self, fname):
        """Save data to a CSV file

        Parameters
        ----------
        fname : String
                CSV file name

        """
        f = open(fname, "wb")
        f.write("X,Y,V, E\n")
        for p in self.grid:
            f.write("%s,%s,%s,%s\n"%(p.x, p.y, p.v, p.e))
        f.close()

    def regularBasicGrid(self, xmin=None, ymin=None, xmax=None, ymax=None,
                         nx=30, ny=30):
        """
        Creates a new regular grid with controled parameters. The new grid will
        not be created like a source krigging grid (no semivariance computed)

        Parameters
        ----------
        xmin, ymin, xmax, ymax : Floats
            lower left and upper right coordinates of the new grid. If None, the
            parameters will be set from the original grid
        nx, ny : integers
            numbers of points in each axis on the new regular grid

        Returns
        ----------
        A new regular point grid

        """
        if xmin is None:
            xmin = self.grid.x.min()
        if ymin is None:
            ymin = self.grid.y.min()
        if xmax is None:
            xmax = self.grid.x.max()
        if ymax is None:
            ymax = self.grid.y.max()
        X = np.linspace(xmin, xmax, nx)
        Y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        return X, Y
