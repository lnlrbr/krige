#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#       ordinarykriging.py
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

import numpy as np
import scipy.spatial as spatial
import models
from base import *


class Grid(BasicGrid):
    def __init__(self, *args):
        """ Constructs a grid from coordinates and attributes.

        Parameters
        ----------
        Can be :
        1/- x, y, v, e : sequences with same lengths
                    x, y : coordinates
                    v    : values
                    e    : standards errors. Can be omitted and will be filled
                           with 0.
            the first point is represented by x[0], y[0], v[0], e[0]
        2/- u : recarray, list of lists, ...
            the first point is represented by u[0] = [x[0], y[0], v[0], e[0]]

        """
        super(Grid, self).__init__(*args)
        self.infos = self.compInfos(self.grid)

    def __invGammatrix__(self, model, coords):
        """Computes the inverse gamma matrix of the grid.

        Parameters
        ----------
        model  : Model
                 the used model
        coords : ndarray
                 list of points coordinates [[x0,y0],[x1,y1],...]

        Returns
        -------
        ndarray
        inverse gamma matrix (like)

        """
        # model.func is vectorized...
        matrix = model.func(spatial.distance_matrix(coords, coords))
        #Unbiasedness constraints
        m = np.ones( (matrix.shape[0]+1, matrix.shape[1]+1) )
        m[:-1,:-1] = matrix
        m[-1:-1] = 0.
        res = np.matrix(m).I
        return res

    def predictedPoint(self, x, y, model, coords, values, invg):
        """Prediction of the Big Kriging for a point \o/

        Parameters
        ----------
        x, y : floats
               coordinates of the desired predicted point
        model : Model
                what model to use (and not your favorite color!)
        coords : ndarray
                 original grid coordinates
        values : ndarray
                 original grid values, ordered like coords
        invg : the resulting inverse gamma matrix based on model and coords

        Returns
        ----------
        array(x,y,v,e)
            x, y : coordinates of the desired predicted point
            v    : the predicted value
            e    : the standard error

        """
        dist = spatial.distance_matrix(coords, [[x, y],])
        gg = np.matrix( np.vstack([model.func(dist), [1,]]) )
        weights = invg*gg
        v = np.sum( values[:, np.newaxis]*np.asarray(weights[:-1]) )
        e = np.sqrt( abs(np.sum(gg.A1*weights.A1)) )
        return np.asarray([x, y, v, e])

    def predictedGrid(self, x, y, model, full=False):
        """Prediction for an entire grid of points

        Parameters
        ----------
        x, y : 1D array
               coordinates of the points on the predicted grid
        model : Model
                model to use
        full : boolean
               If False, the original grid will be downsampled for performances
               issue. Use True for a prediction based on the entire original
               grid (maybe very slow, like a european swallow)

        Returns
        ----------
        BasicGrid containing coordinates, predicted values and standards errors
        for each point.

        """
        if full:
            grid = self.grid
        else:
            print("Downsampling...")
            if self.grid.shape[0]>30:
                grid = randin(self.grid, 30)
            else:
                print("Not enough datas for downsampling. Continue with all datas...")
                grid = self.grid
        coords = np.asarray([grid.x, grid.y]).T
        invg = self.__invGammatrix__(model, coords)
        pg = [self.predictedPoint(x[i], y[i], model, coords, grid.v, invg) for i in range(len(x))]
        pg = BasicGrid(pg)
        return pg


def main():
    return 0

if __name__ == '__main__': main()
