#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#       models.py
#
#       Copyright 2009 Lionel Roubeyrie <lroubeyrie@limair.asso.fr>
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
from matplotlib import pyplot as plt


class Model(object):
    def __init__(self, type=None, sill=None, range=None, nugget=None):
        self.type = type
        self.sill = float(sill)
        self.range = float(range)
        self.nugget = float(nugget)
        if self.nugget < 0:
            self.nugget = 0
        self.func = np.vectorize(self.f)
        self.variance = 0

    def f(self, h):
        return 0

    def __cmp__(self, other):
        return cmp(self.variance, other.variance)

    def residual(self, params, dist, svar):
        self.dist = dist
        self.svar = svar
        self.sill, self.range, self.nugget = params
        if self.nugget < 0:
            self.nugget = 0
        if self.nugget > self.sill:
            self.nugget = self.sill
        err = self.svar - self.func(self.dist) #Reals variances less computed variances
        return err

    def plot(self):
        h = np.arange(0, np.max(self.dist), 0.1)
        v = self.func(h)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h, v, 'r-')
        ax.plot(self.dist, self.svar, 'bo')
        plt.legend(['Fit', 'True'])
        plt.xlabel("distance")
        plt.ylabel("semivariance")
        plt.title("SemiVariogram\n %.1f*%s(%.1f)+%.1f"%(self.sill, self.type,
        self.range, self.nugget))

    def getCorrectedSill(self):
        return self.nugget + self.sill

    corrected_sill = property(getCorrectedSill)


class Spherical(Model):
    __type__ = "Spherical"
    def __init__(self, sill=None, range=None, nugget=None):
        super(Spherical, self).__init__("Spherical", sill, range, nugget)

    def f(self, h):
        if self.range <= h:
            return self.nugget + self.sill
        else:
            range = float(h)/self.range
            return (self.nugget + self.sill*( (1.5*range) - (0.5*(range**3))))


class Exponential(Model):
    __type__ = "Exponential"
    def __init__(self, sill=None, range=None, nugget=None):
        super(Exponential, self).__init__("Exponential", sill, range, nugget)

    def f(self, h):
        range = -(3*h)/self.range
        try:
            return self.nugget + self.sill*(1 - np.exp(range))
        except OverflowError:
            return -np.infty


class Gaussian(Model):
    __type__ = "Gaussian"
    def __init__(self, sill=None, range=None, nugget=None):
        super(Gaussian, self).__init__("Gaussian", sill, range, nugget)

    def f(self, h):
        range = -3*((h/self.range)**2)
        return self.nugget + self.sill*(1 - np.exp(range) )


class Pentaspherical(Model):
    __type__ = "Pentaspherical"
    def __init__(self, sill=None, range=None, nugget=None):
        super(Pentaspherical, self).__init__("Pentaspherical", sill, range, nugget)

    def f(self, h):
        if self.range <= h:
            return self.nugget + self.sill
        else:
            range = float(h)/self.range
            return self.nugget + self.sill*( ((15.0/8.0)*range) - ((5.0/4.0)*range**3) + ((3.0/8.0)*range**5) )


class Nugget(Model):
    __type__ = "Nugget"
    def __init__(self, sill=0, range=None, nugget=None):
        super(Nugget, self).__init__("Nugget", sill, range, nugget)

    def f(self, h):
        return self.nugget


def getModels(sill=1.0, nugget=0.0, range=100.0):

    models = {
    "Spherical" : Spherical(sill, range, nugget),
    "Exponential" : Exponential(sill, range, nugget),
    "Gaussian" : Gaussian(sill, range, nugget),
    "Pentaspherical" : Pentaspherical(sill, range, nugget),
    "Nugget" : Nugget(sill, range, nugget)
    }
    return models

