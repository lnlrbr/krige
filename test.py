import ordinarykriging as ok
import matplotlib.mlab as mpl
import pylab as plb
import time

def plotit(x, y, z, title):
    plb.figure()
    try:
        X = list()
        Y = list()
        [X.append(i) for i in x if i not in X]
        [Y.append(i) for i in y if i not in Y]
        Z = plb.reshape(z, (len(Y), len(X)))
        plb.contourf(X, Y, Z, 10)
    except:
        plb.scatter(x, y, c=z)
    plb.title(title)
    plb.colorbar()
    plb.show()

def run():
    t1=time.time()
    a=mpl.csv2rec('datas.csv')
    g=ok.Grid(a.x, a.y, a.v)
    ##plotit(g.grid.x, g.grid.y, g.grid.v, "Initial grid")
    model=g.fitSermivariogramModel('Exponential', nlag=20)
    ##model.plot()
    x,y=g.regularBasicGrid(nx=40, ny=40)
    pg=g.predictedGrid(x, y, model)
    ##plotit(pg.grid.x, pg.grid.y, pg.grid.v, "Predicted grid")
    ##plotit(pg.grid.x, pg.grid.y, pg.grid.e, "Predicted Error grid")
    t2=time.time()
    print("Operation performed in %.2f seconds"%(t2-t1))

if __name__ == '__main__':
    run()
