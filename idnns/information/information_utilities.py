import numpy


def calc_information(probTgivenXs, PYgivenTs, PXs, PYs, PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    Ht = numpy.nansum(-numpy.dot(PTs, numpy.log2(PTs + numpy.spacing(1))))
    Htx = - numpy.nansum((numpy.dot(numpy.multiply(probTgivenXs, numpy.log2(probTgivenXs)), PXs)))
    Hyt = - numpy.nansum(numpy.dot(PYgivenTs * numpy.log2(PYgivenTs + numpy.spacing(1)), PTs))
    Hy = numpy.nansum(-PYs * numpy.log2(PYs+numpy.spacing(1)))

    IYT = Hy - Hyt
    ITX = Ht - Htx

    return ITX, IYT
