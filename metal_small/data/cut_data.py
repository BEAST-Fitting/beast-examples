from numpy.random import default_rng
import numpy as np

from astropy.table import Table


def sample_obsdata(filename, nsamp=500):
    """
    Sample an observed dataset to reduce the number of data points.
    Useful for generating example data files that run quickly.
    E.g., for a BEAST example.

    Parameters
    ----------
    filename : str
        name of the file with the file dataset

    nsamp : int
        number of samples to keep in new dataset
    """
    fulldata = Table.read(filename)

    # remove all sources that do not have measurements in all bands
    basefilters = ["F225W", "F275W", "F336W", "F475W", "F814W", "F110W", "F160W"]
    gvals = [True] * len(fulldata)
    for cfilt in basefilters:
        gvals = gvals & (fulldata[cfilt + "_RATE"] != 0.0)
    fulldata = fulldata[gvals]

    # sort by the F475W flux to ensure the output is ordered by brightness
    sindxs = np.argsort(fulldata["F475W_RATE"])

    rng = default_rng(1234)
    indxs = rng.integers(len(fulldata), size=nsamp)

    fulldata[sindxs[indxs]].write(filename.replace(".fits", "_samp.fits"), overwrite=True)


if __name__ == "__main__":
    sample_obsdata("14675_LMC-13361nw-11112.gst.fits")
