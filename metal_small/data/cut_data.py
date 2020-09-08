from numpy.random import default_rng

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

    rng = default_rng(1234)
    indxs = rng.integers(len(fulldata), size=nsamp)

    fulldata[indxs].write(filename.replace(".fits", "_samp.fits"), overwrite=True)


if __name__ == "__main__":
    sample_obsdata("14675_LMC-13361nw-11112.gst.fits")
