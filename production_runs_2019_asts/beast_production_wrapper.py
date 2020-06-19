import numpy as np
import os
import types

from beast.tools.run import create_physicsmodel, make_ast_inputs

from beast.tools import create_background_density_map
from beast.plotting import plot_mag_hist, make_ds9_region_file, plot_ast_histogram

from shapely import geometry
from astropy.table import Table
from astropy.io import fits


def beast_production_wrapper():
    """
    This does all of the steps for a full production run, and can be used as
    a wrapper to automatically do most steps for multiple fields.
    * make beast settings file
    * make source density map
    * make background density map
    * make physics model (SED grid)
    * make input list for ASTs

    Places for user to manually do things:
    * editing code before use
        - beast_settings_template.py: setting up the file with desired parameters
        - here: list the catalog filter names with the corresponding BEAST names
        - here: choose settings (pixel size, filter, mag range) for the source density map
        - here: choose settings (pixel size, reference image) for the background map

    """

    # get the list of fields
    field_names = ["M31-B17-WEST"]

    # reference image headers
    ast_ref_im = ["./data/M31-B17-WEST_F475W_drz_head.fits"]

    # filter for sorting
    ref_filter = ["F475W", "F475W"]

    # filter for checking flags
    flag_filter = ["F475W", "F475W"]

    # coordinates for the boundaries of the field
    boundary_coord = [
        [
            [11.3569155882981, 11.3332629372269, 11.1996295883163, 11.2230444144334],
            [42.0402119924196, 41.9280756279533, 41.9439995726193, 42.0552016945299],
        ]
    ]

    # number of fields
    # n_field = len(field_names)

    # Need to know what the correspondence is between filter names in the
    # catalog and the BEAST filter names.
    #
    # These will be used to automatically determine the filters present in
    # each GST file and fill in the beast settings file.  The order doesn't
    # matter, as long as the order in one list matches the order in the other
    # list.
    #
    gst_filter_names = ["F275W", "F336W", "F475W", "F814W", "F110W", "F160W"]
    beast_filter_names = [
        "HST_WFC3_F275W",
        "HST_WFC3_F336W",
        "HST_ACS_WFC_F475W",
        "HST_ACS_WFC_F814W",
        "HST_WFC3_F110W",
        "HST_WFC3_F160W",
    ]

    # for b in range(n_field):
    for b in [0]:

        print("********")
        print("field {0} (b={1})".format(field_names[b], b))
        print("********")

        # only create an AST input list if the ASTs don't already exist
        ast_input_file = (
            "./" + field_names[b] + "_beast/" + field_names[b] + "_beast_inputAST.txt"
        )
        if os.path.isfile(ast_input_file):
            print("AST input file already exists... skipping \n")
            continue

        # -----------------
        # data file names
        # -----------------

        # paths for the data/AST files
        gst_file = "./data/" + field_names[b] + ".st.fits"
        ast_file = "./data/" + field_names[b] + ".gst.fake.fits"

        # region file with catalog stars
        # make_ds9_region_file.region_file_fits(gst_file)
        # make_ds9_region_file.region_file_fits(ast_file)

        # -----------------
        # 1a. make magnitude histograms
        # -----------------

        print("")
        print("making magnitude histograms")
        print("")

        # if not os.path.isfile('./data/'+field_names[b]+'.gst_maghist.pdf'):
        peak_mags = plot_mag_hist.plot_mag_hist(gst_file, stars_per_bin=70, max_bins=75)
        # test = plot_mag_hist.plot_mag_hist(ast_file, stars_per_bin=200, max_bins=30)

        # -----------------
        # 1b. make a source density map
        # -----------------

        print("")
        print("making source density map")
        print("")

        # source density map
        sd_map = gst_file.replace(".fits", "_source_den_image.fits")
        if not os.path.isfile(sd_map):
            # if True:
            # - pixel size of 5 arcsec
            # - use ref_filter[b] between vega mags of 15 and peak_mags[ref_filter[b]]-0.5
            sourceden_args = types.SimpleNamespace(
                subcommand="sourceden",
                catfile=gst_file,
                pixsize=5,
                npix=None,
                mag_name=ref_filter[b] + "_VEGA",
                mag_cut=[15, peak_mags[ref_filter[b]] - 0.5],
                flag_name=flag_filter[b] + "_FLAG",
            )
            create_background_density_map.main_make_map(sourceden_args)

        # new file name with the source density column
        # gst_file_sd = gst_file.replace(".fits", "_with_sourceden.fits")

        with fits.open(sd_map) as hdu_sd:
            sd_data = hdu_sd[0].data[hdu_sd[0].data != 0]
            ast_n_bins = np.ceil((np.max(sd_data) - np.min(sd_data)) / 1.0)

        # -----------------
        # 4/5. edit photometry/AST catalogs
        # -----------------

        # remove sources that are
        # - in regions without full imaging coverage,
        # - flagged in flag_filter

        # print('')
        # print('editing photometry/AST catalogs')
        # print('')

        # gst_file_cut = gst_file.replace('.fits', '_with_sourceden_cut.fits')
        # ast_file_cut = ast_file.replace('.fits', '_cut.fits')

        # cut_catalogs.cut_catalogs(
        #    gst_file_sd, gst_file_cut,
        #    #ast_file, ast_file_cut,
        #    partial_overlap=True, flagged=True, flag_filter=flag_filter[b],
        #    region_file=True)

        # -----------------
        # 0. make beast settings file
        # -----------------

        print("")
        print("creating beast settings file")
        print("")

        # get the boundaries of the image
        boundary_ra = boundary_coord[b][0]
        boundary_dec = boundary_coord[b][1]
        # make an eroded version for ASTs (10 pix = 0.5")
        boundary_polygon = geometry.Polygon(
            [
                [float(boundary_ra[i]), float(boundary_dec[i])]
                for i in range(len(boundary_ra))
            ]
        )
        erode_polygon = boundary_polygon.buffer(-0.5 / 3600)
        boundary_ra_erode = [str(x) for x in erode_polygon.exterior.coords.xy[0]]
        boundary_dec_erode = [str(x) for x in erode_polygon.exterior.coords.xy[1]]

        create_beast_settings(
            gst_file,
            ast_file,
            gst_filter_names,
            beast_filter_names,
            ref_image=ast_ref_im[b],
            ast_n_bins=ast_n_bins,
            boundary_ra=boundary_ra_erode,
            boundary_dec=boundary_dec_erode,
        )
        # load in beast settings
        settings = beast_settings.beast_settings(
            "beast_settings_" + field_names[i] + ".txt"
        )

        # -----------------
        # 2. make physics model
        # -----------------

        print("")
        print("making physics model")
        print("")

        model_grid_file = "./{0}_beast/{0}_beast_seds.grid.hd5".format(field_names[b])

        # only make the physics model if it doesn't already exist
        if not os.path.isfile(model_grid_file):
            create_physicsmodel.create_physicsmodel(
                settings, nprocs=1, nsubs=settings.n_subgrid
            )

        # -----------------
        # 3. make ASTs
        # -----------------

        if not os.path.isfile(ast_file):
            # if True:
            if not os.path.isfile(ast_input_file):
                # if True:
                print("")
                print("creating artificial stars")
                print("")
                make_ast_inputs.make_ast_inputs(settings, flux_bin_method=True)

            # make a region file of the ASTs
            make_ds9_region_file.region_file_txt(ast_input_file)

            # make histograms of the fluxes
            plot_ast_histogram.plot_ast(ast_input_file, sed_grid_file=model_grid_files)

            print("\n**** go run ASTs for " + field_names[b] + "! ****\n")
            continue


def create_beast_settings(
    gst_file,
    ast_file,
    gst_filter_label,
    beast_filter_label,
    ref_image="None",
    ast_n_bins=5,
    boundary_ra=None,
    boundary_dec=None,
):
    """
    Create a beast settings file for the given field.  This will open the file to
    determine the filters present - the `*_filter_label` inputs are references
    to properly interpret the file's information.

    Parameters
    ----------
    gst_file : string
        the path+name of the GST file

    ast_file : string
        the path+name of the AST file

    gst_filter_label : list of strings
        Labels used to represent each filter in the photometry catalog

    beast_filter_label : list of strings
        The corresponding full labels used by the BEAST

    ref_image : string (default='None')
        path+name of image to use as reference for ASTs

    ast_n_bins : int (default=5)
        number of SD/bg bins over which to repeat SEDs

    boundary_ra/boundary_dec : lists of floats (default=None)
        list of coordinates to limit area over which ASTs are generated

    Returns
    -------
    nothing

    """

    # read in the catalog
    cat = Table.read(gst_file)
    # extract field name
    field_name = gst_file.split("/")[-1].split(".")[0]

    # get the list of filters
    filter_list_base = []
    filter_list_long = []
    for f in range(len(gst_filter_label)):
        filt_exist = [gst_filter_label[f] in c for c in cat.colnames]
        if np.sum(filt_exist) > 0:
            filter_list_base.append(gst_filter_label[f])
            filter_list_long.append(beast_filter_label[f])

    # read in the template settings file
    orig_file = open("beast_settings_template.py", "r")
    settings_lines = np.array(orig_file.readlines())
    orig_file.close()

    # write out an edited beast_settings
    new_file = open("beast_settings_"+field_name+".txt", "w")

    for i in range(len(settings_lines)):

        # replace project name with the field ID
        if settings_lines[i][0:10] == "project = ":
            new_file.write(
                'project = "' + field_name + '_beast"\n'
            )
        # obsfile
        elif settings_lines[i][0:10] == "obsfile = ":
            new_file.write('obsfile = "' + gst_file + '"\n')
        # AST file name
        elif settings_lines[i][0:10] == "astfile = ":
            new_file.write('astfile = "' + ast_file + '"\n')
        # BEAST filter names
        elif settings_lines[i][0:10] == "filters = ":
            new_file.write("filters = ['" + "','".join(filter_list_long) + "'] \n")
        # catalog filter names
        elif settings_lines[i][0:14] == "basefilters = ":
            new_file.write("basefilters = ['" + "','".join(filter_list_base) + "'] \n")
        # AST stuff
        elif settings_lines[i][0:20] == "ast_density_table = ":
            new_file.write(
                'ast_density_table = "'
                + gst_file.replace(".fits", "_sourceden_map.hd5")
                + '" \n'
            )
        elif settings_lines[i][0:13] == "ast_N_bins = ":
            new_file.write("ast_N_bins = " + str(ast_n_bins) + "\n")
        elif settings_lines[i][0:22] == "ast_reference_image = ":
            new_file.write('ast_reference_image = "' + ref_image + '" \n')
        elif settings_lines[i][0:21] == "ast_coord_boundary = ":
            if boundary_ra is not None:
                new_file.write(
                    "ast_coord_boundary = [np.array(["
                    + ", ".join(str(x) for x in boundary_ra)
                    + "]), \n"
                    + " " * 22
                    + "np.array(["
                    + ", ".join(str(x) for x in boundary_dec)
                    + "]) ] \n"
                )
        # none of those -> write line as-is
        else:
            new_file.write(settings_lines[i])

    new_file.close()


if __name__ == "__main__":

    beast_production_wrapper()
