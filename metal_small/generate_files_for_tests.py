import os
import copy
import numpy as np
import argparse
import asdf

# BEAST imports
from beast.tools.run import (
    create_filenames,
    create_physicsmodel,
    make_ast_inputs,
    create_obsmodel,
    run_fitting,
    merge_files,
)
from beast.physicsmodel.grid import SEDGrid
from beast.observationmodel.observations import Observations
import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.fitting import trim_grid
from beast.tools import (
    beast_settings,
    compare_spec_type,
    star_type_probability,
)


def generate_files_for_tests(run_beast=True, run_tools=True):
    """
    Use the metal_small example to generate a full set of files for the BEAST
    regression tests.

    Parameters
    ----------
    run_beast : boolean (default=True)
        if True, run the BEAST

    run_tools : boolean (default=True)
        if True, run the code to generate things for tools
    """

    # read in BEAST settings
    settings_orig = beast_settings.beast_settings("beast_settings.txt")
    # also make a version with subgrids
    settings_subgrids = copy.deepcopy(settings_orig)
    settings_subgrids.n_subgrid = 2
    settings_subgrids.project = f"{settings_orig.project}_subgrids"

    # ==========================================
    # run the beast for each set of settings
    # ==========================================

    if run_beast:

        for settings in [settings_orig, settings_subgrids]:

            # -----------------
            # physics model
            # -----------------
            create_physicsmodel.create_physicsmodel(
                settings, nsubs=settings.n_subgrid, nprocs=1,
            )

            # -----------------
            # ASTs
            # -----------------

            # currently only works for no subgrids
            if settings.n_subgrid == 1:
                make_ast_inputs.make_ast_inputs(settings, pick_method="flux_bin_method")

            # -----------------
            # obs model
            # -----------------
            create_obsmodel.create_obsmodel(
                settings,
                use_sd=False,
                nsubs=settings.n_subgrid,
                nprocs=1,
                use_rate=True,
            )

            # -----------------
            # trimming
            # -----------------

            # make file names
            file_dict = create_filenames.create_filenames(
                settings, use_sd=False, nsubs=settings.n_subgrid
            )

            # read in the observed data
            obsdata = Observations(
                settings.obsfile, settings.filters, settings.obs_colnames
            )

            for i in range(settings.n_subgrid):

                # get the modesedgrid on which to generate the noisemodel
                modelsedgridfile = file_dict["modelsedgrid_files"][i]
                modelsedgrid = SEDGrid(modelsedgridfile)

                # read in the noise model just created
                noisemodel_vals = noisemodel.get_noisemodelcat(
                    file_dict["noise_files"][i]
                )

                # trim the model sedgrid
                sed_trimname = file_dict["modelsedgrid_trim_files"][i]
                noisemodel_trimname = file_dict["noise_trim_files"][i]

                trim_grid.trim_models(
                    modelsedgrid,
                    noisemodel_vals,
                    obsdata,
                    sed_trimname,
                    noisemodel_trimname,
                    sigma_fac=3.0,
                )

            # -----------------
            # fitting
            # -----------------

            run_fitting.run_fitting(
                settings,
                use_sd=False,
                nsubs=settings.n_subgrid,
                nprocs=1,
                pdf2d_param_list=["Av", "M_ini", "logT"],
                pdf_max_nbins=50,
            )

            # -----------------
            # merging
            # -----------------

            # it'll automatically skip for no subgrids
            merge_files.merge_files(settings, use_sd=False, nsubs=settings.n_subgrid)

            print("\n\n")

    # ==========================================
    # reference files for assorted tools
    # ==========================================

    if run_tools:

        # -----------------
        # compare_spec_type
        # -----------------

        # the input settings
        input = {
            "spec_ra": [72.67213351],
            "spec_dec": [-67.71720515],
            "spec_type": ["A"],
            "spec_subtype": [0],
            "lumin_class": ["IV"],
            "match_radius": 0.2,
        }

        # run it
        output = compare_spec_type.compare_spec_type(
            settings_orig.obsfile,
            "{0}/{0}_stats.fits".format(settings_orig.project),
            **input,
        )

        # save the inputs and outputs
        asdf.AsdfFile({"input": input, "output": output}).write_to(
            "{0}/{0}_compare_spec_type.asdf".format(settings_orig.project)
        )

        # -----------------
        # star_type_probability
        # -----------------


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_beast", type=int, default=1, help="if True (1), run the BEAST",
    )
    parser.add_argument(
        "--run_tools",
        type=int,
        default=1,
        help="if True (1), run the code to generate things for tools",
    )

    args = parser.parse_args()
    generate_files_for_tests(
        run_beast=args.run_beast, run_tools=args.run_tools,
    )
