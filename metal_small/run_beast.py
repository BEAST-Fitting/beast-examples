#!/usr/bin/env python
"""
Script to run the BEAST on the PHAT-like data.
"""

# system imports
import argparse

# BEAST imports
from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_obsmodel,
    run_fitting,
)
from beast.physicsmodel.grid import SEDGrid
from beast.observationmodel.observations import Observations
import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.fitting import trim_grid
from beast.tools import beast_settings


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--physicsmodel",
        help="Generate the physics model grid",
        action="store_true",
    )
    parser.add_argument(
        "-a", "--ast", help="Generate an input AST file", action="store_true"
    )
    parser.add_argument(
        "-o",
        "--observationmodel",
        help="Calculate the observation model (bias and noise)",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--trim",
        help="Trim the physics and observation model grids",
        action="store_true",
    )
    parser.add_argument(
        "-f", "--fit", help="Fit the observed data", action="store_true"
    )
    parser.add_argument(
        "-r", "--resume", help="Resume a fitting run", action="store_true"
    )
    args = parser.parse_args()

    # read in BEAST settings
    settings = beast_settings.beast_settings("beast_settings.txt")

    if args.physicsmodel:

        create_physicsmodel.create_physicsmodel(
            settings, nsubs=settings.n_subgrid, nprocs=1,
        )

    if args.ast:

        make_ast_inputs.make_ast_inputs(settings, flux_bin_method=False)

    if args.observationmodel:
        print("Generating noise model from ASTs and absflux A matrix")

        create_obsmodel.create_obsmodel(
            settings, use_sd=False, nsubs=settings.n_subgrid, nprocs=1, use_rate=True,
        )

        # in the absence of ASTs, the splinter noise model can be used
        # instead of the toothpick model above
        #  **warning** not very realistic
        # import beast.observationmodel.noisemodel.splinter as noisemodel
        #
        # modelsedgridfile = settings.project + '/' + settings.project + \
        #    '_seds.grid.hd5'
        # modelsedgrid = FileSEDGrid(modelsedgridfile)
        #
        # noisemodel.make_splinter_noise_model(
        #    settings.noisefile,
        #    modelsedgrid,
        #    absflux_a_matrix=settings.absflux_a_matrix)

    if args.trim:
        print("Trimming the model and noise grids")

        # read in the observed data
        obsdata = Observations(
            settings.obsfile, settings.filters, settings.obs_colnames
        )

        # get the modesedgrid on which to generate the noisemodel
        modelsedgridfile = settings.project + "/" + settings.project + "_seds.grid.hd5"
        modelsedgrid = SEDGrid(modelsedgridfile)

        # read in the noise model just created
        noisemodel_vals = noisemodel.get_noisemodelcat(settings.noisefile)

        # trim the model sedgrid
        sed_trimname = "{0}/{0}_seds_trim.grid.hd5".format(settings.project)
        noisemodel_trimname = "{0}/{0}_noisemodel_trim.grid.hd5".format(
            settings.project
        )

        trim_grid.trim_models(
            modelsedgrid,
            noisemodel_vals,
            obsdata,
            sed_trimname,
            noisemodel_trimname,
            sigma_fac=3.0,
        )

    if args.fit:

        run_fitting.run_fitting(
            settings,
            use_sd=False,
            nsubs=settings.n_subgrid,
            nprocs=1,
            pdf2d_param_list=["Av", "M_ini", "logT"],
        )

    if args.resume:

        run_fitting.run_fitting(
            settings, use_sd=False, nsubs=settings.n_subgrid, nprocs=1, resume=True
        )

    # print help if no arguments
    if not any(vars(args).values()):
        parser.print_help()
