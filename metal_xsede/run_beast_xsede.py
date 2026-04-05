import numpy as np
import glob
import subprocess
import sys
import os
import shutil
import types
import re

from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_obsmodel,
    make_trim_scripts,
    run_fitting,
    merge_files,
    create_filenames,
)

from beast.tools import (
    beast_settings,
    create_background_density_map,
    subdivide_obscat_by_source_density,
    split_catalog_using_map,
    cut_catalogs,
    write_sbatch_file,
    setup_batch_beast_fit,
    # star_type_probability,
    # compare_spec_type,
    reorder_beast_results_spatial,
    condense_beast_results_spatial,
)
from beast.plotting import (
    plot_mag_hist,
    make_ds9_region_file,
    plot_chi2_hist,
    plot_cmd_with_fits,
    plot_triangle,
    plot_indiv_pdfs,
    # plot_completeness,
)


from astropy.table import Table
from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


def beast_production_wrapper():
    """
    This does all of the steps for a full production run, and can be used as
    a wrapper to automatically do most steps for multiple fields.
    1. make source density and background maps
    2. make beast settings file
    3. make sbatch script for physics model (SED grid)
    4. make quality cuts to photometry & fake stars
    5. split catalog by either source density or background
    6. make sbatch script for noise model
    7. make sbatch script for trimming models
    8. make sbatch script for fitting models
    9. make sbatch script to merge output files
    10. make sbatch script for running some analysis

    """

    # get the list of METAL fields
    metal_info = Table.read("metal_images_by_field.txt", format="ascii")

    field_names = metal_info["field"]
    ref_filter = []
    flag_filter = []
    im_path = []

    for i in range(len(metal_info)):
        if not np.ma.is_masked(metal_info["F475W"][i]):
            ref_filter.append("F475W")
            flag_filter.append("F475W")
            im_path.append(metal_info["F475W"][i])
        elif not np.ma.is_masked(metal_info["F814W"][i]):
            ref_filter.append("F814W")
            flag_filter.append("F814W")
            im_path.append(metal_info["F814W"][i])
        elif not np.ma.is_masked(metal_info["F110W"][i]):
            ref_filter.append("F110W")
            flag_filter.append("F110W")
            im_path.append(metal_info["F110W"][i])
        else:
            print("no matching filter info for " + field_names[i])
            ref_filter.append(None)
            flag_filter.append(None)
            im_path.append(None)

    # number of fields (46)
    n_field = len(field_names)

    # notable fields (by index)
    # 0 : 14675_LMC-13361nw-11112
    # 14 : 14675_LMC-4916ne-18087
    # 20 : 14675_LMC-5665ne-12232
    # 28 : 14675_LMC-8576ne-10141
    # 32 : last of the 14675 fields
    # 33 : first of the 12581 fields

    # Need to know what the correspondence is between filter names in the
    # catalog and the BEAST filter names.  Each filter has a column in
    # metal_info to tell what the corresponding BEAST filter name is for each
    # image.
    gst_filter_names = [
        "F225W",
        "F275W",
        "F336W",
        "F475W",
        "F814W",
        "F110W",
        "F128N",
        "F160W",
    ]

    # keep track of what needs to be submittted
    sbatch_list = []

    for b in range(n_field):
        # for b in [14]:
        # for b in [0]:

        print("********")
        print("field " + field_names[b])
        print("********\n")

        # -----------------
        # data file names
        # -----------------

        # paths for the data/AST files
        gst_file = "./data/" + field_names[b] + ".gst.fits"
        ast_file = "./data/" + field_names[b] + ".gst.fake.fits"

        if not os.path.isfile(ast_file):
            print("no AST file for this field")
            continue

        print("")
        print("copying data fits files over")
        print("")
        # - point source catalog
        if os.path.realpath(gst_file) != os.path.abspath(metal_info["phot_cat"][b]):
            # subprocess.call('cp ' + metal_info['phot_cat'][b] + ' ./data/', shell=True)
            os.symlink(os.path.abspath(metal_info["phot_cat"][b]), gst_file)
        # - artifical star tests
        if os.path.realpath(ast_file) != os.path.abspath(metal_info["ast_cat"][b]):
            # subprocess.call('cp ' + metal_info['ast_cat'][b] + ' ./data/', shell=True)
            os.symlink(os.path.abspath(metal_info["ast_cat"][b]), ast_file)
        # - ref images
        im_file = "./data/" + field_names[b] + "_" + ref_filter[b] + ".fits"
        if os.path.realpath(im_file) != os.path.abspath(im_path[b]):
            # subprocess.call('cp ' + im_path[b] + ' ' + im_file, shell=True)
            os.symlink(os.path.abspath(im_path[b]), im_file)

        # region file with catalog stars
        # make_ds9_region_file.region_file_fits(gst_file)

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
        print("making source density and background maps")
        print("")

        # background map
        bg_map = gst_file.replace(".fits", "_background.fits")
        if not os.path.isfile(bg_map):
            # if True:
            background_args = types.SimpleNamespace(
                subcommand="background",
                catfile=gst_file,
                pixsize=5,
                npix=None,
                reference=im_file,
                mask_radius=10,
                ann_width=20,
                cat_filter=[ref_filter[b], "90"],
            )
            create_background_density_map.main_make_map(background_args)

        # source density map
        sd_map = gst_file.replace(".fits", "_source_den_image.fits")
        if not os.path.isfile(sd_map):
            # if True:
            # - pixel size of 5 arcsec
            # - use ref_filter[b] between vega mags of 17 and peak_mags[ref_filter[b]]-0.5
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
        gst_file_sd = gst_file.replace(".fits", "_with_sourceden.fits")
        # new file name with the background columns
        gst_file_bg = gst_file.replace(".fits", "_with_bg.fits")

        # figure out dynamic range of each map
        with fits.open(bg_map) as hdu_bg, fits.open(sd_map) as hdu_sd:
            bg_data = hdu_bg[0].data[hdu_bg[0].data != 0]
            sd_data = hdu_sd[0].data[hdu_sd[0].data != 0]
            bg_p05, bg_p16, bg_p84, bg_p95 = np.percentile(bg_data, [5, 16, 84, 95])
            sd_p05, sd_p16, sd_p84, sd_p95 = np.percentile(sd_data, [5, 16, 84, 95])
            iqr_bg = bg_p84 - bg_p16
            iqr_sd = sd_p84 - sd_p16
            print("iqrs (bg,SD): {0:.3f}, {1:.2f}".format(iqr_bg, iqr_sd))
            # n_bg_bins = (np.max(bg_data) - np.min(bg_data))/0.05
            # n_sd_bins = (np.max(sd_data) - np.min(sd_data))/0.5
            n_bg_bins = (bg_p95 - bg_p05) / 0.1
            n_sd_bins = (sd_p95 - sd_p05) / 1
            print("n_bins (bg,SD): {0}, {1}".format(n_bg_bins, n_sd_bins))

            if n_bg_bins > n_sd_bins:
                print("using background map")
                bin_table = "background"
                bin_width = 0.1
                ast_n_bins = np.ceil((np.max(bg_data) - np.min(bg_data)) / bin_width)
            else:
                print("using source density map")
                bin_table = "sourceden"
                bin_width = 1.0
                ast_n_bins = np.ceil((np.max(sd_data) - np.min(sd_data)) / bin_width)

        # -----------------
        # 2. make beast settings file
        # -----------------

        print("")
        print("creating beast settings file")
        print("")

        filter_sublist = [
            f for f in gst_filter_names if not np.ma.is_masked(metal_info[f][b])
        ]
        beast_filter_sublist = [
            metal_info[f + "_beast"][b]
            for f in gst_filter_names
            if not np.ma.is_masked(metal_info[f][b])
        ]

        settings_file = create_beast_settings(
            gst_file,
            ast_file,
            filter_sublist,
            beast_filter_sublist,
            ref_image=im_file,
            bin_table=bin_table,
            ast_n_bins=ast_n_bins,
        )

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(settings_file)

        # make a project directory
        proj_dir = "./{0}/".format(settings.project)
        if not os.path.isdir(proj_dir):
            os.mkdir(proj_dir)

        # -----------------
        # 3. make physics model
        # -----------------

        print("")
        print("making physics model")
        print("")

        # master SED files
        if "LMC" in gst_file:
            master_sed_files = [
                "./mastergrid_LMC/mastergrid_LMC_seds.gridsub" + str(i) + ".hd5"
                for i in range(settings.n_subgrid)
            ]
        if "SMC" in gst_file:
            master_sed_files = [
                "./mastergrid_SMC/mastergrid_SMC_seds.gridsub" + str(i) + ".hd5"
                for i in range(settings.n_subgrid)
            ]

        # need to have list of the model grid file names to make
        model_grid_files = [
            "./{0}/{0}_seds.gridsub{1}.hd5".format(settings.project, i)
            for i in range(settings.n_subgrid)
        ]

        # copy from the master
        for i, sed_file in enumerate(model_grid_files):

            # grid doesn't exist -> script to copy over from master grid
            if not os.path.isfile(sed_file):

                # file name for script
                fname = "./{0}/model_batch_jobs/copy_gridsub{1}.script".format(
                    settings.project, i
                )

                # make sure the directory exists
                fname_path = os.path.dirname(fname)
                if not os.path.isdir(fname_path):
                    os.mkdir(fname_path)
                log_path = os.path.join(fname_path, "logs")
                if not os.path.isdir(log_path):
                    os.mkdir(log_path)

                # write out the sbatch file
                write_sbatch_file.write_sbatch_file(
                    fname,
                    (
                        "python -m beast.tools.remove_filters {0}"
                        " --physgrid {1} --physgrid_outfile {2} "
                        " --beast_filt {3}"
                        " >> {4}/rem_filt_{5}.log".format(
                            gst_file,
                            master_sed_files[i],
                            sed_file,
                            " ".join(beast_filter_sublist),
                            log_path,
                            i,
                        )
                    ),
                    "/pylon5/as5pi7p/lhagen",
                    modules=["module load anaconda3", "source activate bdev"],
                    stdout_file="/pylon5/as5pi7p/lhagen/{0}/model_batch_jobs/logs/%j.out".format(
                        settings.project
                    ),
                    run_time="10:00:00",
                    mem="{0:.0f}GB".format(
                        os.path.getsize(master_sed_files[i]) / 10 ** 9 * 2.9
                    ),
                )
                sbatch_list.append("sbatch " + fname)
                print("sbatch " + fname)

        # make a file that contains the SED file names
        with open("./{0}/subgrid_fnames.txt".format(settings.project), "w") as fname:
            for sed_file in model_grid_files:
                fname.write(sed_file + "\n")

        # now let's check if we needed to make any more
        sed_files = glob.glob(
            "./"
            + field_names[b]
            + "_beast/"
            + field_names[b]
            + "_beast_seds.gridsub*.hd5"
        )
        if len(sed_files) < settings.n_subgrid:
            print("\n**** go run physics model code for " + field_names[b] + "! ****")
            continue

        # -----------------
        # make ASTs
        # -----------------

        # -- ALREADY DONE --

        # -----------------
        # 4. edit photometry/AST catalogs
        # -----------------

        # remove sources that are
        # - in regions without full imaging coverage,
        # - flagged in flag_filter

        print("")
        print("editing photometry/AST catalogs")
        print("")

        if bin_table == "sourceden":
            gst_to_use = gst_file_sd
            gst_file_cut = gst_file.replace(".fits", "_with_sourceden_cut.fits")
        if bin_table == "background":
            gst_to_use = gst_file_bg
            gst_file_cut = gst_file.replace(".fits", "_with_bg_cut.fits")

        ast_file_cut = ast_file.replace(".fits", "_cut.fits")

        cut_catalogs.cut_catalogs(
            gst_to_use,
            gst_file_cut,
            ast_file,
            ast_file_cut,
            partial_overlap=True,
            flagged=True,
            flag_filter=flag_filter[b],
            region_file=True,
        )

        # edit the settings file to have the correct photometry file name
        settings_file = create_beast_settings(
            gst_file_cut,
            ast_file_cut,
            filter_sublist,
            beast_filter_sublist,
            ref_image=im_file,
            bin_table=bin_table,
            ast_n_bins=ast_n_bins,
        )

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(settings_file)

        # -----------------
        # 5. split observations
        # -----------------

        print("")
        print("splitting observations by " + bin_table)
        print("")

        split_catalog_using_map.split_main(
            gst_file_cut,
            ast_file_cut,
            gst_file.replace(".fits", "_" + bin_table + "_map.hd5"),
            bin_width=bin_width,
            n_per_file=1000,
            min_n_subfile=10,
        )

        # check for pathological cases of AST bins not matching photometry bins
        gst_list = glob.glob(gst_file_cut.replace(".fits", "*bin?.fits"))
        ast_list = glob.glob(ast_file_cut.replace(".fits", "*bin?.fits"))
        if len(gst_list) != len(ast_list):
            for a in ast_list:
                # the bin number for this AST file
                bin_num = a[a.rfind("_") + 1 : -5]
                # if this bin number doesn't have a corresponding gst file,
                # delete the AST bin file
                if np.sum([bin_num in g for g in gst_list]) == 0:
                    print("removing " + a)
                    os.remove(a)

        # -- at this point, we can run create_filenames to make final lists of filenames
        file_dict = create_filenames.create_filenames(
            settings, use_sd=True, nsubs=settings.n_subgrid
        )

        # figure out how many files there are
        sd_sub_info = file_dict["sd_sub_info"]
        # - number of SD bins
        temp = set([i[0] for i in sd_sub_info])
        print("** total SD bins: " + str(len(temp)))
        # - the unique sets of SD+sub
        unique_sd_sub = [
            x for i, x in enumerate(sd_sub_info) if i == sd_sub_info.index(x)
        ]
        print("** total SD subfiles: " + str(len(unique_sd_sub)))

        # -----------------
        # 6. make noise models
        # -----------------

        print("")
        print("making noise models")
        print("")

        # expected final list of noise files
        noisefile_list = list(set(file_dict["noise_files"]))
        # the current existing ones (if any)
        existing_noisefiles = glob.glob(
            "{0}/{0}_noisemodel_bin*.gridsub*.hd5".format(settings.project)
        )

        # if we don't have all of them yet, write script to make them
        if len(existing_noisefiles) < len(noisefile_list):

            noise_dir = "./{0}/noise_logs".format(settings.project)
            if not os.path.isdir(noise_dir):
                os.mkdir(noise_dir)

            cmd = (
                f"python -m beast.tools.run.create_obsmodel {settings_file} "
                + f"--nsubs {settings.n_subgrid} --use_sd --nprocs 1 "
                + "--subset ${SLURM_ARRAY_TASK_ID} $((${SLURM_ARRAY_TASK_ID} + 1)) "
                + f">> {noise_dir}/create_noisemodel_${{SLURM_ARRAY_TASK_ID}}.log"
            )
            fname = f"{settings.project}/create_noisemodels.script"

            write_sbatch_file.write_sbatch_file(
                fname,
                cmd,
                "/pylon5/as5pi7p/lhagen",
                modules=["module load anaconda3", "source activate bdev"],
                job_name="beast_LH",
                stdout_file="{0}/%A_%a.out".format(os.path.abspath(noise_dir)),
                queue="LM",
                run_time="20:00:00",
                mem="{0:.0f}GB".format(
                    os.path.getsize(file_dict["modelsedgrid_files"][0]) / 10 ** 9 * 3
                ),
                array=[0, settings.n_subgrid - 1],
            )

            sbatch_list.append("sbatch " + fname)

            print(
                "*** go run {0}/create_noisemodels.script ***".format(settings.project)
            )

            continue

        # plot completeness
        if False:
            print("plotting completeness")
            plot_completeness.plot_completeness(
                file_dict["modelsedgrid_files"][0],
                file_dict["noise_files"][0],
                field_names[b] + "_completeness.pdf",
            )

        # -----------------
        # 7. make script to trim models
        # -----------------

        print("")
        print("setting up script to trim models")
        print("")

        job_file_list = make_trim_scripts.make_trim_scripts(
            settings, num_subtrim=1, prefix=None
        )

        if len(job_file_list) > 0:
            print("\n**** go run trimming code for " + field_names[b] + "! ****")

            fname = "{0}/trim_files.script".format(settings.project)

            write_sbatch_file.write_sbatch_file(
                fname,
                '{0}/trim_batch_jobs/BEAST_gridsub"${{SLURM_ARRAY_TASK_ID}}"_batch_trim.joblist'.format(
                    settings.project
                ),
                "/pylon5/as5pi7p/lhagen",
                modules=["module load anaconda3", "source activate bdev"],
                job_name="beast_LH",
                stdout_file="/pylon5/as5pi7p/lhagen/{0}/trim_batch_jobs/logs/%A_%a.out".format(
                    settings.project
                ),
                queue="LM",
                run_time="{0:.0f}:00:00".format(
                    3 * len(set([tuple(x) for x in file_dict["sd_sub_info"]]))
                ),
                mem="{0:.0f}GB".format(
                    os.path.getsize(file_dict["modelsedgrid_files"][0]) / 10 ** 9 * 5
                ),
                array=[0, len(job_file_list) - 1],
            )

            print(f"sbatch {fname}")
            sbatch_list.append("sbatch " + fname)

            continue

        else:
            print("all files are trimmed for " + field_names[b])

        # -----------------
        # 8. make script to fit models
        # -----------------

        print("")
        print("setting up script to fit models")
        print("")

        if False:

            fit_run_info = setup_batch_beast_fit.setup_batch_beast_fit(
                settings,
                num_percore=1,
                overwrite_logfile=False,
                # pdf2d_param_list=['Av', 'Rv', 'f_A', 'M_ini', 'logA', 'Z', 'distance','logT', 'logg'],
                pdf2d_param_list=["Av", "M_ini", "logT"],
                # prefix='source activate bdev',
                use_sd=True,
                nsubs=settings.n_subgrid,
                nprocs=1,
            )

            # check if the fits exist before moving on
            tot_remaining = len(fit_run_info["done"]) - np.sum(fit_run_info["done"])
            if tot_remaining > 0:
                print("\n**** go run fitting code for " + field_names[b] + "! ****")

                fname = "{0}/run_fitting.script".format(settings.project)

                write_sbatch_file.write_sbatch_file(
                    fname,
                    '{0}/fit_batch_jobs/beast_batch_fit_"${{SLURM_ARRAY_TASK_ID}}".joblist'.format(
                        settings.project
                    ),
                    "/pylon5/as5pi7p/lhagen",
                    modules=["module load anaconda3", "source activate bdev"],
                    job_name="beast_LH",
                    stdout_file="/pylon5/as5pi7p/lhagen/{0}/fit_batch_jobs/logs/%A_%a.out".format(
                        settings.project
                    ),
                    queue="LM",
                    run_time="20:00:00",
                    mem="{0:.0f}GB".format(
                        os.path.getsize(file_dict["modelsedgrid_files"][0])
                        / 10 ** 9
                        * 1.5
                    ),
                    array=[1, tot_remaining],
                )

                sbatch_list.append("sbatch " + fname)

                # also write out a file to do partial merging in case that
                # ends up being useful
                write_sbatch_file.write_sbatch_file(
                    "{0}/merge_files_partial.script".format(settings.project),
                    f"python -m beast.tools.run.merge_files {settings_file} --use_sd 1 --nsubs {settings.n_subgrid} --partial 1",
                    "/pylon5/as5pi7p/lhagen",
                    modules=["module load anaconda3", "source activate bdev"],
                    stdout_file="/pylon5/as5pi7p/lhagen/{0}/fit_batch_jobs/logs/%j.out".format(
                        settings.project
                    ),
                    run_time="2:00:00",
                    mem="128GB",
                )

                continue
            else:
                print("all fits are complete for " + field_names[b])

        # -----------------
        # 9. merge stats files from each fit
        # -----------------

        print("")
        print("merging files")
        print("")

        # use the merged stats file to decide if merging is complete
        merged_stats_file = "{0}_beast/{0}_beast_stats.fits".format(field_names[b])

        if not os.path.isfile(merged_stats_file):

            # write out the sbatch file
            fname = "{0}/merge_files.script".format(settings.project)
            write_sbatch_file.write_sbatch_file(
                fname,
                f"python -m beast.tools.run.merge_files {settings_file} --use_sd 1 --nsubs {settings.n_subgrid}",
                "/pylon5/as5pi7p/lhagen",
                modules=["module load anaconda3", "source activate bdev"],
                stdout_file="/pylon5/as5pi7p/lhagen/{0}/fit_batch_jobs/logs/%j.out".format(
                    settings.project
                ),
                run_time="2:00:00",
                mem="128GB",
            )

            sbatch_list.append("sbatch " + fname)

            continue

        # -----------------
        # make some plots
        # -----------------

        # print('')
        # print('making some plots')
        # print('')

        # chi2 histogram
        # plot_chi2_hist.plot(stats_filebase+'_stats.fits', n_bins=100)
        # CMD color-coded by chi2
        # plot_cmd_with_fits.plot(gst_file, stats_filebase+'_stats.fits',
        #                            mag1_filter='F475W', mag2_filter='F814W', mag3_filter='F475W',
        #                            param='chi2min', log_param=True)

        #'F275W','F336W','F390M','F555W','F814W','F110W','F160W'

        # -----------------
        # reorganize results into spatial regions
        # -----------------

        # print('')
        # print('doing spatial reorganizing')
        # print('')

        # region_filebase = './' + field_names[b] + '_beast/' + field_names[b] + '_beast_sd'
        # output_filebase = './' + field_names[b] + '_beast/spatial/' + field_names[b]

        # reorder_beast_results_spatial.reorder_beast_results_spatial(stats_filename=stats_filebase + '_stats.fits',
        #                                                                region_filebase=region_filebase,
        #                                                                output_filebase=output_filebase)

        # condense_beast_results_spatial.condense_files(filedir='./' + field_names[b] + '_beast/spatial/')

        # -----------------
        # 10. some sciency things
        # -----------------

        print("")
        print("doing some science")
        print("")

        cmd_list = []

        # naive maps
        if not os.path.isfile(merged_stats_file.replace("stats", "mapAv")):
            cmd_list.append(
                f"python -m megabeast.make_naive_maps {merged_stats_file} --pix_size 10"
            )
        # naive IMF
        if not os.path.isfile("{0}/{0}_imf.pdf".format(settings.project)):
            cmd_list.append(
                f"python -m megabeast.make_naive_imf {settings_file} --use_sd 1 --compl_filter {ref_filter[b]}"
            )

        # if there are things to make, write out sbatch file
        if len(cmd_list) > 0:

            fname = "{0}/run_science.script".format(settings.project)

            write_sbatch_file.write_sbatch_file(
                fname,
                cmd_list,
                "/pylon5/as5pi7p/lhagen",
                modules=["module load anaconda3", "source activate bdev"],
                stdout_file="/pylon5/as5pi7p/lhagen/{0}/fit_batch_jobs/logs/%j.out".format(
                    settings.project
                ),
                run_time="12:00:00",
                mem="128GB",
            )

            sbatch_list.append("sbatch " + fname)

    # write out all the sbatch commands to run
    with open("sbatch_commands.script", "w") as f:
        for cmd in sbatch_list:
            f.write(cmd + "\n")


def create_beast_settings(
    gst_file,
    ast_file,
    gst_filter_label,
    beast_filter_label,
    ref_image="None",
    bin_table="sourceden",
    ast_n_bins=5,
):
    """
    Create a beast settings file for the given field.

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

    bin_table : string (default='sourceden')
        choose 'sourceden' or 'background' to use either the source density or
        background level for noise model bins

    ast_n_bins : int (default=5)
        number of SD/bg bins over which to repeat SEDs

    Returns
    -------
    settings_file : str
        name of settings file

    """

    # read in the catalog
    # cat = Table.read(gst_file)
    # extract field name
    field_name = gst_file.split("/")[-1].split(".")[0]

    # get the list of filters
    filter_list_base = gst_filter_label
    filter_list_long = beast_filter_label

    # read in the template settings file
    if "SMC" in gst_file:
        orig_filename = "beast_settings_template_SMC.txt"
    if "LMC" in gst_file:
        orig_filename = "beast_settings_template_LMC.txt"
    with open(orig_filename, "r") as orig_file:
        settings_lines = np.array(orig_file.readlines())

    # write out an edited settings file
    settings_file = "beast_settings_" + field_name + ".txt"
    new_file = open(settings_file, "w")

    for i in range(len(settings_lines)):

        # replace project name with the field ID
        if settings_lines[i][0:10] == "project = ":
            new_file.write('project = "' + field_name + '_beast"\n')
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
                + gst_file.replace(".fits", "_" + bin_table + "_map.hd5")
                + '" \n'
            )
        elif settings_lines[i][0:13] == "ast_N_bins = ":
            new_file.write("ast_N_bins = " + str(ast_n_bins) + "\n")
        elif settings_lines[i][0:22] == "ast_reference_image = ":
            new_file.write('ast_reference_image = "' + ref_image + '" \n')
        # none of those -> write line as-is
        else:
            new_file.write(settings_lines[i])

    new_file.close()

    return settings_file


def make_mastergrid():
    """
    Setup sbatch files to make master grids for LMC and SMC
    """

    for gal_name in ["LMC", "SMC"]:

        settings_file = f"beast_settings_{gal_name}_mastergrid.txt"

        # read in the settings
        settings = beast_settings.beast_settings(settings_file)

        # make physics model scripts
        create_physicsmodel.split_create_physicsmodel(
            settings, nprocs=1, nsubs=settings.n_subgrid
        )

        # make an sbatch file for them
        write_sbatch_file.write_sbatch_file(
            f"create_{gal_name}_mastergrid.script",
            f'./mastergrid_{gal_name}/model_batch_jobs/create_physicsmodel_"${{SLURM_ARRAY_TASK_ID}}".job',
            "/pylon5/as5pi7p/lhagen",
            modules=["module load anaconda3", "source activate bdev"],
            job_name=f"{gal_name}grid",
            stdout_file=f"/pylon5/as5pi7p/lhagen/mastergrid_{gal_name}/model_batch_jobs/logs/%A_%a.out",
            egress=True,
            queue="LM",
            run_time="40:00:00",
            mem="570GB",
            array=[0, settings.n_subgrid - 1],
        )


if __name__ == "__main__":

    beast_production_wrapper()
