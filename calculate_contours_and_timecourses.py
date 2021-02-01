import h5py

import numpy as np

import wueric_sta_tools

from tqdm import tqdm

import pickle

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculated derived quantities from STAs')
    parser.add_argument("sta_path", type=str, help='path to STA hdf5')
    parser.add_argument("output_path", type=str, help='output pickle path')
    parser.add_argument("--sig_stixel_cutoff", "-s", type=float, default=5.0, help='sig stixel cutoff')
    parser.add_argument("--border_trim", "-b", type=int, default=None, help="border trim")

    args = parser.parse_args()

    sig_stixel_cutoff = args.sig_stixel_cutoff
    border_trim = args.border_trim

    print("Calculating average timecourse")
    timecourse_average = None
    n_sig_stixels = 0
    with h5py.File(args.sta_path, 'r') as h5_file:

        with tqdm(total=len(list(h5_file.keys()))) as pbar:
            for key in h5_file.keys():

                if border_trim is not None:
                    sta_matrix = np.array(h5_file[key])[:, border_trim:-border_trim, border_trim:-border_trim, :]
                else:
                    sta_matrix = np.array(h5_file[key])

                timepoints, width, height, should_be_3 = sta_matrix.shape
                x = wueric_sta_tools.greg_field_simpler_significant_stixels_rgb_matrix(sta_matrix, sig_stixel_cutoff)

                if timecourse_average is None:
                    timecourse_average = np.zeros((timepoints, should_be_3))

                if np.any(x):
                    sig_stixels = np.nonzero(x)

                    sig_stixel_timecourses = sta_matrix[:, sig_stixels[0], sig_stixels[1], :]
                    summed_timecourses = np.sum(sig_stixel_timecourses, axis=(1,))
                    timecourse_average += summed_timecourses

                    n_sig_stixels += sig_stixel_timecourses.shape[1]
                pbar.update(1)

    timecourse_average = timecourse_average / n_sig_stixels
    timecourse_average = timecourse_average / np.max(timecourse_average)

    print("Calculating STA spatial contours")
    contour_by_cell_id = {}
    centers_by_cell_id = {}
    with h5py.File(args.sta_path, 'r') as h5_file:

        with tqdm(total=len(list(h5_file.keys()))) as pbar:
            for key in h5_file.keys():
                if border_trim is not None:
                    sta_matrix = np.array(h5_file[key])[:, border_trim:-border_trim, border_trim:-border_trim, :]
                else:
                    sta_matrix = np.array(h5_file[key])

                timepoints, width, height, should_be_3 = sta_matrix.shape

                contour_mat = np.squeeze(wueric_sta_tools.find_spatial_sta_fit_regression_sum_channels_rgb_matrix(
                    sta_matrix,
                    timecourse_average), axis=2)

                center = wueric_sta_tools.greg_field_calculate_weighted_centers_of_mass(contour_mat,
                                                                                        contour_mat > 0.25)

                contour_by_cell_id[int(key)] = contour_mat
                centers_by_cell_id[int(key)] = np.array(center)
                pbar.update(1)

    print("Saving output")
    with open(args.output_path, 'wb') as pfile:

        output_dict = {
            'timecourse' : timecourse_average.T,
            'contours_by_cell_id' : contour_by_cell_id,
            'centers_by_cell_id' : centers_by_cell_id
        }

        pickle.dump(output_dict, pfile)


