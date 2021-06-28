from lib.torch_sta import bin_frames_by_spike_times
from lib.save_data import generate_save_dict

import torch

import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

from typing import List

import pickle
import argparse

import numpy as np

import h5py

CELL_BATCH_SIZE = 512
N_DISPLAY_FRAMES_PER_TTL = 100
SAMPLE_FREQ = 20000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute STAs at arbitrary frequency (frame rate and frame instabilities do not matter!)')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('xml_path', type=str, help='path to stimulus XML file')
    parser.add_argument('output', type=str, help='path to save location')
    parser.add_argument('-r', '--frame_rate', type=float, help='effective frame rate [Hz]', default=120.0)
    parser.add_argument('-n', '--n_frames', type=int, help='STA depth', default=51)
    parser.add_argument('-b', '--batch', type=int, help='number of cells batch size', default=CELL_BATCH_SIZE)
    parser.add_argument('-j', '--jitter', action='store_true', default=False, help='Use jittered stimulus')
    parser.add_argument('-l', '--list', type=str, default=None,
                        help='text file of cell ids to compute for (useful for super-large stimulus)')
    parser.add_argument('-s', '--superbatch', type=int, default=-1,
                        help='Superbatch size (use if STA accumulator too big for GPU memory)')
    parser.add_argument('-o', '--manual_frame_offset', type=int, default=0,
                        help='Stimulus frame to start at. Example: if N, the first trigger in the .neurons file is associated with N * N_DISPLAY_FRAMES_PER_TTL frames after the start of the stimulus')
    parser.add_argument('-t', '--manual_trigger_offset', type=int, default=0, help='Skip this many triggers')


    args = parser.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(8)

    print("Loading spike times...")
    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_neurons=True)
    if args.list is None:
        all_cells = dataset.get_cell_ids()  # type: List[int]
    else:
        with open(args.list, 'r') as cell_id_list_file:
            all_cells = list(
                map(lambda x: int(x), cell_id_list_file.readline().strip('\n').split(',')))  # type: List[int]

    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    if args.manual_trigger_offset != 0:
        ttl_times = ttl_times[args.manual_trigger_offset:]

    n_samples_per_bin = SAMPLE_FREQ / args.frame_rate

    if args.superbatch == -1:
        print("Calculating STAs")
        framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path, args.jitter)
        if args.manual_trigger_offset != 0:
            framegen.advance_seed_n_frames(args.manual_trigger_offset * N_DISPLAY_FRAMES_PER_TTL)

        sta_dict = bin_frames_by_spike_times(spike_times_dict,
                                             ttl_times,
                                             framegen,
                                             N_DISPLAY_FRAMES_PER_TTL,
                                             n_samples_per_bin,
                                             args.n_frames,
                                             args.batch,
                                             device)
    else:
        print("Calculating STAs batched by cell")
        sta_dict = {}

        n_batches = int(np.ceil(len(all_cells) / args.superbatch))
        for batch_idx, i in enumerate(range(0, len(all_cells), args.superbatch)):
            print("Batch {0}/{1}".format(batch_idx+1, n_batches))
            framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path, args.jitter)
            if args.manual_trigger_offset != 0:
                framegen.advance_seed_n_frames(args.manual_trigger_offset * N_DISPLAY_FRAMES_PER_TTL)

            relevant_cell_ids = all_cells[i:min(len(all_cells), i+args.superbatch)]
            relevant_spike_times = {cell_id : spike_times_dict[cell_id] for cell_id in relevant_cell_ids}

            partial_sta_dict = bin_frames_by_spike_times(relevant_spike_times,
                                                         ttl_times,
                                                         framegen,
                                                         N_DISPLAY_FRAMES_PER_TTL,
                                                         n_samples_per_bin,
                                                         args.n_frames,
                                                         args.batch,
                                                         device)

            for cell_id, sta_mat in partial_sta_dict.items():
                sta_dict[cell_id] = sta_mat

    print("Writing STAs")
    with h5py.File(args.output, 'w') as h5_file:
        for cell_id, sta_mat in sta_dict.items():
            h5_file.create_dataset(str(cell_id), data=sta_mat)
