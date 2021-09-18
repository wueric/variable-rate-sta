from lib.torch_sta import bin_spike_times_by_frames
from lib.save_data import generate_save_dict
from lib.trigger_interpolation import interpolate_trigger_times

import numpy as np
import torch

import visionloader as vl
import visionwriter as vw
from whitenoise import RandomNoiseFrameGenerator
import electrode_map as el_map

import pickle

import argparse
from typing import Dict, Tuple, Union, Sequence
import os

CELL_BATCH_SIZE = 64
N_DISPLAY_FRAMES_PER_TTL = 100
SAMPLE_FREQ = 20000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute STAs at monitor frequency')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('xml_path', type=str, help='path to stimulus XML file')
    parser.add_argument('output', type=str, help='path to save location')
    parser.add_argument('-n', '--n_frames', type=int, help='number of frames', default=51)
    parser.add_argument('-b', '--batch', type=int, help='number of cells batch size', default=CELL_BATCH_SIZE)
    parser.add_argument('-j', '--jitter', action='store_true', default=False, help='Use jittered stimulus')
    parser.add_argument('-o', '--manual_frame_offset', type=int, default=0,
                        help='Frame offset. Example: if N, the first trigger in the .neurons file is associated with N * N_DISPLAY_FRAMES_PER_TTL frames after the start of the stimulus')
    parser.add_argument('-t', '--manual_trigger_offset', type=int, default=0, help='Skip this many triggers')
    parser.add_argument('-d', '--trigger_interp_deviation', type=float, default=0.1,
                        help='maximum allowable deviation from expected trigger interval. Used for trigger interpolation')


    args = parser.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(8)

    print("Loading spike times...")
    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_neurons=True)
    all_cells = dataset.get_cell_ids()
    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    if args.manual_trigger_offset != 0:
        ttl_times = ttl_times[args.manual_trigger_offset:]

    ttl_times = interpolate_trigger_times(ttl_times, deviation_interval=args.trigger_interp_deviation)

    avg_ttl_time = np.median(ttl_times[1:] - ttl_times[:-1]) # type: float
    monitor_freq = 1.0 / (N_DISPLAY_FRAMES_PER_TTL * (avg_ttl_time / SAMPLE_FREQ))

    framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path, args.jitter)
    if args.manual_frame_offset != 0:
        framegen.advance_seed_n_frames(args.manual_frame_offset * N_DISPLAY_FRAMES_PER_TTL)

    print("Calculating STAs")
    sta_dict = bin_spike_times_by_frames(spike_times_dict,
                                         ttl_times,
                                         framegen,
                                         N_DISPLAY_FRAMES_PER_TTL,
                                         args.n_frames,
                                         args.batch,
                                         device)

    with open(args.output, 'wb') as pfile:
        save_dict = generate_save_dict(sta_dict, monitor_freq)
        pickle.dump(save_dict, pfile)
