from lib.torch_sta import bin_spike_times_by_frames
from lib.save_data import generate_save_dict

import numpy as np
import torch

import visionloader as vl
import visionwriter as vw
from whitenoise import RandomNoiseFrameGenerator

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
    parser.add_argument('-v', '--visionwriter', type=bool, default=False, help='save in Vision .sta format')

    args = parser.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(8)

    print("Loading spike times...")
    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_neurons=True)
    all_cells = dataset.get_cell_ids()
    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    avg_ttl_time = np.median(ttl_times[1:] - ttl_times[:-1])
    monitor_freq = 1.0 / (N_DISPLAY_FRAMES_PER_TTL * (avg_ttl_time / SAMPLE_FREQ))

    framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path)

    print("Calculating STAs")
    sta_dict = bin_spike_times_by_frames(spike_times_dict,
                                         ttl_times,
                                         framegen,
                                         N_DISPLAY_FRAMES_PER_TTL,
                                         args.n_frames,
                                         args.batch,
                                         device)

    if args.visionwriter:

        sta_container_by_cell_id = {}  # type: Dict[int, vl.STAContainer]
        for cell_id, sta_matrix in sta_dict.items():
            depth, width, height, n_channels = sta_matrix.shape
            no_error = np.zeros_like(sta_matrix[..., 0])
            sta_container = vl.STAContainer(framegen.stixel_width,
                                            monitor_freq,
                                            0,
                                            sta_matrix[..., 0],
                                            no_error,
                                            sta_matrix[..., 1],
                                            no_error,
                                            sta_matrix[..., 2],
                                            no_error)
            sta_container_by_cell_id[cell_id] = sta_container

        with vw.STAWriter(args.output,
                          args.ds_name,
                          framegen.field_width,
                          framegen.field_height,
                          args.n_frames,
                          monitor_freq,
                          0,
                          framegen.stixel_width) as staw:

            staw.write_sta_by_cell_id(sta_container_by_cell_id)

    else:
        with open(args.output, 'wb') as pfile:
            save_dict = generate_save_dict(sta_dict, monitor_freq)
            pickle.dump(save_dict, pfile)
