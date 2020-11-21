from lib.torch_sta import bin_frames_by_spike_times
from lib.save_data import generate_save_dict

import torch

import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

import pickle
import argparse

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

    args = parser.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(8)

    print("Loading spike times...")
    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_neurons=True)
    all_cells = dataset.get_cell_ids()
    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path)

    n_samples_per_bin = SAMPLE_FREQ / args.frame_rate

    print("Calculating STAs")
    sta_dict = bin_frames_by_spike_times(spike_times_dict,
                                         ttl_times,
                                         framegen,
                                         N_DISPLAY_FRAMES_PER_TTL,
                                         n_samples_per_bin,
                                         args.n_frames,
                                         args.batch,
                                         device)

    print("Writing STAs")
    with open(args.output, 'wb') as pfile:
        save_dict = generate_save_dict(sta_dict, args.frame_rate)
        pickle.dump(save_dict, pfile)
