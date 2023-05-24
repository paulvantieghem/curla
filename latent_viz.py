import os
import json
import utils
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)
    args = parser.parse_args()
    return args

def plot_latent_tsne(file_path, file_name):
    
    # Load latent_value_dict dictionary from file
    with open(file_path, 'r') as f:
        latent_value_dict = json.load(f)

    # Get the latent representations and q values
    representations = []
    q_values = []
    for step in latent_value_dict.keys():
        representations.append(latent_value_dict[step]['latent_representation'])
        q_values.append(latent_value_dict[step]['q_value'])
    assert len(representations) == len(q_values)
    representations = np.array(representations)
    q_values = np.array(q_values)

    # Visualize latent representations
    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2)
    repr = np.concatenate(representations, axis=0)
    vec_2d = tsne.fit_transform(repr)
    ax.scatter(vec_2d[:,0], vec_2d[:,1], c=q_values, cmap='viridis', s=1)
    title = file_name.split('.')[0]
    ax.set_title(title)
    plt.savefig(os.path.join(f'{title}.png'))

def main():

    # Parse arguments
    args = parse_args()
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))

    # Set a fixed random seed for reproducibility across weather presets.
    args.seed = 0
    utils.set_seed_everywhere(args.seed)

    # Iterate over every .json file in the latent_data directory
    for file_name in os.listdir(os.path.join(args.experiment_dir_path, 'latent_data')):
        if file_name.endswith('.json'):
            file_path = os.path.join(args.experiment_dir_path, 'latent_data', file_name)
            plot_latent_tsne(file_path, file_name)

if __name__ == '__main__':
    main()