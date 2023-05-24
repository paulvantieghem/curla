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
    with open(os.path.join(file_path, file_name), 'r') as f:
        latent_value_dict = json.load(f)

    # Get the latent representations and q values in the right format
    representations = []
    q_values = []
    for step in latent_value_dict.keys():
        representations.append(np.array(latent_value_dict[step]['latent_representation']))
        q_values.append(latent_value_dict[step]['q_value'])
    assert len(representations) == len(q_values)
    representations = np.array(representations)
    q_values = np.array(q_values)


    # Perform t-SNE on the latent representations
    tsne = TSNE(n_components=2)
    vec_2d = tsne.fit_transform(representations)


    # Visualize latent representations in function of Q-values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    scatter = ax1.scatter(vec_2d[:,0], vec_2d[:,1], c=q_values, cmap='viridis', s=30)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Q Value')

    # Visualize latent representations in function of steps
    scatter = ax2.scatter(vec_2d[:,0], vec_2d[:,1], c=np.arange(q_values.shape[0]), cmap='viridis', s=30)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Step')

    # Save the figure
    title = file_name.split('.')[0]
    fig.suptitle(f'Weather preset: {title}')
    plt.savefig(os.path.join(file_path, f'{title}.png'))

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
            file_path = os.path.join(args.experiment_dir_path, 'latent_data')
            plot_latent_tsne(file_path, file_name)

if __name__ == '__main__':
    main()