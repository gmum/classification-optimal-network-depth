import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='Saved state files.', type=Path, nargs='+')
    args = parser.parse_args()

    for file_path in args.files:
        print(f'=============== {file_path.name} ===============')
        with open(file_path, 'rb') as f:
            res_dict = torch.load(f, map_location='cpu')
        for k in ['current_x', 'final_acc', 'final_loss', 'final_train_acc', 'final_train_loss', 'chosen_layer',
                  'cutout_final_acc', 'cutout_final_loss', 'cutout_final_train_acc', 'cutout_final_train_loss']:
            if k in res_dict:
                v = res_dict[k]
                print(f'{k}: {v}')
        print(f'================================================\n')


if __name__ == '__main__':
    main()
