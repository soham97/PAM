import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PAM import PAM
from dataset import ExampleDatasetFolder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PAM")
    parser.add_argument('--folder', type=str, help='Folder path to evaluate')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples per batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    args = parser.parse_args()

    # initialize PAM
    pam = PAM(use_cuda=torch.cuda.is_available())

    # Create Dataset and Dataloader
    dataset = ExampleDatasetFolder(
        src=args.folder,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False,
                            num_workers = args.num_workers,
                            pin_memory = False, drop_last=False, collate_fn=dataset.collate)

    # Evaluate and print PAM score
    collect_pam, collect_pam_segment = [], []
    for files, audios, sample_index in tqdm(dataloader):
        pam_score, pam_segment_score = pam.evaluate(audios, sample_index)
        collect_pam += pam_score
        collect_pam_segment += pam_segment_score

    print(f"PAM Score: {sum(collect_pam)/len(collect_pam)}")