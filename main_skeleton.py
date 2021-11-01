import torch
import torchaudio
from torch.utils.data import DataLoader

from dataset import DCASE

torchaudio.set_audio_backend("sox_io")

def main():
#EXAMPLE CODE DEMONSTRATING HOW TO LOAD DATA AND PASS MULTIPLE CLIPS THROUGH YOUR MODEL. THIS CODE DOES NOT RUN AS IS
#YOU WILL NEED TO IMPLEMENT A FULL MODEL AND TRAINING LOOP IN ORDER TO RE-PRODUCE THE RESULTS


    #CREATING DATASET
    dataset = DCASE(path_to_data, clip_len)

    #DATALOADER FOR FULL TRAINING
    if full_train
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #DATALOADERS FOR NON-FULL TRAINING
    elif non_full_train:
        length = len(dataset)
        train_len = int(round(length * 0.8))
        val_len = length - train_len
        train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)


    #INSIDE TRAINING LOOP:
        #Combine clips into batch dimension
        spectrograms = spectrograms.contiguous().view(-1, 1, np.shape(spectrograms)[2], np.shape(spectrograms)[3])

        #pass through model
        output = model(spectrograms)

        #seperate clips out from batch and average
        num_clips = dataset.get_num_clips()
        output = output.reshape(-1, num_clips, num_classes)

if __name__ == '__main__':
    main()
