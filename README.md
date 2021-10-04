Impementation of [DCASE 2016 ACOUSTIC SCENE CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS](http://dcase.community/documents/workshop2016/proceedings/Valenti-DCASE2016workshop.pdf)

# Usage

Train with an 80/20 split on the devel set:
 `python main.py TUT-acoustic-scenes-2016/development/`

Train on full devel set:
`python main.py TUT-acoustic-scenes-2016/development/ --full_train`

Eval on eval set:
`python main.py TUT-acoustic-scenes-2016/evaluation/ --eval --model_checkpoint model.pt`



# Assumptions:
* Ignored K-fold cross validation, instead did a standard 80/20 training/val split.
* Ignored full/non-full training as in the paper. Just did standard training from scratch
* For "destruction of the time axis" max pooling, I used adaptive max pool as it's easier to get dimensions correct. This means the "non-overlapping frequency bands" are not enforced (could be overlapping)
* For sequence splitting, I combined the sequences into the batch dimension on input into the model. The model output was then seperated back out and averaged acrossed the splits.  
