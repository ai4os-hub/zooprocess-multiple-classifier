"""
This is the training function. It is meant to work with data in the data folder
"""

# torchvision for image datasets and base model
from torchvision import datasets, models

# torch for model modification
import torch
import torch.nn as nn

# torcheval for model evaluation
import torcheval.metrics.functional as metf

from zooprocess_multiple_classifier.utils import transform_train, transform_valid
from zooprocess_multiple_classifier import misc
import functools
import os
import datetime
from tensorboardX import SummaryWriter

def run_train(data_dir, out_dir, device, n_epochs=10, bottom_crop=31, batch_size=128, n_cores=10):
    # store training results in a timestamped directory
    out_dir = os.path.join(out_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(out_dir)
    writer = SummaryWriter(logdir=out_dir, flush_secs=1)
    # misc.launch_tensorboard(logdir=out_dir)
    # TODO this is currently blocking the rest of the execution of the function

    ## Load data ----
    print('Create datasets')
    
    # create datasets
    # partially fill the transform_***() arguments, to take bottom_crop into account
    transform_train_with_crop = functools.partial(transform_train, bottom_crop=bottom_crop)
    transform_valid_with_crop = functools.partial(transform_valid, bottom_crop=bottom_crop)
    data_transforms = {
        'train': transform_train_with_crop,
        'valid': transform_valid_with_crop
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                         for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"  Nb of images in train={dataset_sizes['train']}, valid={dataset_sizes['valid']}")

    # convert into data loaders
    print(f'Create dataloaders; batch_size={batch_size} n_cores={batch_size}')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=n_cores)
                      for x in ['train', 'valid']}

    # get class names
    class_names = image_datasets['train'].classes


    ## Prepare model ----

    # print(f'Create model')
    print(f'Create model')
    # start with a MobileNet
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')

    # replace the classifier by our own
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=300, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=300, out_features=len(class_names), bias=True)
    )

    # transfer the model to the GPU
    model = model.to(device)

    ## Train model ----

    # define the loss, with more weight on the multiple objects to increase recall
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([4,1]).to(device))

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # decay LR by a factor at every epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    print("Start training")

    # prepare saving checkpoints
    best_model_path = os.path.join(out_dir, 'best_model.pt')
    torch.save(model, best_model_path)
    best_loss = 10**6

    # add header for logged quantities
    print('epoch\tphase\tloss\tacc\tm_rec\tm_prec')
    #                          accuracy recall_of_multiples precision_of_multiples

    for epoch in range(n_epochs):
        for phase in ['train', 'valid']:
            # set the model in the current mode
            if phase == 'train':
                model.train()
            else:
                model.eval()

            run_loss = 0.0
            run_cm = torch.zeros((2,2), dtype=torch.int64).to(device)

            # iterate over batches
            for inputs, labels in dataloaders[phase]:
                # copy batch to GPU (or CPU as fallback)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # multi is 0, single is 1
                # see image_datasets['train'].class_to_idx
                # keep track of statistics
                run_loss += loss.item() * inputs.size(0)
                run_cm += metf.binary_confusion_matrix(input=preds, target=labels.data)

            ## End of epoch
            # update the scheduler
            if phase == 'train':
                scheduler.step()

            # compute statistics
            tot_n = torch.sum(run_cm)
            epoch_loss = run_loss / tot_n
            epoch_acc = torch.sum(torch.diag(run_cm))/tot_n
            epoch_multi_recall = run_cm[0,0] / torch.sum(run_cm[0,:])
            epoch_multi_precision = run_cm[0,0] / torch.sum(run_cm[:,0])
            print(f'{epoch+1}/{n_epochs}\t{phase}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_multi_recall:.4f}\t{epoch_multi_precision:.4f}')

            # log to tensorboard
            writer.add_scalar("scalars/loss", epoch_loss, epoch)
            writer.add_scalar("scalars/accuracy", epoch_acc, epoch)
            writer.add_scalar("scalars/multi_recall", epoch_multi_recall, epoch)
            writer.add_scalar("scalars/multi_precision", epoch_multi_precision, epoch)

            # save the model if it is better
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model, best_model_path)

    writer.close()

    return None
