
# torchvision for image datasets and base model
from torchvision import datasets, models

# torch for model modification
import torch
import torch.nn as nn

# torcheval for model evaluation
import torcheval.metrics.functional as metf

from zooprocess_multiple_classifier.utils import transform_train, transform_valid
import functools
import os
import datetime
# from tensorboardX import SummaryWriter
# import logging

def run_train(data_dir, out_dir, device, n_epochs=10, bottom_crop=31, batch_size=128, n_cores=10):
    # store training results in a timestamped directory
    train_dir = os.path.join(out_dir, datetime.datetime.now().strftime('train_%Y-%m-%d_%H-%M-%S'))
    os.makedirs(train_dir)
    # TODO check if we really need this if we have tensorboard
    
    # writer = SummaryWriter(logdir=train_dir, flush_secs=1)
    # misc.launch_tensorboard(logdir=train_dir)

    # # VR TODO 2-7-25 A remplacer par tensorboard
    # # prepare loggers
    # log = logging.getLogger()
    # log.setLevel(logging.INFO)
    # 
    # # define the output format for log messages
    # log_formatter = logging.Formatter('%(asctime)s.%(msecs)03d\t%(message)s',\
    #                                 datefmt='%Y-%m-%dT%H:%M:%S')
    # 
    # # log to console
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(log_formatter)
    # log.addHandler(console_handler)

    # # prepare logging to file (activated just before the training loop)
    # log_file = os.path.join(train_dir, 'log.tsv')
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setFormatter(log_formatter)


    ## Data loading ----
    # log.info('Create datasets')
    # create datasets
    # partially fill the transform_***() arguments
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
    # log.info(f"Nb of images in train={dataset_sizes['train']}, valid={dataset_sizes['valid']}")

    # convert into data loaders
    # log.info(f'Create dataloaders; batch_size={batch_size} n_cores={batch_size}')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=n_cores)
                      for x in ['train', 'valid']}

    # get class names
    class_names = image_datasets['train'].classes

    ## Prepare model ----

    # log.info(f'Create model')
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

    # start training
    # log.info("Start training")
    # log.addHandler(file_handler)
    # 
    # # initiate logging to file
    # log.info('epoch\tphase\tloss\taccuracy\tAUC\tmulti_recall\tmulti_precision')

    # # prepare saving checkpoints
    best_model_path = os.path.join(train_dir, 'best_model.pt')
    torch.save(model, best_model_path)
    best_loss = 10**6

    # list_data_json_for_df = []

    # num_epochs = num_epochs_input
    for epoch in range(n_epochs):
        # print(f'Epoch {epoch + 1}/{num_epochs}')

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
                # ipdb.set_trace()
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
            # log.info(f'{epoch+1}\t{phase}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t\
            # {epoch_auc:.4f}\t{epoch_multi_recall:.4f}\t{epoch_multi_precision:.4f}')

            # list_data_json_for_df.append({
            #     "epoch": epoch + 1,
            #     "phase": phase,
            #     "loss": epoch_loss,
            #     "accuracy": epoch_acc,
            #     "AUC": epoch_auc,
            #     "multi_recall": epoch_multi_recall,
            #     "multi_precision": epoch_multi_precision
            # })

            # writer.add_scalar("scalars/loss", epoch_loss, epoch)
            # writer.add_scalar("scalars/accuracy", epoch_acc, epoch)
            # writer.add_scalar("scalars/AUC", epoch_auc, epoch)
            # writer.add_scalar("scalars/epoch_multi_recall", epoch_multi_recall, epoch)
            # writer.add_scalar("scalars/epoch_multi_precision", epoch_multi_precision, epoch)

            # save the model if it is better
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model, best_model_path)


    # writer.close()
    # log.removeHandler(file_handler)

    # # load best model weights
    # log.info("Training complete")
    # log.info(f'Load model with best valid AUC: {best_auc:4f}')
    # model.load(torch.load(best_model_path, weights_only=False))
    # import pandas as pd
    # df_trace = pd.DataFrame(list_data_json_for_df)
    return None
