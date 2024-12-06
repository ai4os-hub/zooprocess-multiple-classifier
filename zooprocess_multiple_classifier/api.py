# -*- coding: utf-8 -*-
"""
Functions to integrate the model with the DEEPaaS API.
To keep this file minimal, functions can be written in a separate file
and only called from here.

To start populating this file, take a look at the docs [1] and at
an exemplar module [2].
[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

from pathlib import Path
import logging
import os

from zooprocess_multiple_classifier import config
from zooprocess_multiple_classifier.utils import transform_valid,ZooScanEvalDataset
from zooprocess_multiple_classifier.misc import _catch_error

from webargs import fields

from zipfile import ZipFile
from PIL import Image
import torch


# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "name": config.API_METADATA.get("name"),
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


# initialise model (which is a global variable)
model = None
# define device  = cuda when available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def warm():
    """
    Load model upon application startup
    """
    global model, device

    # NB: get the model file from a github release
    model_path = os.path.join(BASE_DIR,
                              'models',
                              'best_model-2024-07-29_21-23-29.pt')
    if not os.path.exists(model_path):
        print("Model not found.")
    model = torch.load(model_path,    # nosec B614 (force bandit to ignore this error)
                       weights_only=False,
                       map_location=torch.device(device))
    model = model.to(device)


def get_predict_args():
    """
    Get the list of arguments for the predict function
    """
    arg_dict = {
        "images": fields.Field(
            metadata={
                'type': "file",
                'location': "form",
                'description': "A zip file containing the images to classify (all\
                images should be at the root of the zip file) or a single image file."
            },
            required = True
        ),
        "bottom_crop": fields.Int(
            metadata={
                'description': "Number of pixels to crop from the bottom of the\
                image (e.g. to remove the scale bar). [Default: 31px]"
            },
            required=False,
            load_default=31,
       )
    }

    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Predict the classification of an object
    
    Args:
        See get_predict_args() above.
    
    Returns:
        See schema below.
    """

    import tempfile
    data = kwargs['images']

    # get input files
    # either as a zip
    if data.content_type == 'application/zip':
        # extract
        tmp_input = tempfile.mkdtemp()
        with ZipFile(data.filename, 'r') as zip_file:
            zip_file.extractall(tmp_input)
        # keep only images
        filenames = sorted(os.listdir(tmp_input))
        filenames = [file for file in filenames if file.endswith(('jpg', 'png', 'jpeg'))]
        filepaths = [os.path.join(tmp_input, name) for name in filenames]
    # or as a single item
    else:
        filepaths = [data.filename]
        filenames = [data.original_filename]

    # prepare data structures for the CNN
    ds = ZooScanEvalDataset(paths=filepaths, names=filenames,
                            transform=transform_valid, bottom_crop=kwargs['bottom_crop'])
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    
    # set model in evaluation mode
    model.eval()
    torch.set_grad_enabled(False)

    # prepare storage of probabilities to be multiple (e.g. the score)
    scores = []
    # iterate over batches
    for imgs,names in dl:
        # process batch through model, on GPU
        imgs = imgs.to(device)
        logits = model(imgs)
        # convert to "probabilities" with a softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        # store the proba to be a multiple
        # NB: store as a list of tensors of type float64
        #     because the original float32 is not JSON serializable
        scores += probs[:,0].cpu().detach().to(torch.float64)
    
    # extract each element of the list to have a list of floats, not tensors
    scores = [s.item() for s in scores]

    return {"names": filenames, "scores": scores}

# Schema to validate the output of `predict()`
schema = {
    "names": fields.List(fields.Str(),
        required=True,
        metadata={
            'description': "A list containing the names of input images."
        }
    ),
    "scores": fields.List(fields.Float(),
        required=True,
        metadata={
            'description': "A list containing the probabilities for each image to be a multiple, in [0,1].\n\
            A natural threshold to classify an image as `multiple` is 0.5 but lowering this\
            threshold can increase the recall of multiples, at the expense of precision."
        }
    )
}

# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None
