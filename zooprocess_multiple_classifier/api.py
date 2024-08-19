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
from zooprocess_multiple_classifier.utils import transform_valid
from zooprocess_multiple_classifier.misc import _catch_error

from webargs import fields

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
    model = torch.load(model_path,
                       weights_only=False,
                       map_location=torch.device(device))
    model = model.to(device)


def get_predict_args():
    """
    Get the list of arguments for the predict function
    """
    arg_dict = {
        "image": fields.Field(
            metadata={
                'required': True,
                'type': "file",
                'location': "form",
                'description': "An image containing object(s) to classify"
            }
        ),
    }

    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Predict the classification of an object
    """

    # read image
    filename = kwargs['image'].filename
    img = Image.open(filename)

    # prepare it for the network
    img = img.convert('RGB')
    img = transform_valid(img)
    img = img.to(device)
    img = img[None, :, :, :]  # add empty dimension as for a batch

    # get predicted classification
    model.eval()
    with torch.no_grad():
        score = model(img)
    # NB: at this point, the softmax as not been applied yet
    score = torch.nn.functional.softmax(score, dim=1)
    # print('score =', score)

    # NB: extract the float value from the tensor, otherwise validation fails
    return {"score": score[0][0].item()}


# Schema to validate the `predict()` output
schema = {
    "score": fields.Float()
}

# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None
