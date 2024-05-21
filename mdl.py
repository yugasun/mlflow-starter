import argparse
from modelscope import snapshot_download

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Specify the model name", required=True)
args = parser.parse_args()

# get model name from command line argument
model_name = args.model


# if model name is not provided, throw an error
if model_name is None:
    raise ValueError("Model name is required")

# model = Model.from_pretrained(
#     model_name,
# )
snapshot_download(model_name)