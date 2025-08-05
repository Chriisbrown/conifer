"""Example of BDT creation with Yggdrasil Decision Forests (https://ydf.readthedocs.io)."""

import datetime
import logging
import sys
import scipy

import numpy as np
import ydf
from sklearn.datasets import make_hastie_10_2
import conifer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create dataset
X, y = make_hastie_10_2(random_state=0)
y = y == 1  # Converts Hastie's labels from {-1, 1} to {False, True}.

# Train a Gradient Boosted Decision Trees model using YDF.
# Sparse oblique weights: BINARY, CONTINUOUS, POWER_OF_TWO, INTEGER -> TO TEST
# hls_cfg['weight_precision'] = 'ap_fixed<18,8>' -> TO TEST

mode = "POSITIVE_POW2"

input_widtht = 10
input_integer = 1
n_feaures = 10

args = {
    "num_trees": 100,
    "max_depth": 3,
    "apply_link_function": False,
    "split_axis": "SPARSE_OBLIQUE",
    "label": "y",
}

if mode == "BINARY":
    weight_width = 1
    weight_integer = 1
    weight_type = f"ap_fixed<{weight_width},{weight_integer}>"
elif mode == "CONTINUOUS":
    weight_width = 10
    weight_integer = 1
    weight_type = f"ap_fixed<{weight_width},{weight_integer}>"
elif mode == "POWER_OF_TWO":
    weight_width = 10
    weight_integer = 10
    args["sparse_oblique_weights_power_of_two_max_exponent"] = weight_integer - 1
    args["sparse_oblique_weights_power_of_two_min_exponent"] = 0
    weight_type = f"ap_fixed<{weight_width},{weight_integer}>"
elif mode == "INTEGER":
    weight_width = 5
    weight_integer = 5
    args["sparse_oblique_weights_integer_maximum"] = pow(2, weight_integer - 1)
    args["sparse_oblique_weights_integer_minimum"] = -pow(2, weight_integer - 1)
    weight_type = f"ap_fixed<{weight_width},{weight_integer}>"
elif mode == "POSITIVE_POW2":
    mode = "INTEGER"
    weight_width = 5
    weight_integer = 5
    args["sparse_oblique_weights_integer_maximum"] = pow(2, weight_integer)
    args["sparse_oblique_weights_integer_minimum"] = 0
    weight_type = f"ap_ufixed<{weight_width},{weight_integer}>"
else:
    exit(f"Unknown mode {mode}")

args["sparse_oblique_weights"] = mode

model = ydf.GradientBoostedTreesLearner(**args).train({"x": X, "y": y})

# Create a Conifer configuration.
stamp = int(datetime.datetime.now().timestamp())
cpp_cfg = conifer.backends.cpp.auto_config()
# cpp_cfg["Precision"] = "float"  # Optional float precision.
cpp_cfg["OutputDir"] = "prj_cpp_{}".format(stamp)

# Convert the YDF model to a C++ Conifer model.
cpp_model = conifer.converters.convert_from_ydf(model, cpp_cfg)
cpp_model.compile()

# Create a conifer config
hls_cfg = conifer.backends.xilinxhls.auto_config()
# hls_cfg["Precision"] = "float"  # Optional float precision.
hls_cfg["OutputDir"] = "prj_hls_{}".format(stamp)

hls_cfg["input_precision"] = f"ap_fixed<{input_widtht},{input_integer}>"
hls_cfg["weight_precision"] = weight_type

# Convert the YDF model to a HLS Conifer model.
hls_model = conifer.converters.convert_from_ydf(model, hls_cfg)
hls_model.compile()

print(f"n_features: {hls_model.n_features}")

# Compare the predictions of YDF, C++ Conifer, and HLS Conifer model.
y_cpp = scipy.special.expit(cpp_model.decision_function(X))[:, 0]
y_hls = scipy.special.expit(hls_model.decision_function(X))
y_ydf = model.predict({"x": X})

if np.array_equal(y_hls, y_cpp):
    print(f"HLS and CPP predictions agree 100% ({len(y_cpp)}/{len(y_cpp)})")
else:
    abs_diff = np.abs(y_hls - y_cpp)
    rel_diff = abs_diff / np.abs(y_hls)
    print(
        f"HLS and CPP predictions disagree. Biggest absolute difference: {abs_diff.max():.4f}, biggest relative difference: {rel_diff.max():.4f}"
    )
