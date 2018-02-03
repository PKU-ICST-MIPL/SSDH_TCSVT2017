#!/usr/bin/env sh

TOOLS=./build/tools

#train flickr25k
$TOOLS/caffe train --solver=examples/semihash/flickr25k_semihash_solver.prototxt --weights=examples/semihash/Pre_trained/VGG_CNN_F.caffemodel --log_dir=examples/semihash/12bit_log/flickr25k --gpu=0
