
from darkflow.net.build import TFNet

#%config InlineBackend.figure_format = 'svg'

options = {"model": "cfg/yolo.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 8,
           "epoch": 100,
           "train": True,
           "annotation": "new_data/annots/",
           "dataset": "new_data/images/"}
#           'gpu': 1.0,
tfnet = TFNet(options)

tfnet.train()

## si queremos hacerlo por consola
# python flow --model cfg/yolo.cfg --load bin/yolov2.weights --train --annotation new_data\annots --dataset new_data\images --epoch 1



