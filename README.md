# Visual Localization with 2D Semantic Maps

TLDR: We train a deep neural network to predict the SE(2) pose of an image with respect to a 2D semantic map obtained from OpenStreetMap.

- Internship project by [Paul-Edouard Sarlin](https://psarlin.com/).
- [Slide deck of the final presentation.](https://docs.google.com/presentation/d/1lSHjDHlm9YtPplzWNtmN_3ugspKwd6GgjmdXWZfO7tI/edit?usp=sharing)
- [Slide deck of the project.](https://docs.google.com/presentation/d/1kVAUeA0K9p2JJWV6wHE2X2OkO_-gZqKhJIz5NwzKUG8/edit?usp=sharing)

## Bento
Build the kernel:
```
buck build @mode/opt -c python.package_style=inplace --show-output :bento_kernel_maploc
```
[Collection of notebooks](https://www.internalfb.com/intern/anp/dashboard/?collection_path[0]=670305553958094), for example:
- [Visualize the predictions on the Mapillary or Metropolis datasets.](https://fburl.com/anp/1ah9gqxo)
- [Visualize single-image predictions on Aria data.](https://fburl.com/anp/p0r9b51x)
- [Run the sequence inference on Aria sequences.](https://fburl.com/anp/5zlr84lp)

## Training
To train the full BEV model on the Mapillary dataset:
```
buck run @mode/opt :train -- -m  --config-name bev_depth_final experiment.name=name_of_experiment
```
To run on fblearner, simply append `hydra.launcher.mode=flow`:
```
buck run @mode/opt :train -- -m  --config-name bev_depth_final hydra.launcher.mode=flow experiment.name=name_of_experiment
```

To train the image retrieval baseline on the Mapillary dataset:
```
buck run @mode/opt :train -- -m  --config-name basic_v2 hydra.launcher.mode=flow experiment.name=name_of_experiment
```

To finetune on the KITTI dataset:
```
buck run @mode/opt :train -- -m  --config-name basic_v2 hydra.launcher.mode=flow experiment.name=name_of_experiment \
    training.finetune_from_checkpoint='"manifold://psarlin/tree/maploc/experiments/name_of_first_experiment/checkpoint-epoch=0N.ckpt"' \
    data=kitti
```

- [Link to the spreadsheet with all experimetns.](https://docs.google.com/spreadsheets/d/1ZYyO74uwcdu8zybk52xSPZWUl8jMgaG5qLa3Fi_mx4U/edit#gid=0)
- [Link to Tensorboard to compare the experiments.](https://www.internalfb.com/intern/tensorboard/?dir=manifold://psarlin/tree/maploc/experiments)

Common options with their defaults:
- change the experiment directory: `experiment.root=manifold://psarlin/tree/maploc/experiments`
- change the data directory: `data.dump_dir=manifold://psarlin/tree/maploc/data/mapillary_v4`
- train with fewer GPUS: `experiment.gpus=3`
- change the batch size: `data.loading.train.batch_size=9`

## Preparing Mapillary data
1. Add a new location to [maploc/mapillary/run.py](./mapillary/run.py)
2. Download the OpenStreetMap files using [this notebook](https://fburl.com/anp/4wgx64bt).
3. Run the processing:
```
buck run @mode/opt -c python.package_style=inplace :mapillary_ingestion -- --locations name_of_location
```

## Preparing Aria data
[Run this notebook.](https://fburl.com/anp/o247lta9)
