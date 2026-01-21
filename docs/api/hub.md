# Hub

Utilities for downloading models from HuggingFace Hub.

Models are cached locally after first download (~500MB each).

## download_model

::: panodac.hub.download_model
    options:
      show_root_heading: true

## load_config

::: panodac.hub.load_config
    options:
      show_root_heading: true

## Available Models

Models are hosted at [huggingface.co/yuliangguo/depth-any-camera](https://huggingface.co/yuliangguo/depth-any-camera).

| Model               | Config File                  | Weights File                |
| ------------------- | ---------------------------- | --------------------------- |
| `outdoor-resnet101` | `dac_resnet101_outdoor.json` | `dac_resnet101_outdoor.pt`  |
| `outdoor-swinl`     | `dac_swinl_outdoor.json`     | `dac_swinl_outdoor.pt`      |
| `indoor-resnet101`  | `dac_resnet101_indoor.json`  | `dac_resnet101_indoor.pt`   |
| `indoor-swinl`      | `dac_swinl_indoor.json`      | `dac_swinl_indoor.pt`       |
