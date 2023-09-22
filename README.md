# CRAFTs
CRAFT model exported to OpenVINO

## Installation

- Python >= 3.9
- Install the requirements: `pip install -r requirements.txt`

## Usage

- Run `python run.py` to run the demo on the images in the `images` folder.

```bash
usage: run.py [-h] 
              --model_path MODEL_PATH 
              --image_path IMAGE_PATH 
              [--output_dir OUTPUT_DIR] 
              [--text_threshold TEXT_THRESHOLD] 
              [--link_threshold LINK_THRESHOLD]
              [--low_text LOW_TEXT] 
              [--long_size LONG_SIZE] 
              [--poly] 
```

| Argument  | Description | Default |
| ------------- | ------------- | ------------- | 
| `--model_path`  | Path to the trained model | `./models/craft.xml` | |
| `--image_path`  | Path to the input image | `./image.jpg` | 
| `--output_dir`  | Path to the output directory | `./output` | 
| `--text_threshold`  | Text confidence threshold | `0.7` | 
| `--link_threshold`  | Link confidence threshold | `0.4` | 
| `--low_text`  | Text low-bound score | `0.4` | 
| `--long_size`  | Desired longest image size | `1280` |
| `--poly`  | Enable polygon type result | `True` |


- The output images will be saved in the `output` folder.

## Results
- input

    <img src="images/input.png" width="400" height="300" alt="1" align="center" />
- output

    <img src="images/int8/output.png" width="400" height="300" alt="1" align="center" />

## Maintainer

[Doku Tran](https://github.com/trancongman276)
