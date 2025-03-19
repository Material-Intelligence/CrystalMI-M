
<p align="center"><h1 align="center">CrystalMI-M</h1></p>

## Overview
CrystalMI-M is a large language model for Crystal Morphology Design.
Trained on a vast dataset comprising over 3 million data points of crystal morphologies spanning 7 crystal systems and 32 point groups, CrystalGPT-1 excels in predicting surface energy from surface area, or predicting surface area from surface energy.


<div align=center>
<img src=figures/framework.png width="500px">
</div>


## Database

We generate a comprehensive database, comprising over 3 million data points across 7 crystal systems and 32 point groups. 

- You can download the dataset from [Crystal Database](http://43.138.168.107/crystal).

- Alternatively, you can generate the dataset yourself:

```bash
cd crystal_database
conda env create -f environment.yml -n database
conda activate database
cd utils
python setup.py build
```
You need to move the files in the generated /build/lib.win-amd64-3.7 directory to the utils folder.

After this you can run the generated code.

```bash
cd ..  ### the crystal_database folder
python main.py
```

In addition, you can run the following three programs to draw images or create videos of crystals:

- `draw.py`: Based on the given crystal plane parameters, draw the corresponding crystal plane structure.
- `image_generation.py`: Based on the given crystal surface parameters CSV file, generate images and save them in the **image_temp** folder.
- `video_generation`: Read the images in the **image_temp** folder and generate a video of the crystal surface structure changing with the crystal surface parameters.


## CrystalGPT-1

### Setup Environment

```bash
cd crystal_gpt
conda env create -f environment.yml -n crystal_gpt
conda activate crystal_gpt
```

### Preprocess
Then you can preprocess the generated database:
```bash
python data_processing.py
```
The data generated in the database is summarized and converted into JSON format during preprocessing. The data will be stored in the **preprocessed** folder.

To prepare the datasets for the model training and evaluation:
```bash
python split_json.py
```
The summary data can be divided into three parts and stored in the **preprocessed/splits** directory: test.json, train.json, and val.json.

#### Example input and output

- Input: {"encoding": "cubic(m-3m)_face{[1, 0, 0]:[1, 1, 0]:[1, 1, 2]}_area{6363:211:3424}"}
- Output: {"energy": "energy{10000:13525:12372}"}

### Train
You can train the model from scratch using the preprocessed dataset:
```bash
python main.py -mode train
```
We recommend using the default settings to train the model. Additionally, you can set the parameters in the commands. The trained model will be saved in **preprocessed/checkpoints/final_model.pt** defaultly.

### Generate
You can use your own trained model or our prepared model checkpoint [CrystalGPT Model](http://43.138.168.107/crystal) to generate the predicted energy of the given crystals:
```bash
python main.py -mode generate
```
The predicted results will be saved in **preprocessed/predicted_energy_result.json**.

Note: If you use our prepared model checkpoint, please download it from [CrystalGPT Model](http://43.138.168.107/crystal) and then place it in **preprocessed/checkpoints/**.


### Some points to note

- `GPU memory`: Make sure your GPU has enough memory to handle the specified batch size and model size. If you don't have enough memory, try reducing batch_size or n_embd.
- `Data format`: Make sure all JSON files are in the same format, including necessary fields such as encoding and energy.
- `Dependency version`: Use the same version of the library specified in environment_train.yml or environment.yml to avoid compatibility issues.


<!-- ## Citation
Please cite our work as:
```
@misc{,
      title={}, 
      author={},
      year={2024},
}
``` -->

## License
<!-- All code is licensed under the MIT License - see the LICENSE.md file for details. -->
All code is licensed under the MIT License.
