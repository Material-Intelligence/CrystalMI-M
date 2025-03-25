
<p align="center"><h1 align="center">CrystalMI-M</h1></p>

## Overview
CrystalMI-M is a large language model for Crystal Morphology Design.
Trained on a vast dataset comprising over 3 million data points of crystal morphologies spanning 7 crystal systems and 32 point groups, CrystalMI-M excels in predicting surface energy from surface area, or predicting surface area from surface energy.


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


## CrystalMI-M

### Setup Environment

```bash
cd crystalMI_M
conda env create -f environment.yml -n crystalMI_M
conda activate crystalMI_M
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

#### Data format example
Each data entry follows the JSON structure outlined below. The dataset consists of key-value pairs, where each entry contains an encoding field representing the crystal structure and an energy field representing the associated energy values.

##### Input Format
Each input entry is a JSON object with the following structure:
```json
{
    "encoding": "string",
    "energy": "string"
}
```
- `encoding`: A string describing the crystal structure, including symmetry, face vectors, and area values.

   - `Example`: "hexagonal(6)_face{[3,-4,0]:[0,2,-1]:[2,-2,1]}_area{3559:3220:3220}"

- `energy` : A string representing the energy values associated with the crystal structure.
   - `Example`: "energy{10000:9767:11804}"

##### Example Entries
Below are examples of actual data entries:
```json
   [
    {
        "encoding": "hexagonal(6)_face{[3,-4,0]:[0,2,-1]:[2,-2,1]}_area{3559:3220:3220}",
        "energy": "energy{10000:9767:11804}"
    },
    {
        "encoding": "tetragonal(4-mmm)_face{[1,-1,0]:[0,1,1]:[0,2,0]:[2,1,1]}_area{2658:2689:2344:2308}",
        "energy": "energy{10000:12957:10272:12473}"
    },
    {
        "encoding": "tetragonal(-42m)_face{[0,2,-1]:[1,3,0]}_area{8931:1068}",
        "energy": "energy{10000:12709}"
    }
]
```

- `encoding`: 
   - A string describing the crystal structure, including symmetry, face vectors, and area values.
     - `Symmetry group` : Specifies the crystal system and point group (e.g., hexagonal(6), tetragonal(4-mmm)).
     - `Face vectors` : Lists the Miller indices of the crystal faces (e.g., [3,-4,0]:[0,2,-1]:[2,-2,1]).
     - `Area values` : Provides the surface area values for each face (e.g., 3559:3220:3220).
   - `Example`: "hexagonal(6)_face{[3,-4,0]:[0,2,-1]:[2,-2,1]}_area{3559:3220:3220}"

- `energy` : A string representing the energy values associated with the crystal structure.
   - `Example`: "energy{10000:9767:11804}"


### Train
You can train the model from scratch using the preprocessed dataset:
```bash
python main.py -mode train
```
We recommend using the default settings to train the model. Additionally, you can set the parameters in the commands. The trained model will be saved in **preprocessed/checkpoints/final_model.pt** defaultly.

The following is the loss curve during model training:

<div align=center>
<img src=figures/loss.png width="500px">
</div>

### Generate
You can use your own trained model or our prepared model checkpoint [CrystalMI_M Model](http://43.138.168.107/crystal) to generate the predicted energy of the given crystals:
```bash
python main.py -mode generate
```
The predicted results will be saved in **preprocessed/predicted_energy_result.json**.

Note: If you use our prepared model checkpoint, please download it from [CrystalMI_M Model](http://43.138.168.107/crystal) and then place it in **preprocessed/checkpoints/**.


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
