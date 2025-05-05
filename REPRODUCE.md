# Steps to reproduce solution

### Prepare DCI & deps

1. Create conda env
```
conda create -n ir_project python=3.10
conda activate ir_project
```

2. Install deps from DCI 
```
cd DCI/dataset
pip install -e .
```
3. Setup data installation path. Change value `data_dir` in `DCI/dataset/config.yaml`

4. Install DCI data (text and annotations only)
`python dataset/densely_captioned_images/dataset/scripts/download.py `

5. Install images from google drive. Install photos to `DCI/data/densely_captioned_images/photos`
```
pip install gdown
gdown "https://drive.google.com/uc?id=1fdWEW7ejNnGHg1nipunSz0GkVbicMyMZ"
```
Or just load `.tar` file from [this link](https://drive.google.com/file/d/1fdWEW7ejNnGHg1nipunSz0GkVbicMyMZ/view?usp=sharing) and place to the right dir.


### BUILD k-gram index
```
cd kgram_index
python build_index.py --k=<YOUR_VALUE>
```
`.pkl` file with index will be saved into `kgram_index/data` 

### Run streamlit demo
```
cd demo
pip install -r requirements.txt
streamlit run 
```

123 test
