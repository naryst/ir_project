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

4. Install additional requirements from `environment.yml` - `conda env update -f environment.yml`

5. Install DCI data (text and annotations only) `python dataset/densely_captioned_images/dataset/scripts/download.py`

6. Install images from google drive. Install photos to `DCI/data/densely_captioned_images/photos`
Refer to readme of DCI in [readme](DCI/README.md) to the section of images loading.

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

### BUILD dense index 

1. Embed data to torch tensors `python get_jina_embeddings.py` in `dense_index_v2/`

2. Build index `python faiss_index.py` in `dense_index_v2/`

3. Run demo `streamlit run demo.py` in `dense_index_v2/`

### Run retrieval on custom images in `segmentation pipeline`

1. load images with `python retriever.py`
2. generate segmentation masks with `python mask_images.py`
3. generate descriptions `python generate_descriptions.py`
4. embed data `python embed_data.py`
5. build index `python build_faiss.py`
6. run demo `streamlit run demo.py` in `segmentation_pipeline/`