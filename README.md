# SPORT Agent

SPORT Agent is a comprehensive system for automated generation and verification of query-image pairs and trajectory samplings, designed to support the development of VLM agents. The system implements a robust pipeline to create and validate high-quality task-specific data. It features a modular architecture with components for data generation, verification, and trajectory sampling, supporting the entire workflow from query generation to DPO training and evaluation.

## Project Structure

The project consists of several main components:
- `data_generation/`: Contains the data generation pipeline
- `closed_loop_verifier/`: Verification and validation tools
- `tongagent/`: Core agent implementation
- `script/`: Utility scripts

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

The project uses environment variables for configuration. Make sure to set up your `.env` file and `configs/agent_config.yaml` with the necessary credentials and settings.


## Preparing Images and embeddings
The image captions and caption embeddings can be downloaded via the following link:
[Google Drive](https://drive.google.com/drive/folders/1Ek6qfmhcaTd7zTEQcBvELh6i7unVhTrk?usp=sharing).
Please download the images and embeddings and put them in 'data_generation/sharegpt4v'.

Please follow [ShareGPT4V](https://sharegpt4v.github.io/) to organize the image source in 'data_generation/sharegpt4v' as follows:
```none

├── ...
├── data
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   ├── share_textvqa
│   │   ├── images
│   ├── chatqa
│   │   ├── train
│   │   │   ├── png

```
## Data Generation Pipeline

The data generation process follows a sequential pipeline:

### 1. Query Generation
```bash
python data_generation/gta_pipeline/gta0_query_generation.py
```
Generates initial queries for the task.

### 2. Image Content Generation
```bash
python data_generation/gta_pipeline/gta1_query2image_content_parallel.py
```
Processes the queries to generate corresponding image content descriptions.

### 3. Image Retrieval
```bash
python data_generation/gta_pipeline/gta2_image_content2image_file.py
```
Retrieves actual images based on the generated content descriptions. The images are saved at `data/tongagent`.

### 4. Quality Filtering
```bash
python data_generation/gta_pipeline/gta3_q_f_filter_parallel.py
```
Performs quality checks and filtering on the query-image pairs.


## Trajectory Sampling

```bash
bash script/trajectory_sampling.sh
```
## Data Formatting

```bash
python data_generation/dpo_gta_traj/data_reformat/data_formating.py
```

## DPO Training 

Please refer to the llama-factory [repo](https://github.com/hiyouga/LLaMA-Factory) for DPO training.

## Evaluation

Download the gta dataset via [Huggingface](https://huggingface.co/datasets/Jize1/GTA) and put it in `data/gta_dataset`.

run the following command to evaluate the model:
```bash
bash script/gta_evaulation.sh
```
