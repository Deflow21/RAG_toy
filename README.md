# RAG_toy
alpha beta RAG for the graduation paper

![](pablo-james-pablo.gif)

### Project Structure
- `download.py` - Downloads a zip file containing models from the ABC dataset; currently limited to one link and 1000 models (each link actually contains around 10k models) due to extremely slow processing.
  
- `BD_new.py` – Creates a MongoDB database and loads the data previously processed in `download.py`.
  
- `generate_json_rag.py` – Uses RAG to generate a JSON description of the technological process for processing a part, based on a drawing from the ABC dataset and data from the database.
  
- `genrate_without_RAG.py` – Generates output without using RAG at all.
  
- `model.py` – A test script for generation, simply to see how the model works.

---

## Installation and Setup

### 1. Installing the Required Libraries
```sh
pip install -r requirements.txt
```

### 2. Setting Up the MongoDB Database
Visit the official MongoDB website to download, install, and run MongoDB. You will immediately be prompted to create a localhost—do so. No further action is required. Once you have loaded all the information into the database, you can verify in MongoDB to ensure everything is 100% in order.

<p><span style="color: red; font-weight: bold;">Configure the database connection parameters in the file <code>mongodb.env</code></span></p>

---

## Usage (relevant for working in VSCode)

### 1. Downloading Files from the Selected Link
Run `download.py` to download all the necessary files for loading on local machine

### 2. Generating JSON Using RAG
Run `BD_new.py` to load the files into the database

### 2. Generating JSON Using RAG
Run `generate_json_rag.py`, providing an image of the model's drawing from the ABC dataset

### 3. Testing the Model Without RAG
You can test the model separately using `generate_without_RAG.py`
