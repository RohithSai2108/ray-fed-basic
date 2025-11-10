# ray-fed-basic-implementation


## Getting Started

### Clone the Repository

```bash
git clone https://github.com/RohithSai2108/ray-fed-basic.git
cd ray-fed-basic
```

### Local Development Setup 

If you want to develop or run the code locally without Docker, follow these steps:

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
.\venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Building the Docker Image

Build the Docker image using the following command:

```bash
docker build -t team-python-env:local .
```

### Running the Application

Run the federated learning experiment using Docker with the following command:

```bash
docker run --rm --shm-size=4.0gb team-python-env:local
```

Note: The `--shm-size=1.07gb` parameter is required to allocate sufficient shared memory for the federated learning process.

## Project Structure

- `model.py`: Contains the neural network architecture
- `datasets.py`: Dataset handling and preprocessing
- `client.py`: Federated learning client implementation
- `run_experiment.py`: Main script to run the federated learning experiment
- `test_ray.py`: Ray framework testing script

## Requirements

- Docker
- At least 2GB of available RAM
- Docker daemon with access to sufficient disk space
