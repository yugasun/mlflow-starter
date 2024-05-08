## mlflow starter

Mlflow starter is a project that demonstrates how to use mlflow to track and manage machine learning experiments.

### Pre-requisites

- Docker
- Python 3.10+

### Setup environment

```bash
docker-compose up -d
```

### Installation

```bash
pip install -r requirements.txt
```

### Run Tracking Server

```bash
./start_server.sh
```

### Run Training

```bash
python src/quickstart.py
```