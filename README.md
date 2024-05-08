## mlflow starter

Mlflow starter is a project that demonstrates how to use mlflow to track and manage machine learning experiments.

### Pre-requisites

- Docker
- Python 3.10+

### Configuration

Copy the `.env.example` file to `.env` and update the values as needed.

```bash
cp .env.example .env
```

> If you want to use openai api, you need to set the `OPENAI_API_KEY` in the `.env` file.

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

### Server LLM

```bash
./src/llm/run.sh
```