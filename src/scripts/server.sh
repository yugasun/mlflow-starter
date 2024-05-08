#!/bin/sh

mlflow models serve -m "models:/wine-quality/1" --port 5002