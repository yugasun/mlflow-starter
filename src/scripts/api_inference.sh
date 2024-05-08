#!/bin/sh

curl -d '{"dataframe_split": {
"columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
"data": [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]}}' \
  -H 'Content-Type: application/json' -X POST localhost:5002/invocations
