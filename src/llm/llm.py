import mlflow
from mlflow.deployments import get_deploy_client
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.mlflow import Mlflow

# setup environment variables
from config.setup_env import setup_env

_, MLFLOW_DEPLOY_URI = setup_env()


def chat_by_api():
    client = get_deploy_client(MLFLOW_DEPLOY_URI)
    data = {
        "prompt": (
            "What would happen if an asteroid the size of "
            "a basketball encountered the Earth traveling at 0.5c? "
            "Please provide your answer in .rst format for the purposes of documentation."
        ),
        "temperature": 0.5,
        "max_tokens": 1000,
        "n": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    res = client.predict(endpoint="completions", inputs=data)

    print(res)


def chat_by_langchain():
    llm = Mlflow(target_uri=MLFLOW_DEPLOY_URI, endpoint="completions")
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["adjective"],
            template="Tell me a {adjective} joke",
        ),
    )
    result = llm_chain.run(adjective="funny")
    print(result)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(llm_chain, "model")

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny"}]))
