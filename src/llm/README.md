### General Configuration Parameters

The `config.yaml` file contains the following configuration parameters:

- endpoints: This is a list of endpoint configurations. Each endpoint represents a unique endpoint that maps to a particular language model service.

Each endpoint has the following configuration parameters:

- name: This is the name of the endpoint. It needs to be a unique name without spaces or any non-alphanumeric characters other than hyphen and underscore.

- endpoint_type: This specifies the type of service offered by this endpoint. This determines the interface for inputs to an endpoint and the returned outputs. Current supported endpoint types are:

  “llm/v1/completions”  
  “llm/v1/chat”  
  “llm/v1/embeddings”

- model: This defines the provider-specific details of the language model. It contains the following fields:

- provider: This indicates the provider of the AI model. It accepts the following values:

  * “openai”
  * “mosaicml”
  * “anthropic”
  * “cohere”
  * “palm”
  * “azure” / “azuread”
  * “mlflow-model-serving”
  * “huggingface-text-generation-inference”
  * “ai21labs”
  * “bedrock”
  * “mistral”

- name: This is an optional field to specify the name of the model.

- config: This contains provider-specific configuration details.


More details on the configuration parameters can be found in the [configuration-parameters](https://mlflow.org/docs/latest/llms/deployments/index.html#configuring-the-deployments-server) section.