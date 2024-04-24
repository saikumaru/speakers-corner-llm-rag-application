# How to deploy a flask RAG application on Azure, specifically Azure Container App service- A serverless enginer.

## Embeddings storage
QdrantDB(Container)

## Chat history storage
Redis(Container)

## API server
Python Flask

## LLM
OpenAI GPT 3.4




## In the python flask project we have-
1. Added Qdrant as a vector db store
2. Methods that have been enhanced with inclusion of Qdrant APIs(Initialize, search_results, generate methods)
3. Redis implementation to store the chat history, which will be used as context
4. Basic chat UI in HTML, CSS to communicate with the bot



# Running this code on your local machine

## Pre requistes to be installed
1. Docker 
2. Azure CLI
3. Python 3.9
4. Create a `.env` file under `src` folder with below data

 >   OPENAI_API_KEY="<your open ai key only used for local run>"
 >   QDRANT_HOST="localhost"
 >   QDRANT_PORT="6333"
 >   REDIS_HOST="localhost"
 >   REDIS_PORT="6379"
 >   REDIS_PASSWORD="None"


# On CMD line
1. Create and activate your python environment for this project

2. Install the required packages
`pip install -r requirements.txt`

3. Run Qdrant docker container
`docker run -d -p 6333:6333 qdrant/qdrant`

4. Run Redis docker container
`docker run --name redis -d -p 6379:6379 redis`

5. Run flask server and check locally if all works(from src folder)
`python flask_rag_llm_openai_hf.py`


-------------------------------
# Azure deployment
-------------------------------

6. Prerequisites
- Provison a suitable Azure container app service environment, based on scale, redundancy and region of choice
- Providon a azure container registry. You could also use a docker public image, unless you want it to be available for everyone else to use.


7. Login to Azure on your command line and push your image to registry

`az login`
`az account set --subscription <your specific subscription>`
`az acr login -n <registry name created on Azure>`


8. cd to correct folder having dockerfile, and build the image
`docker build . -t speakerscornerregistry.azurecr.io/openai`
takes about 100s-200s

Then push the image to registry with tag
`docker push <registry name created on Azure>.azurecr.io/openai:latest`

9. On container page pick the specific registry, image and tag of the image just pushed.
Key in your environment variables manually while creating the conatiner app, i.e. your openAi key and its value.

11. On bindings create and select your two sidecars qdrant and redis

12. In the network tab while creating the ACA check `"Enable ingress"`, and `"Traffic from anywhere"` to allow receiving traffic from internet when you open the container app link.
- Update the target port to `5000` as this is the port where the flask api is being served.

13. Once you hit `Create`. You can check status of you deployment under "Revisons and replicas" menu.

14. Once the app is ready, you can access the bot page. The link would be similar to below
https://<container app name>.<random word>.<region>.azurecontainerapps.io/openai



# Substitutes for Serverless GPU for inferencing purpose
Replicate
Runpod
Huggingface


This code release is being done as part of speakers corner session that was conducted on 16th April 2024
https://www.landing.ciklum.com/sc-architecting-scalable-ai

- [Ciklum](https://www.ciklum.com/)