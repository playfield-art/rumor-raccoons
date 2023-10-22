# Welcome to RUMOR BY [playField](https://playfield.be/)

# Table of Contents

1. [Rumor API](#rumor-api)
2. [Render.com Deployment](#rendercom-deployment)
3. [Running Locally](#running-locally)
   - [Docker Container (Option 1, Recommended)](#docker-container-option-1-recommended)
   - [Uncommented Code (Option 2)](#uncommented-code-option-2)
4. [Connecting with MongoDB](#connecting-with-mongodb)
   - [Connecting with MongoDB on Render.com](#connecting-with-mongodb-on-rendercom)
   - [Connecting with MongoDB Atlas](#connecting-with-mongodb-atlas)

# Rumor API

This project was developed during an internship at PlayField by Jelle Vercalsteren. 

The Rumor API is built using FastAPI and relies on a MongoDB backend.<br> It can be deployed on Google Cloud App Engine, Render, or run locally.

## Render.com Deployment

The deployment process is managed through Render.com. <br> Ensure all required environment variables are properly set before initiating the deployment.

## Running Locally

To run the FastAPI application locally, you have two options: running it within a Docker container or uncommenting specific code blocks in the `main.py` and `settings.py` files. Follow the steps below for a seamless local setup.

### Docker Container (Option 1, Recommended)

1. Ensure Docker is installed on your local machine.

2. Create a Docker Compose file to specify the necessary environment variables required by your application. Here's an example:

   ```yaml
   services:
     rumor-raccoon:
       build:
         context: .
       environment:
         BEARER_TOKEN: ${BEARER_TOKEN} 
         RUMOR_URL: ${RUMOR_URL}
         RUMOR_TOKEN: ${RUMOR_TOKEN}
         OPENAI_ORGANIZATION: ${OPENAI_ORGANIZATION}
         OPENAI_API_KEY: ${OPENAI_API_KEY}
         MONGODB_URL: ${MONGODB_URL}
         MONGODB_USER: ${MONGODB_USER}
         MONGODB_PASSWORD: ${MONGODB_PASSWORD}
       ports:
         - 8080:8080 # Ensure the port matches the one specified in the Dockerfile (port forwarding).

   ```
   In your dockerfile change CMD to the following
   ```dockerfile
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
   ```
   
3. Run the following command to start the FastAPI application within a Docker container:

   ```bash
   docker compose up -d
   ```
4. Visit http://localhost:8080/docs to access the FastAPI documentation and explore the API.

### Uncommented Code (Option 2)

1. Open the main.py file and uncomment the following code block
   ```python 
   if __name__ == "__main__":
    settings.logger.info(f"{str(datetime.now())[:-3]} - Starting API")
    uvicorn.run("main:app", host='0.0.0.0', port=default_settings.port)
   ```
2. In the settings.py file, uncomment the following line
   ```python 
   port: int = Field(..., env='PORT') # used to run locally
    ```

3. Create an environment file (.env) and ensure that the required settings can be accessed using os.getenv.
   ```python
   # Credentials
    class RumorCredentials(BaseSettings):
        bearer_token: str = Field(..., env='BEARER_TOKEN')
        rumor_url: str = Field(..., env='RUMOR_URL')
        rumor_token: SecretStr = Field(..., env='RUMOR_TOKEN')
        openai_organization: str = Field(..., env='OPENAI_ORGANIZATION')
        openai_api_key: SecretStr = Field(..., env='OPENAI_API_KEY')
        mongodb_url: str = Field(..., env='MONGODB_URL')
        mongodb_user: str = Field(..., env='MONGODB_USER')
        mongodb_password: SecretStr = Field(..., env='MONGODB_PASSWORD')
    ```
4. Run main.py and go to /docs to explore the FastAPI in action.

## Connecting with MongoDB
A new path, /mongo_connection, has been added to check if the credentials are valid.
### Connecting with MongoDB on render.com
1. Set up your private service on render.
2. Instead of using a connection string, use the host and port format (e.g., mongo-xyz:27017) with MongoClient.
   
```python
mongodb_host = credentials.mongodb_host #e.g. "mongo-xyz"
mongodb_port = credentials.mongodb_port #e.g. "27017" (default port)

conn = MongoClient(mongodb_host, mongodb_port)
```

### Connecting with MongoDB Atlas
> ⚠️ This is required when running local ⚠️
1. Set up your database on MongoDB Atlas and obtain your connection string.
2. Change the RumorCredentials to take the user, password and url.
3. Use the connection string to connect to the database.
```python
mongodb_user = credentials.mongodb_user
mongodb_password = credentials.mongodb_password.get_secret_value()
mongodb_url = credentials.mongodb_url

mongodb_connect = f"mongodb+srv://{mongodb_user}:{mongodb_password}@{mongodb_url}/?retryWrites=true&w=majority"
conn = MongoClient(mongodb_connect)
```
