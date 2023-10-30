import time
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, ValidationError
from typing import ClassVar, Optional
import logging
import os
import sys
import time

# Logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rumor-api")

# Timezone for Unix system
if os.name == "posix":
    os.environ['TZ'] = 'Europe/Brussels'
    time.tzset()


# Settings
class Settings(BaseSettings):
    project_name: str = "RUMOR API"
    app_url: Optional[str] = None
    # port: int = Field(..., env='PORT') # used to run locally
    tags_metadata: ClassVar[str] = [
        {
            "name": "rumor output",
            "description": "Get specific iteration(s) of Rumor output.",
        },
        {
            "name": "rumor generation",
            "description": "Generate Rumor output.",
        },
        {
            "name": "rumor actions",
            "description": "Actions to perform such as refresh input data, update rumor output, ...",
        },
    ]

    def set_app_url(self, request):
        self.app_url = str(request.base_url)


# Credentials
class RumorCredentials(BaseSettings):
    bearer_token: str = Field(..., env='BEARER_TOKEN')
    rumor_url: str = Field(..., env='RUMOR_URL')
    rumor_token: SecretStr = Field(..., env='RUMOR_TOKEN')
    openai_organization: str = Field(..., env='OPENAI_ORGANIZATION')
    openai_api_key: SecretStr = Field(..., env='OPENAI_API_KEY')
    mongodb_host: str = Field(..., env='MONGODB_HOST')
    mongodb_port: int = Field(..., env='MONGODB_PORT')
    google_translate_api: SecretStr = Field(...,env='GOOGLE_TRANSLATE_API')

@lru_cache
def get_settings():
    """retrieves the settings"""
    try:
        return Settings()
    except ValidationError as ve:
        print(ve.json())
        sys.exit(1)


@lru_cache
def get_credentials():
    """retrieves the credentials"""
    try:
        return RumorCredentials()
    except ValidationError as ve:
        print(ve.json())
        sys.exit(1)


