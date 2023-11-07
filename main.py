from datetime import datetime
from fastapi import FastAPI, APIRouter, BackgroundTasks, HTTPException, Request, Security, status
from fastapi.responses import HTMLResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from helpers import data_helper, mongodb_helper
from pydantic import BaseModel, HttpUrl, parse_obj_as, ValidationError
from typing import Union, Optional
import settings
import uvicorn

default_settings = settings.get_settings()
credentials = settings.get_credentials()
app = FastAPI(title=default_settings.project_name, openapi_tags=default_settings.tags_metadata)

app.mount("/files", StaticFiles(directory="files"), name="files")
templates = Jinja2Templates(directory="files")

security = HTTPBearer(auto_error=False)


class UnauthorizedMessage(BaseModel):
    detail: str = "Bearer token missing or not accepted."


iteration_callback_router = APIRouter()


@iteration_callback_router.post(
    "{$callback_url}", response_model=data_helper.IterationEventReceived
)
def iteration_notification(body: data_helper.IterationEvent):
    pass


@app.get("/")
async def root(auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        return {"message": "This is the API for project Rumor."}


@app.get("/keepalive")
async def keepalive():
    return "OK"


@app.get("/mongo_connection")
async def mongo_connection(auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        result = mongodb_helper._connect_mongo()
        if result.succeeded:
            return {"response": "MongoDB connection is OK"}
        else:
            return {"response": "MongoDB connection is NOT OK"}
        

@app.get("/status_codes", response_class=HTMLResponse)
async def status_codes(request: Request):
    try:
        with open("files/status_codes.md", encoding='utf8') as f:
            status_codes_data = f.read()
        return templates.TemplateResponse("html_md_template.html",
                                          {"request": request, "title": "Status codes",
                                           "data_md": status_codes_data})
    except Exception as e:
        settings.logger.exception(e)
        result = data_helper.Result.fail(f"Can not serve the status codes file.")
        raise HTTPException(status_code=int(result.error_code), detail=str(result))


@app.get("/input/refresh", tags=["rumor actions"])
async def refresh_input(auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        rumor_url = credentials.rumor_url
        rumor_token = credentials.rumor_token.get_secret_value()
        result = data_helper.get_data(rumor_token, rumor_url)
        return {"response": str(result)}


@app.get("/rumor/generate/{tag}", tags=["rumor generation"])
async def generate_summary(request: Request, tag: str, length: int = 700,
                           auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        collection = "sessions"
        result = mongodb_helper.read_dataframe(collection)
        if result.succeeded:
            # Set app_url for keepalive if not set
            if not default_settings.app_url:
                default_settings.set_app_url(request)
            # Generate summary
            sessions_df = result.value
            tag = tag.lower().strip()
            result = data_helper.generate_tag_summary(tag, sessions_df, summary_max_chars=length)
            if result.succeeded:
                return {"summary": str(result)}
            else:
                raise HTTPException(status_code=int(result.error_code), detail=str(result))
        else:
            raise HTTPException(status_code=int(result.error_code), detail=str(result))


@app.post("/rumor/generate", tags=["rumor generation"], callbacks=iteration_callback_router.routes)
async def generate_rumor(request: Request, outline: data_helper.Outline, background_tasks: BackgroundTasks,
                         callback_url: Union[HttpUrl, None] = None,
                         auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        # Start time
        start_time = datetime.now()
        iteration_id = start_time.strftime("%Y%m%d-%H%M%S")
        # Set app_url for keepalive if not set
        if not default_settings.app_url:
            default_settings.set_app_url(request)
        # Run task
        background_tasks.add_task(data_helper.create_rumor_iteration, start_time=start_time,
                                  iteration_id=iteration_id, outline=outline, callback_url=callback_url)
        return {"iteration_id": iteration_id}


@app.put("/rumor/rephrase", tags=["rumor generation"])
async def rephrase_rumor(request: Request, background_tasks: BackgroundTasks, auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        # Start time
        start_time = datetime.now()
        iteration_id = start_time.strftime("%Y%m%d-%H%M%S")
        # Set app_url for keepalive if not set
        if not default_settings.app_url:
            default_settings.set_app_url(request)
        # Run task
        background_tasks.add_task(data_helper.rephrase_rumor_iteration,
                                start_time=start_time, iteration_id=iteration_id)
        
        return {"iteration_id": iteration_id}
   

@app.get("/rumor", tags=["rumor output"], response_model=data_helper.Iteration, response_model_by_alias=False)
async def get_rumor(lang: Optional[str] = "" ,auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        rumor = await get_rumor_iteration(iteration_id="latest", lang=lang,  auth=auth)
        return rumor
    

@app.get("/rumor/pretty", tags=["rumor output"])
async def get_rumor_pretty(request: Request, lang: Optional[str] = "", auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        rumor = await get_rumor_iteration(iteration_id="latest", lang=lang, auth=auth)
        rumor_pretty = data_helper.pretty_outline(rumor)
        return templates.TemplateResponse("html_md_template.html",
                                          {"request": request, "title": "Rumor Output Pretty",
                                           "data_md": rumor_pretty})


@app.get("/rumor/iteration/{iteration_id}", tags=["rumor output"], response_model=data_helper.Iteration,
         response_model_by_alias=False)
async def get_rumor_iteration(iteration_id: str, lang: str = "", auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:

        #check if not latest, then set the iteration_id to the correct collection by lang
        if lang == "en":
            collection = "rumor"
        else:
            if not lang:
                collection = "rumor_nl"
            else:
                collection = "rumor_" + lang

        if iteration_id != "latest" and lang != "en":
            iteration_id += f"_{lang}" if lang else "_nl"

        post_result = mongodb_helper.get_post(collection, iteration_id)
        if post_result.succeeded:
            rumor_post = post_result.value
            try:
                rumor = data_helper.Iteration.parse_obj(rumor_post)
                
                return rumor
            except ValidationError as e:
                settings.logger.exception(e)
                raise HTTPException(status_code=data_helper.StatusCodes.generic_error.value,
                                    detail="Post is not a valid Iteration object and could not be parsed")
        else:
            raise HTTPException(status_code=int(post_result.error_code), detail=str(post_result))


@app.post("/rumor/iteration/update", tags=["rumor actions"])
async def update_rumor_iteration(iteration: data_helper.Iteration,
                                 auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        iteration_id = iteration.iteration_id
        document = iteration.dict(by_alias=False)
        document.pop("iteration_id", None)
        document.pop("_id", None)
        result = mongodb_helper.update_post("rumor", iteration_id, document)
        return {"response": str(result)}


@app.post("/rumor/iterations", tags=["rumor output"], response_model=list[data_helper.Iteration],
          response_model_by_alias=False)
async def get_rumor_iterations(iteration_ids: list[str], auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        rumor_iterations = []
        for iteration_id in iteration_ids:
            collection = data_helper.get_collection_from_iteration(iteration_id)
            post_result = mongodb_helper.get_post(collection, iteration_id)
            if post_result.succeeded:
                rumor_post = post_result.value
                try:
                    rumor = data_helper.Iteration.parse_obj(rumor_post)
                    rumor_iterations.append(rumor)
                except ValidationError as e:
                    settings.logger.exception(e)
                    raise HTTPException(status_code=data_helper.StatusCodes.generic_error.value,
                                        detail="Post is not a valid Iteration object and could not be parsed")
            else:
                raise HTTPException(status_code=int(post_result.error_code), detail=str(post_result))
        return rumor_iterations


@app.get("/rumor/iterations/all", tags=["rumor output"], response_model=list[data_helper.Iteration],
         response_model_by_alias=False)
async def get_rumor_all_iterations(lang: Optional[str] = "", auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        if lang and lang != 'en':
            collection = "rumor_" + lang
        else:
            collection = "rumor"

        posts_result = mongodb_helper.get_posts(collection)

        if posts_result.succeeded:
            rumor_list = posts_result.value
            try:
                rumor = parse_obj_as(list[data_helper.Iteration], rumor_list)
                return rumor
            except ValidationError as e:
                settings.logger.exception(e)
                raise HTTPException(status_code=data_helper.StatusCodes.generic_error.value,
                                    detail="Post is not a valid Iteration object and could not be parsed")
        else:
            raise HTTPException(status_code=int(posts_result.error_code), detail=str(posts_result))


@app.get("/rumor/iterations/metadata", tags=["rumor output"], response_model=list[data_helper.Iteration],
         response_model_by_alias=False, response_model_exclude_none=True)
async def get_rumor_iterations_metadata(lang: Optional[str] = "" , auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        if lang and lang != 'en':
            collection = "rumor_" + lang
        else:
            collection = "rumor"
        posts_result = mongodb_helper.get_posts_metadata(collection)
        if posts_result.succeeded:
            rumor_list = posts_result.value
            try:
                rumor = parse_obj_as(list[data_helper.Iteration], rumor_list)
                return rumor
            except ValidationError as e:
                settings.logger.exception(e)
                raise HTTPException(status_code=data_helper.StatusCodes.generic_error.value,
                                    detail="Post is not a valid Iteration object and could not be parsed")
        else:
            raise HTTPException(status_code=int(posts_result.error_code), detail=str(posts_result))


@app.get("/rumor/analyze/{tag}", tags=["rumor generation"])
async def analyze_tag(tag: str, lang: Optional[str] = 'en', auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        collection = "sessions"
        result = mongodb_helper.read_dataframe(collection)
        if result.succeeded:
            sessions_df = result.value
            tag = tag.lower().strip()
            result = data_helper.generate_analysis(tag, sessions_df)

            translated_result = None  # Initialize the variable
            if result.succeeded:
                if lang and lang != 'en':
                    translated_result = data_helper.translate_text(lang, str(result))
            if translated_result is not None:
                return {"analysis": str(translated_result)}
            
            else:
                return {"analysis": str(result)}

        if not result.succeeded:
            raise HTTPException(status_code=int(result.error_code), detail=str(result))

@app.delete("/rumor/delete/{iteration_id}", tags=["rumor actions"])
async def delete_rumor_iteration(iteration_id: str, auth: HTTPAuthorizationCredentials = Security(security)):
    if (auth is None) or (auth.credentials != credentials.bearer_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=UnauthorizedMessage().detail)
    else:
        try:
            collection = "rumor"
            result = mongodb_helper.delete_post(collection, iteration_id)

            for lang in default_settings.language_codes:
                lang_code = lang['lang']
                collection = "rumor_" + lang['lang']
                iteration_id_lang = iteration_id + "_" + lang_code
                result_lang = mongodb_helper.delete_post(collection, iteration_id_lang)

            return {"succesfully deleted all entries for iteration_id": iteration_id}

        except Exception as e:
            settings.logger.exception(e)
            return {"response": str(e)}
        
# Run locally - uncomment this block
# if __name__ == "__main__":
#     settings.logger.info(f"{str(datetime.now())[:-3]} - Starting API")
#     uvicorn.run("main:app", host='0.0.0.0', port=default_settings.port)

# Log the timestamp when starting the application
settings.logger.info(f"{str(datetime.now())[:-3]} - Starting API")
