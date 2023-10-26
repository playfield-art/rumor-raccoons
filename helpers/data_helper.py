from datetime import datetime
from enum import Enum
from helpers import mongodb_helper
from pydantic import BaseModel, Field, HttpUrl
from typing import Any, Optional, Union
import httpx
import json
import openai
import pandas as pd
import settings
import tiktoken
from tenacity import (Retrying, RetryError, retry_if_not_exception_type, stop_after_attempt,
                      wait_exponential)  # for exponential backoff

credentials = settings.get_credentials()

class Tag(BaseModel):
    tag: str
    length: int = 700


class Section(BaseModel):
    title: str
    tags: list[Tag]
    summary: Optional[dict[str, str]] = {}


class Outline(BaseModel):
    intro: str
    sections: list[Section]
    outro: str


class Iteration(BaseModel):
    iteration_id: str = Field(alias='_id')
    datetime: str
    runtime: str | None = None
    status: str | None = None
    data: Union[Outline, str] | None = None

    class Config:
        populate_by_name = True


class IterationEvent(BaseModel):
    iteration_id: str
    runtime: str | None = None
    status: str


class IterationEventReceived(BaseModel):
    ok: bool


class StatusCodes(Enum):
    # generic
    generic_error = 520
    # openai
    openai_auth_error = 530
    openai_load_error = 531
    openai_invalid_request_error = 532
    openai_connect_error = 533
    # database
    database_connect_error = 580
    database_read_error = 581
    database_empty_error = 582
    database_write_error = 583
    database_data_parse_error = 584


class TaskStatus(Enum):
    processing = 1
    done = 2
    error = 3


class Result:
    """Represents the outcome of an operation.

        Attributes
        ----------
        success : bool
            A flag that is set to True if the operation was successful, False if
            the operation failed.
        value : object
            The result of the operation if successful, value is None if operation
            failed or if the operation has no return value.
        error : str
            Error message detailing why the operation failed, value is None if
            operation was successful.
        error_code: int
            Error code must be a valid HTTP response status code, custom codes can be used in ranges as 4xx and 5xx
            see https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
        """

    def __init__(self, success: bool, value: Union[Any, None], error: Union[str, None], error_code: Union[int, None]):
        self.success = success
        self.value = value
        self.error = error
        self.error_code = error_code

    @property
    def succeeded(self):
        """True if operation succeeded, False if failed (read-only)."""
        return self.success

    def __str__(self):
        if self.success:
            return f"{self.value}"
        else:
            return f"ERROR - {self.error}"

    @classmethod
    def fail(cls, error: str, error_code: int = StatusCodes.generic_error.value):
        """Create a Result object for operation with error."""
        return cls(False, value=None, error=error, error_code=error_code)

    @classmethod
    def ok(cls, value=None):
        """Create a Result object for successful operation."""
        return cls(True, value=value, error=None, error_code=None)


def keepalive():
    url = f"{settings.get_settings().app_url}keepalive"
    try:
        httpx.get(url=url, timeout=5)
        result = Result.ok(f"Successfully pinged {url}")
    except Exception as e:
        settings.logger.exception(e)
        result = Result.fail(f"Failed to ping {url}.")
    return result


# Get Data
def get_data(rumor_token: str, url: str, chunk_amount: int = 50):

    offset = 0
    all_sessions = []
    all_json = []

    while True:
        query = """
        query GetSessions($chunk_amount: Int!, $offset: Int!){
        sessions(
            pagination: { limit: $chunk_amount , start: $offset }
            filters: { moderated: { eq: true } }
        ) {
            data {
            id
            attributes {
                language
                session_id
                date
                time
                createdAt
                updatedAt
                moderated
                answers (pagination: {limit: 30}) {
                id
                question {
                    data {
                    id
                    attributes {
                        title
                        description
                        question_tags {
                        data {
                            attributes {
                            name
                            }
                        }
                        }
                    }
                    }
                }
                original_transcript
                moderated_transcript
                common_language
                }
            }
            }
        }
        }
        """
        variables = {
            "chunk_amount": chunk_amount,
            "offset": offset
        }
        headers = {"Authorization": f"Bearer {rumor_token}"}
        payload = {'query': query, 'variables': variables}

        try:
            req = httpx.post(url=url, json=payload,headers=headers, timeout=300)
        except Exception as e:
            settings.logger.exception(e)
            return Result.fail(f"Failed to get data from {url}")
        
        settings.logger.info(f"Response status from {url}: {req}")

        try:
            json_data = req.json()
            sessions = json_data.get("data", {}).get("sessions", {}).get("data", [])
        except Exception as e:
            settings.logger.exception(e)
            return Result.fail("Failed to parse JSON response")

        if not sessions:
            break  # No more sessions in the response

        sessions_df = parse_data(json_data)
        all_json.extend(sessions)
        all_sessions.append(sessions_df)
        offset += chunk_amount
    
    combined_sessions_df = pd.concat(all_sessions, ignore_index=True)

    result1 = mongodb_helper.add_post("json_sessions", all_json)
    result2 = mongodb_helper.add_dataframe("sessions", combined_sessions_df)
    settings.logger.info(str(result1))
    settings.logger.info(str(result2))

    if result1.succeeded and result2.succeeded:
        return Result.ok("OK - Data was refreshed and written to the database")
    else:
        return Result.fail("Data could not be written to the database",error_code=StatusCodes.database_write_error.value)


# Parse the json data to dataframe
def parse_data(json_data):
    # Data processing manual
    sessions_json = json_data['data']['sessions']['data']
    sessions = []
    for session_json in sessions_json:
        session_id = session_json['attributes']['session_id']
        language = session_json['attributes']['language']
        date = session_json['attributes']['date']
        time = session_json['attributes']['time']
        createdAt = session_json['attributes']['createdAt']
        updatedAt = session_json['attributes']['updatedAt']
        for answer in session_json['attributes']['answers']:
            session = dict()
            tags_main = answer['question']['data']['attributes']['question_tags']['data']
            tags_list = []
            for tag in tags_main:
                tags_list.append(tag['attributes']['name'].lower().strip())
            session['_id'] = f"{session_id}-{answer['id']}"
            session['session_id'] = session_id
            session['language'] = language
            session['date'] = date
            session['time'] = time
            session['createdAt'] = createdAt
            session['updatedAt'] = updatedAt
            session['answer_id'] = answer['id']
            session['tags'] = tags_list
            session['question'] = answer['question']['data']['attributes']['title']
            session['question_description'] = answer['question']['data']['attributes']['description']
            session['moderated_transcript'] = answer['moderated_transcript']
            session['original_transcript'] = answer['original_transcript']
            session['common_language'] = answer['common_language']
            sessions.append(session)

    sessions_df = pd.DataFrame.from_dict(sessions)
    tags = sessions_df.pop('tags')
    tags = tags.apply(lambda x: pd.Series(1, index=x)).fillna(0)
    sessions_df = pd.concat([sessions_df, tags.astype('int32')], axis=1)

    return sessions_df


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base", model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Prepare DataFrame tag data
def prepare_tag_data(tag: str, sessions_df: pd.DataFrame):
    df_filter = sessions_df[tag] == 1
    tag_data = sessions_df[df_filter]

    if len(tag_data.moderated_transcript.dropna()) == 0:
        return Result.fail(f"The supplied tag <{tag}> has no data.")

    # Transform dataframe data to large string for gpt
    entries_data = tag_data.moderated_transcript.dropna().tolist()
    entry_prefix = "Inzending: "
    entries = [entry_prefix + entry for entry in entries_data]
    formatted_data = prepare_list_split_tokens(entries)
    # Return
    return Result.ok(formatted_data)


# split list in multiple parts based on tokens
def prepare_list_split_tokens(data: list, split_tokens: int = 7200):
    formatted_data = []
    data_string = '\n'.join(data)
    num_tokens_full = num_tokens_from_string(data_string)
    num_indexes = -(-num_tokens_full // split_tokens)  # round up without math
    if num_indexes == 1:
        formatted_data.append(data_string)
    else:
        index = 0
        formatted_data.append("")
        tokens_per_part = -(-num_tokens_full // num_indexes)  # round up without math
        for entry in data:
            formatted_data[index] += str(entry)
            num_tokens = num_tokens_from_string(formatted_data[index])
            if num_tokens > tokens_per_part:
                index += 1
                formatted_data.append("")

    return formatted_data


# Create Rumor output outline pretty
def pretty_outline(iteration: Iteration):
    outline = ""
    # Title
    outline += "# Rumor  \n"
    outline += "## Intro  \n"
    outline += f"{str(iteration.data.intro or '')}  \n\n"
    for section in iteration.data.sections:
        outline += f"## {section.title}  \n"
        outline += f"{section.summary.get('overall', '')}  \n\n"
    outline += "## Outro  \n"
    outline += f"{str(iteration.data.outro or '')}  \n"
    return outline


def create_rumor_iteration(start_time: datetime, iteration_id: str, outline: Outline,
                           callback_url: Union[HttpUrl, None] = None):
    
    # Database: Set status to processing
    status = TaskStatus.processing
    mongo_result = mongodb_helper.add_post("rumor", None, doc_id=iteration_id, status=status.name)
    settings.logger.info(mongo_result)

    if not mongo_result.succeeded:
        return Result.fail("Data could not be written to the database", error_code=StatusCodes.database_write_error.value)

    # read data from database
    sessions_df, status = read_input_data(iteration_id, status)

    # Generate summaries for each section
    for section in outline.sections:
        # Console output for clarity
        divider = "-" * 10
        settings.logger.info(f"{divider} Generating summaries for section {section.title}")
        status = generate_summaries(section, sessions_df, iteration_id, status)

    runtime = calculate_runtime(start_time)
    settings.logger.info(f"Total runtime for iteration {iteration_id}: {runtime}")

    outline_dict = outline.dict()
    result_update, status = update_database(iteration_id, runtime, status, outline_dict)

    if callback_url:
        handle_callback(iteration_id, runtime, status.name, callback_url)

    return Result.ok(outline)


def read_input_data(iteration_id: str, status):
    result = mongodb_helper.read_dataframe("sessions")
    
    if not result.succeeded:
        updated_status = TaskStatus.error
        update_dict = {"status": updated_status.name, "data": result.error}
        mongodb_helper.update_post("rumor", doc_id=iteration_id, document=update_dict)
        return result, updated_status
    
    return result.value, status


def generate_summaries(section: Outline , sessions_df: pd.DataFrame , iteration_id:str, status: TaskStatus):
    #For each tag in a section -> generate summary
    for tag in section.tags:
        # Console output for clarity
        tag_name = tag.tag.lower().strip()
        divider = "-" * 10
        settings.logger.info(f"{divider * 2} Generating summary for tag {tag_name}")

        # Generate summary
        summary_result = generate_tag_summary(tag_name, sessions_df, summary_max_chars=tag.length)
        if summary_result.succeeded:
            section.summary[tag_name] = str(summary_result.value or "") # or "" to prevent NoneType error
        elif summary_result.error_code in [code.value for code in StatusCodes if code.name.startswith("openai")]:
            result, status = handle_openai_error(iteration_id, summary_result.error, status) 
            return status
        else:
            settings.logger.info(str(summary_result))
            section.summary[tag_name] = "" 

    # All summaries in a list to generate an overall summary
    overall_summary_list = [str(summ) for summ in section.summary.values()]

    if len(overall_summary_list) in [0, 1]: # if there is only one summary = overall summary
        section.summary["overall"] = '\n'.join(overall_summary_list).strip()
    else:
        overall_summary_list_split = prepare_list_split_tokens(overall_summary_list) 
        full_summary_result = generate_multipart_summary(overall_summary_list_split)
        section.summary["overall"] = str(full_summary_result.value or "") 
    
    return Result.ok(section)


def calculate_runtime(start_time):
    runtime_timedelta = datetime.now() - start_time
    return str(runtime_timedelta)[:-3]


def update_database(iteration_id, runtime, status, outline_dict):
    updated_status = TaskStatus.done
    update_dict = {"runtime": runtime,"status": updated_status.name, "data": outline_dict}
    result_update = mongodb_helper.update_post("rumor", doc_id=iteration_id, document=update_dict)
    if not result_update.succeeded:
        return result_update, updated_status
    
    return result_update, updated_status


def handle_openai_error(iteration_id, error_data):
    updated_status = TaskStatus.error
    update_dict = {"status": updated_status.name, "data": error_data}
    mongodb_helper.update_post(
        "rumor", doc_id=iteration_id, document=update_dict)
    return Result.fail(error_data, error_code=StatusCodes.openai_invalid_request_error.value), updated_status


def handle_callback(iteration_id, runtime, status, callback_url):
    params = {'id': iteration_id}
    callback_json = {
        "iteration_id": iteration_id,
        "runtime": runtime,
        "status": status
    }
    try:
        httpx.post(url=callback_url, params=params,
                   json=callback_json, timeout=20)
    except Exception as e:
        settings.logger.exception(e)
        settings.logger.error(f"Callback URL {callback_url} is not available")
        return Result.fail(f"Callback URL {callback_url} is not available")


# Generate tag summary
def generate_tag_summary(tag: str, data: pd.DataFrame, temperature: float = 0.5, summary_max_chars: int = 700):
    # pre-check
    if tag not in data:
        return Result.fail(f"The supplied tag <{tag}> was not found in the data.")

    system_instruct = """
    Je bent een intelligente assistent. Je maakt een samenvatting van resultaten van 
    een onderzoek. Schrijf een samenhangend geheel. Doe dit met volgende 
    richtlijnen:
    Alle tekst wordt verteld vanuit het personage "Rumor". Rumor is de installatie 
    zelf en spreekt de toeschouwer aan als mensheid met "jullie" en rekent zichzelf 
    daar nooit bij. Ze vertelt over de gemeenschappelijke toekomst als voldongen 
    feit, als een voorspelling. Maak voorspellingen die beginnen met bijvoorbeeld 
    "Mensen zullen", "er zal", ...
    """

    tag_prompt = f"Het onderzoek bevat resultaten over {tag}."
    summary_limit_prompt = f"Verkort volgende samenvatting tot maximaal {summary_max_chars} tekens exclusief spaties."

    # get data for the tag as string
    result = prepare_tag_data(tag, data)
    # check result and generate if succeeded
    if result.succeeded:
        tag_data = result.value
        result_gpt = call_openai_gpt(temperature, system_instruct, tag_prompt, tag_data, summary_max_chars,
                                     summary_limit_prompt)
        return result_gpt
    else:
        return result


# Generate multipart summary will create a single summary from multiple summaries
def generate_multipart_summary(data: list, temperature: float = 0.5, summary_max_chars: int = 700):
    system_instruct = """
    Je bent een intelligente assistent. Je maakt een samenvatting van tekst. Schrijf een samenhangend geheel. 
    Doe dit met volgende richtlijnen:
    Alle tekst wordt verteld vanuit het personage "Rumor". Rumor is de installatie 
    zelf en spreekt de toeschouwer aan als mensheid met "jullie" en rekent zichzelf 
    daar nooit bij. Ze vertelt over de gemeenschappelijke toekomst als voldongen 
    feit, als een voorspelling. Maak voorspellingen die beginnen met bijvoorbeeld 
    "Mensen zullen", "er zal", ...
    """

    summary_prompt = f"Maak een samenvatting van volgende tekst."
    summary_limit_prompt = f"Verkort de samenvatting tot maximaal {summary_max_chars} tekens exclusief spaties."

    result_gpt = call_openai_gpt(temperature, system_instruct, summary_prompt, data, summary_max_chars,
                                 summary_limit_prompt)
    return result_gpt


def generate_analysis(tag: str, data: pd.DataFrame, temperature: float = 0.5):
    # pre-check
    if tag not in data:
        return Result.fail(f"The supplied tag <{tag}> was not found in the data.")

    system_instruct = """
    Je bent een intelligente assistent. Je analyseert de resultaten van een onderzoek.
    """

    tag_prompt = f"Welk(e) {tag} komt het meest voor in de resultaten van het onderzoek?"

    # get data for the tag as string
    result = prepare_tag_data(tag, data)
    # check result and generate if succeeded
    if result.succeeded:
        tag_data = result.value
        result_gpt = call_openai_gpt(temperature, system_instruct, tag_prompt, tag_data)
        return result_gpt
    else:
        return result


def call_openai_gpt(temperature: float, system_instruct: str, first_prompt: str, data: list,
                    summary_max_chars: int = 700, summary_limit_prompt: str = ""):
    # Looping over the data list is not implemented yet, will need to be implemented later when
    # more responses are present in the input data

    # Check how many entries are in the data list and generate accordingly
    if len(data) == 1: # if there is only one entry
        gpt_response_result = completion_openai(temperature, system_instruct, first_prompt, data[0])
        if gpt_response_result.succeeded:
            gpt_response = gpt_response_result.value
            settings.logger.info(f"Generated first iteration of summary with {len(gpt_response)} characters.")

            if summary_limit_prompt.strip():
                if len(gpt_response) > summary_max_chars:
                    gpt_response_result = completion_openai(temperature, system_instruct, summary_limit_prompt,
                                                            gpt_response)
                    gpt_response = gpt_response_result.value
                    settings.logger.info(f"Generated second iteration of summary with {len(gpt_response)} characters.")

            return Result.ok(gpt_response)
        else:
            return gpt_response_result
        
    else: # if there are multiple entries -> split in multiple parts -> generate -> combine -> multipart summary
        gpt_response_all = []
        for number, text in enumerate(data, start=1):
            gpt_response_result = completion_openai(temperature, system_instruct, first_prompt, text)
            if gpt_response_result.succeeded:
                gpt_response = gpt_response_result.value
                gpt_response_all.append(gpt_response)
                settings.logger.info(f"Generated first iteration part {number}/{len(data)} of summary "
                                     f"with {len(gpt_response)} characters.")
            else:
                return gpt_response_result
        gpt_response_all_split = prepare_list_split_tokens(gpt_response_all)
        generate_result = generate_multipart_summary(gpt_response_all_split)
        return generate_result


def completion_openai(temperature: float, system_instruct: str, prompt: str, text: str, max_tries: int = 7):
    # Try connection
    try:
        # Set required parameters & retrieve model
        openai.organization = credentials.openai_organization
        openai.api_key = credentials.openai_api_key.get_secret_value()
        openai.Model.retrieve("gpt-4")
    except openai.error.AuthenticationError as auth_error:
        return Result.fail(str(auth_error), error_code=StatusCodes.openai_auth_error.value)
    except Exception as e:
        settings.logger.exception(e)
        return Result.fail("Failed to load the OpenAI GPT model.", error_code=StatusCodes.openai_load_error.value)

    # Try completion
    try:
        for attempt in Retrying(wait=wait_exponential(multiplier=30, min=30, max=1200),
                                stop=stop_after_attempt(max_tries),
                                retry=retry_if_not_exception_type(openai.error.InvalidRequestError)):
            with attempt:
                # Send keepalive request with keepalive function
                ping = keepalive()

                # Get OpenAI response
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    temperature=temperature,
                    top_p=1,
                    messages=[
                        {"role": "system", "content": system_instruct},
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": text}
                    ]
                )

                # Return response
                gpt_response = completion.choices[0].message.content
                settings.logger.debug(gpt_response)
                return Result.ok(gpt_response)
    except openai.error.InvalidRequestError as ire:
        # Handle Invalid Request error
        return Result.fail(str(ire), error_code=StatusCodes.openai_invalid_request_error.value)
    except openai.error.APIError as e:
        # Handle API error
        settings.logger.exception(f"OpenAI API returned an API Error: {e}")
    except openai.error.APIConnectionError as e:
        # Handle connection error
        settings.logger.exception(f"Failed to connect to OpenAI API: {e}")
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        settings.logger.exception(f"OpenAI API request exceeded rate limit: {e}")
    except openai.error.ServiceUnavailableError as e:
        # Handle API error
        settings.logger.exception(f"OpenAI API returned a Service Unavailable Error: {e}")
    except openai.error.Timeout as e:
        # Handle API error
        settings.logger.exception(f"OpenAI API returned a Timeout Error: {e}")
    except RetryError:
        return Result.fail("Failed to call the OpenAI GPT model.", error_code=StatusCodes.openai_connect_error.value)
    except Exception as e:
        # Other errors
        settings.logger.exception(e)
