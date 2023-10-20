from datetime import datetime
from helpers import data_helper
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from typing import Any
import pandas as pd
import pymongo
import settings

credentials = settings.get_credentials()

def _connect_mongo(db="rumor"):
    # Get credentials
    mongodb_user = credentials.mongodb_user
    mongodb_password = credentials.mongodb_password.get_secret_value()
    mongodb_url = credentials.mongodb_url
    # Connect
    try:
        mongodb_connect = f"mongodb+srv://{mongodb_user}:{mongodb_password}@{mongodb_url}/?retryWrites=true&w=majority"
        conn = MongoClient(mongodb_connect)
        conn.admin.command('ismaster')
        return data_helper.Result.ok(conn[db])
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Could not connect to database. Check the connection parameters "
                                       f"and verify connection can be made.",
                                       error_code=data_helper.StatusCodes.database_connect_error.value)


def add_post(collection: str, data, doc_id: str = None, runtime: str = None, status: str = None):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    # Document
    now = datetime.now()
    if doc_id is None:
        doc_id = now.strftime("%Y%m%d-%H%M%S")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+"Z"
    post = {"_id": doc_id, "datetime": timestamp}
    if status is not None:
        post["status"] = status
    if runtime is not None:
        post["runtime"] = runtime
    post["data"] = data
    # Insert
    try:
        connect_collection.insert_one(post)
        return data_helper.Result.ok(f"Data was written to database collection {collection}")
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be written to database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_write_error.value)


def update_post(collection: str, doc_id: str, document: dict[str, Any]):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    # Document
    query = {'_id': doc_id}

    # Update
    try:
        connect_collection.update_one(query, {"$set": document}, upsert=False)
        return data_helper.Result.ok(f"Data was written to database collection {collection}")
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be written to database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_write_error.value)


def get_post(collection: str, iteration_id: str):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    if iteration_id == "latest":
        query = {}
        search_error = f"No iterations found in database collection {collection}"
    else:
        query = {'_id': iteration_id}
        search_error = f"No iteration found for id {iteration_id} in database collection {collection}"

    # Read
    try:
        data = connect_collection.find_one(query, sort=[("_id", pymongo.DESCENDING)])
        if data is not None:
            return data_helper.Result.ok(data)
        else:
            return data_helper.Result.fail(search_error, error_code=data_helper.StatusCodes.database_empty_error.value)
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be read from database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_read_error.value)


def get_posts(collection: str):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    # Read
    try:
        data = list(connect_collection.find({}).sort("_id", pymongo.DESCENDING))
        return data_helper.Result.ok(data)
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be read from database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_read_error.value)


def get_posts_metadata(collection: str):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    # Read
    try:
        data = list(connect_collection.find({"status": "done"},
                                            {"_id": 1, "status": 1, "datetime": 1}).sort("_id", pymongo.DESCENDING))
        return data_helper.Result.ok(data)
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be read from database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_read_error.value)


def add_dataframe(collection: str, df: pd.DataFrame):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    data = df.to_dict(orient='records')

    # Insert

    try:
        results = connect_collection.insert_many(data, ordered=False)
        settings.logger.info(f"Inserted {len(results.inserted_ids)} answers.")
        return data_helper.Result.ok(f"Data was written to database collection {collection}")
    except BulkWriteError as e:
        # ignore E11000 duplicate key error collection
        error_list = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
        if len(error_list) > 0:
            return data_helper.Result.fail(f"Data could not be written to database collection {collection}",
                                           error_code=data_helper.StatusCodes.database_write_error.value)
        else:
            return data_helper.Result.ok(f"Data was written to database collection {collection}"
                                         f" & duplicates were ignored")
    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Data could not be written to database collection {collection}",
                                       error_code=data_helper.StatusCodes.database_write_error.value)


def read_dataframe(collection: str):
    result = _connect_mongo()
    if result.succeeded:
        db = result.value
        connect_collection = db[collection]
    else:
        return result

    try:
        cursor = connect_collection.find({})
        data = list(cursor)
        df = pd.DataFrame.from_dict(data)
        if not df.empty:
            return data_helper.Result.ok(df)
        else:
            return data_helper.Result.fail(f"Pandas DataFrame read from collection {collection} is empty.",
                                           error_code=data_helper.StatusCodes.database_empty_error.value)

    except Exception as e:
        settings.logger.exception(e)
        return data_helper.Result.fail(f"Could not read data from collection {collection} as Pandas DataFrame",
                                       error_code=data_helper.StatusCodes.database_data_parse_error.value)
