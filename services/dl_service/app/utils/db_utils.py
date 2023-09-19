import os
import json
import logging
import fastapi
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
from .db_tables import Base, PredictionsTable, APILogTable

logger = logging.getLogger('main')

POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_PREDICTION_TABLE_NAME = os.getenv('DB_PREDICTION_TABLE_NAME', 'predictions')
DB_API_LOG_TABLE_NAME = os.getenv('DB_API_LOG_TABLE_NAME', 'api_log')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://dlservice_user:SuperSecurePwdHere@postgres:{POSTGRES_PORT}/dlservice_pg_db')

required_db_tables = [DB_PREDICTION_TABLE_NAME, DB_API_LOG_TABLE_NAME]

def prepare_db() -> None:
    logger.info("Preparing database")
    engine = create_engine(DB_CONNECTION_URL)
    Base.metadata.create_all(engine)
    logger.info("Database is ready.")

def check_db_healthy() -> None:
    engine = create_engine(DB_CONNECTION_URL)
    with engine.connect() as connection:
        result = connection.execute(text(f"select 1 from {DB_API_LOG_TABLE_NAME} limit 1"))
        result.all()

def open_db_session(engine: sqlalchemy.engine) -> sqlalchemy.orm.Session:
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def create_api_log_entry(request_obj: fastapi.Request, resp_code: int, resp_message: str, timespan: float,
                         prediction: Optional[PredictionsTable] = None) -> APILogTable:
    entry = APILogTable(request_method=str(request_obj.method), request_url=str(request_obj.url),
                        response_status_code=resp_code, response_message=resp_message, timespan=timespan,
                        prediction=prediction)
    return entry

def commit_only_api_log_to_db(request_obj: fastapi.Request, resp_code: int, resp_message: str, timespan: float) -> None:
    engine = create_engine(DB_CONNECTION_URL)
    session = open_db_session(engine)
    record = create_api_log_entry(request_obj, resp_code, resp_message, timespan)
    session.add(record)
    session.commit()
    session.close()

def commit_results_to_db(request_obj: fastapi.Request, resp_code: int, resp_message: str, timespan: float,
                         model_name: str, input_img: str, raw_hm_img: str, overlaid_img: str, pred_json: dict,
                         uae_feats: np.ndarray, bbsd_feats: np.ndarray) -> None:
    engine = create_engine(DB_CONNECTION_URL)
    session = open_db_session(engine)
    json_str = json.dumps(pred_json)
    prediction_record = PredictionsTable(model_name=model_name, input_img=input_img, raw_hm_img=raw_hm_img, 
                                         overlaid_img=overlaid_img, prediction_json=json_str, uae_feats=uae_feats,
                                         bbsd_feats=bbsd_feats)
    api_log_record = create_api_log_entry(request_obj, resp_code, resp_message, timespan, prediction=prediction_record)
    session.add(api_log_record)
    session.commit()
    session.close()