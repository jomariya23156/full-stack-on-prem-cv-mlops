import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

DB_PREDICTION_TABLE_NAME = os.getenv('DB_PREDICTION_TABLE_NAME', 'predictions')
DB_API_LOG_TABLE_NAME = os.getenv('DB_API_LOG_TABLE_NAME', 'api_log')

class APILogTable(Base):
    __tablename__ = DB_API_LOG_TABLE_NAME
    id = Column(Integer, primary_key=True)
    request_method = Column(String)
    request_url = Column(String)
    response_status_code = Column(Integer)
    response_message = Column(String)
    timespan = Column(Float)
    created_on = Column(DateTime, default=datetime.now)
    prediction_id = Column(Integer, ForeignKey(f"{DB_PREDICTION_TABLE_NAME}.id"), nullable=True)
    prediction = relationship("PredictionsTable", back_populates="api_log", uselist=False)

class PredictionsTable(Base):
    __tablename__ = DB_PREDICTION_TABLE_NAME
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    input_img = Column(String) # base64-encoded img
    raw_hm_img = Column(String) # base64-encoded img
    overlaid_img = Column(String) # base64-encoded img
    prediction_json = Column(String) # {'class_1': prob_1, 'class_2': prob_2, ...}
    uae_feats = Column(ARRAY(Float, dimensions=1)) # extracted features to be used in drift detection
    bbsd_feats = Column(ARRAY(Float, dimensions=1)) # extracted features to be used in drift detection
    created_on = Column(DateTime, default=datetime.now)
    api_log = relationship("APILogTable", back_populates="prediction")