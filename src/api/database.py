"""
Database module for storing time-series sensor data, predictions, and alerts
Uses PostgreSQL/TimescaleDB for efficient time-series operations
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
from contextlib import contextmanager

Base = declarative_base()


class SensorReading(Base):
    """Sensor reading time-series data"""
    __tablename__ = 'sensor_readings'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equipment_id = Column(String, index=True, nullable=False)
    equipment_type = Column(String, nullable=False)
    temperature = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    rpm = Column(Float, nullable=False)
    power_consumption = Column(Float, nullable=False)
    acoustic_emission = Column(Float, nullable=False)


class Prediction(Base):
    """Model prediction results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equipment_id = Column(String, index=True, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    failure_probability = Column(Float, nullable=False)
    failure_predicted = Column(Boolean, nullable=False)
    inference_time_ms = Column(Float, nullable=False)


class Alert(Base):
    """System alerts"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equipment_id = Column(String, index=True, nullable=False)
    alert_level = Column(String, nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    message = Column(String, nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String, nullable=True)


class ModelMetrics(Base):
    """Model training/evaluation metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_type = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)
    dataset_size = Column(Integer, nullable=False)


class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL connection string
                         Format: postgresql://user:password@host:port/dbname
        """
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/predictive_maintenance'
            )
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
        # Create hypertable for TimescaleDB (if using TimescaleDB)
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT create_hypertable('sensor_readings', 'timestamp', if_not_exists => TRUE);")
                conn.execute("SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);")
                conn.execute("SELECT create_hypertable('alerts', 'timestamp', if_not_exists => TRUE);")
                conn.commit()
        except Exception:
            # Not using TimescaleDB, skip
            pass
    
    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def insert_sensor_reading(self, data: Dict[str, Any]) -> int:
        """Insert sensor reading"""
        with self.get_session() as session:
            reading = SensorReading(**data)
            session.add(reading)
            session.flush()
            return reading.id
    
    def insert_prediction(self, data: Dict[str, Any]) -> int:
        """Insert prediction result"""
        with self.get_session() as session:
            prediction = Prediction(**data)
            session.add(prediction)
            session.flush()
            return prediction.id
    
    def insert_alert(self, data: Dict[str, Any]) -> int:
        """Insert alert"""
        with self.get_session() as session:
            alert = Alert(**data)
            session.add(alert)
            session.flush()
            return alert.id
    
    def get_recent_readings(self, equipment_id: str, limit: int = 100) -> List[SensorReading]:
        """Get recent sensor readings for equipment"""
        with self.get_session() as session:
            return session.query(SensorReading)\
                .filter(SensorReading.equipment_id == equipment_id)\
                .order_by(SensorReading.timestamp.desc())\
                .limit(limit)\
                .all()
    
    def get_active_alerts(self, equipment_id: Optional[str] = None) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        with self.get_session() as session:
            query = session.query(Alert)\
                .filter(Alert.acknowledged == False)
            
            if equipment_id:
                query = query.filter(Alert.equipment_id == equipment_id)
            
            return query.order_by(Alert.timestamp.desc()).all()
    
    def acknowledge_alert(self, alert_id: int, user: str):
        """Acknowledge an alert"""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = user
    
    def get_equipment_health_history(
        self, 
        equipment_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, List]:
        """Get equipment health history (readings + predictions)"""
        with self.get_session() as session:
            readings = session.query(SensorReading)\
                .filter(
                    SensorReading.equipment_id == equipment_id,
                    SensorReading.timestamp >= start_time,
                    SensorReading.timestamp <= end_time
                )\
                .order_by(SensorReading.timestamp)\
                .all()
            
            predictions = session.query(Prediction)\
                .filter(
                    Prediction.equipment_id == equipment_id,
                    Prediction.timestamp >= start_time,
                    Prediction.timestamp <= end_time
                )\
                .order_by(Prediction.timestamp)\
                .all()
            
            return {
                'readings': readings,
                'predictions': predictions
            }


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager
