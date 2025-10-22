"""
MQTT Broker Integration for Real-Time Sensor Data
Receives sensor data from IoT devices and processes predictions
"""
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
from typing import Callable, Optional, Dict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.edge_inference.inference_engine import EdgeInferenceEngine
from src.utils import setup_logger


class MQTTSensorBroker:
    """
    MQTT broker integration for streaming sensor data
    Subscribes to sensor topics and processes predictions in real-time
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        models_dir: str = "models/saved_models"
    ):
        """
        Initialize MQTT sensor broker
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            username: MQTT username (optional)
            password: MQTT password (optional)
            models_dir: Directory containing trained models
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        
        # Initialize logger
        self.logger = setup_logger('mqtt_broker', 'logs/mqtt_broker.log', 'INFO')
        
        # Initialize inference engine
        self.engine = EdgeInferenceEngine(models_dir)
        try:
            self.engine.load_models()
            self.logger.info("Inference models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"predictive_maintenance_{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Callbacks
        self.prediction_callback: Optional[Callable] = None
        self.alert_callback: Optional[Callable] = None
        
        # Statistics
        self.messages_received = 0
        self.predictions_made = 0
        self.alerts_triggered = 0
        self.errors = 0
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Subscribe to sensor data topics
            topics = [
                ("sensors/+/temperature", 0),
                ("sensors/+/vibration", 0),
                ("sensors/+/data", 0),  # Combined sensor data
            ]
            
            for topic, qos in topics:
                client.subscribe(topic, qos)
                self.logger.info(f"Subscribed to topic: {topic}")
        else:
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            self.messages_received += 1
            
            # Parse JSON message
            payload = json.loads(msg.payload.decode())
            
            self.logger.debug(f"Received message on topic {msg.topic}: {payload}")
            
            # Process sensor data
            self._process_sensor_data(payload, msg.topic)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            self.errors += 1
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.errors += 1
    
    def _process_sensor_data(self, sensor_data: Dict, topic: str):
        """Process incoming sensor data and make prediction"""
        try:
            # Ensure required fields are present
            required_fields = [
                'equipment_id', 'equipment_type', 'temperature',
                'vibration', 'pressure', 'rpm', 'power_consumption',
                'acoustic_emission'
            ]
            
            if not all(field in sensor_data for field in required_fields):
                self.logger.warning(f"Missing required fields in sensor data: {sensor_data}")
                return
            
            # Make prediction
            result = self.engine.predict_single(sensor_data)
            self.predictions_made += 1
            
            # Call prediction callback if registered
            if self.prediction_callback:
                self.prediction_callback(result)
            
            # Check for alerts
            if result['alert']['level'] in ['HIGH', 'CRITICAL']:
                self.alerts_triggered += 1
                self.logger.warning(
                    f"Alert triggered for {sensor_data['equipment_id']}: "
                    f"{result['alert']['level']} - {result['alert']['message']}"
                )
                
                # Call alert callback if registered
                if self.alert_callback:
                    self.alert_callback(result)
                
                # Publish alert to MQTT
                alert_topic = f"alerts/{sensor_data['equipment_id']}"
                self.client.publish(alert_topic, json.dumps(result['alert']))
            
            # Publish prediction result
            result_topic = f"predictions/{sensor_data['equipment_id']}"
            self.client.publish(result_topic, json.dumps(result))
            
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {e}")
            self.errors += 1
    
    def set_prediction_callback(self, callback: Callable):
        """Set callback for when predictions are made"""
        self.prediction_callback = callback
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for when alerts are triggered"""
        self.alert_callback = callback
    
    def start(self):
        """Start MQTT broker connection and message processing"""
        self.logger.info("Starting MQTT sensor broker...")
        
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_forever()
        except Exception as e:
            self.logger.error(f"Failed to start MQTT broker: {e}")
            raise
    
    def stop(self):
        """Stop MQTT broker connection"""
        self.logger.info("Stopping MQTT sensor broker...")
        self.client.disconnect()
        self.client.loop_stop()
    
    def get_statistics(self) -> Dict:
        """Get broker statistics"""
        return {
            'messages_received': self.messages_received,
            'predictions_made': self.predictions_made,
            'alerts_triggered': self.alerts_triggered,
            'errors': self.errors,
            'engine_stats': self.engine.get_performance_stats()
        }


def main():
    """Run MQTT broker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MQTT Sensor Data Broker')
    parser.add_argument('--host', default='localhost', help='MQTT broker host')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--username', help='MQTT username')
    parser.add_argument('--password', help='MQTT password')
    parser.add_argument('--models-dir', default='models/saved_models', help='Models directory')
    
    args = parser.parse_args()
    
    broker = MQTTSensorBroker(
        broker_host=args.host,
        broker_port=args.port,
        username=args.username,
        password=args.password,
        models_dir=args.models_dir
    )
    
    # Example callbacks
    def on_prediction(result):
        print(f"Prediction: {result['equipment_id']} - "
              f"Failure Prob: {result['predictions']['failure_probability']:.2%}")
    
    def on_alert(result):
        print(f"🚨 ALERT: {result['equipment_id']} - "
              f"{result['alert']['level']}: {result['alert']['message']}")
    
    broker.set_prediction_callback(on_prediction)
    broker.set_alert_callback(on_alert)
    
    try:
        broker.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        broker.stop()


if __name__ == '__main__':
    main()
