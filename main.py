"""
AI-Driven Urban Security & Surveillance System
A framework for intelligent urban monitoring with privacy-conscious design
"""

import cv2
import numpy as np
import threading
import time
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from cryptography.fernet import Fernet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Data structure for security alerts"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    location: str
    confidence: float
    description: str
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL

class DataEncryption:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data).decode()

class FaceDetection:
    """Privacy-conscious face detection (detection only, no identification)"""
    
    def __init__(self):
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame, return bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]

class BehaviorAnalysis:
    """Analyze movement patterns and detect anomalies"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.motion_history = []
        self.anomaly_threshold = 0.15  # Reduced from 0.7 to prevent false positives
        self.alert_cooldown = 10  # Seconds between alerts
        self.last_alert_time = 0
        self.motion_buffer = []  # Store recent motion data
        self.buffer_size = 10
    
    def detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion in the frame"""
        fg_mask = self.background_subtractor.apply(frame)
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return fg_mask
    
    def analyze_behavior(self, frame: np.ndarray) -> Dict:
        """Analyze behavior patterns in the frame with improved logic"""
        motion_mask = self.detect_motion(frame)
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        # Add to motion buffer for trend analysis
        self.motion_buffer.append(motion_ratio)
        if len(self.motion_buffer) > self.buffer_size:
            self.motion_buffer.pop(0)
        
        # Calculate motion variance for anomaly detection
        if len(self.motion_buffer) >= 5:
            motion_variance = np.var(self.motion_buffer)
            avg_motion = np.mean(self.motion_buffer)
            
            # Improved anomaly detection based on sudden changes
            is_anomalous = (motion_ratio > self.anomaly_threshold and 
                          motion_variance > 0.01 and 
                          avg_motion > 0.05)
        else:
            is_anomalous = motion_ratio > self.anomaly_threshold
        
        # Apply cooldown to prevent spam alerts
        current_time = time.time()
        if is_anomalous and (current_time - self.last_alert_time) < self.alert_cooldown:
            is_anomalous = False
        elif is_anomalous:
            self.last_alert_time = current_time
        
        return {
            'motion_ratio': motion_ratio,
            'is_anomalous': is_anomalous,
            'motion_pixels': motion_pixels,
            'motion_variance': motion_variance if len(self.motion_buffer) >= 5 else 0,
            'timestamp': datetime.now()
        }

class CrowdDensityMonitor:
    """Monitor crowd density to prevent overcrowding"""
    
    def __init__(self, max_capacity: int = 100):
        self.max_capacity = max_capacity
        self.density_threshold = 0.8  # 80% capacity warning
    
    def estimate_crowd_density(self, face_count: int, area_size: float = 1000) -> Dict:
        """Estimate crowd density based on detected faces"""
        density = face_count / area_size * 100  # people per 100 sq units
        capacity_ratio = face_count / self.max_capacity
        
        status = "NORMAL"
        if capacity_ratio > 1.0:
            status = "OVERCROWDED"
        elif capacity_ratio > self.density_threshold:
            status = "HIGH_DENSITY"
        
        return {
            'face_count': face_count,
            'density': density,
            'capacity_ratio': capacity_ratio,
            'status': status,
            'timestamp': datetime.now()
        }

class AlertSystem:
    """Manages and dispatches security alerts"""
    
    def __init__(self):
        self.alerts = []
        self.encryption = DataEncryption()
    
    def generate_alert(self, alert_type: str, location: str, 
                      confidence: float, description: str) -> Alert:
        """Generate a new security alert"""
        alert_id = hashlib.md5(
            f"{alert_type}{location}{time.time()}".encode()
        ).hexdigest()[:8]
        
        # Determine priority based on alert type and confidence
        if alert_type in ["SUSPICIOUS_BEHAVIOR", "OVERCROWDING"] and confidence > 0.8:
            priority = "HIGH"
        elif confidence > 0.6:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            location=location,
            confidence=confidence,
            description=description,
            priority=priority
        )
        
        self.alerts.append(alert)
        self.dispatch_alert(alert)
        return alert
    
    def dispatch_alert(self, alert: Alert):
        """Send alert to appropriate authorities"""
        # Encrypt sensitive alert data
        alert_data = {
            'id': alert.alert_id,
            'type': alert.alert_type,
            'location': alert.location,
            'confidence': alert.confidence,
            'description': alert.description,
            'priority': alert.priority,
            'timestamp': alert.timestamp.isoformat()
        }
        
        encrypted_data = self.encryption.encrypt_data(json.dumps(alert_data))
        
        logger.info(f"ALERT DISPATCHED: {alert.alert_type} at {alert.location} "
                   f"(Confidence: {alert.confidence:.2f}, Priority: {alert.priority})")
        
        # In a real system, this would send to law enforcement systems
        self.send_to_authorities(encrypted_data, alert.priority)
    
    def send_to_authorities(self, encrypted_data: bytes, priority: str):
        """Simulate sending encrypted alert to law enforcement"""
        if priority == "CRITICAL":
            logger.warning("CRITICAL ALERT - Immediate response required")
        # Implementation would integrate with actual emergency systems

class UrbanSecuritySystem:
    """Main surveillance system coordinator"""
    
    def __init__(self, camera_sources: List[str], location: str = "Urban Area"):
        self.camera_sources = camera_sources
        self.location = location
        self.face_detector = FaceDetection()
        self.behavior_analyzer = BehaviorAnalysis()
        self.crowd_monitor = CrowdDensityMonitor()
        self.alert_system = AlertSystem()
        self.is_running = False
        self.processing_threads = []
    
    def process_camera_feed(self, camera_id: str, source):
        """Process individual camera feed"""
        cap = cv2.VideoCapture(source)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Face detection
            faces = self.face_detector.detect_faces(frame)
            
            # Behavior analysis
            behavior_data = self.behavior_analyzer.analyze_behavior(frame)
            
            # Crowd density monitoring
            density_data = self.crowd_monitor.estimate_crowd_density(len(faces))
            
            # Generate alerts based on analysis
            self.evaluate_and_alert(camera_id, faces, behavior_data, density_data)
            
            # Optional: Display processed frame (for debugging)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            time.sleep(0.1)  # Prevent excessive CPU usage
        
        cap.release()
    
    def evaluate_and_alert(self, camera_id: str, faces: List, 
                          behavior_data: Dict, density_data: Dict):
        """Evaluate conditions and generate alerts if necessary"""
        
        # Check for suspicious behavior with enhanced details
        if behavior_data['is_anomalous']:
            description = (f"Anomalous behavior detected - "
                         f"Motion: {behavior_data['motion_ratio']:.3f}, "
                         f"Variance: {behavior_data['motion_variance']:.3f}")
            
            self.alert_system.generate_alert(
                "SUSPICIOUS_BEHAVIOR",
                f"{self.location} - Camera {camera_id}",
                min(0.95, 0.5 + behavior_data['motion_variance'] * 10),  # Dynamic confidence
                description
            )
        
        # Check for overcrowding
        if density_data['status'] == "OVERCROWDED":
            self.alert_system.generate_alert(
                "OVERCROWDING",
                f"{self.location} - Camera {camera_id}",
                0.9,
                f"Area overcrowded: {density_data['face_count']} people detected "
                f"({density_data['capacity_ratio']:.1%} capacity)"
            )
        elif density_data['status'] == "HIGH_DENSITY":
            # Only alert for high density every 30 seconds to avoid spam
            if not hasattr(self, '_last_density_alert'):
                self._last_density_alert = 0
            
            if time.time() - self._last_density_alert > 30:
                self.alert_system.generate_alert(
                    "HIGH_DENSITY",
                    f"{self.location} - Camera {camera_id}",
                    0.6,
                    f"High crowd density: {density_data['capacity_ratio']:.1%} capacity "
                    f"({density_data['face_count']} people)"
                )
                self._last_density_alert = time.time()
        
        # Real-time monitoring display
        if hasattr(self, '_last_status_update'):
            if time.time() - self._last_status_update > 5:  # Update every 5 seconds
                self._display_status(camera_id, faces, behavior_data, density_data)
                self._last_status_update = time.time()
        else:
            self._last_status_update = time.time()
    
    def _display_status(self, camera_id: str, faces: List, 
                       behavior_data: Dict, density_data: Dict):
        """Display real-time monitoring status"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        motion = behavior_data['motion_ratio']
        people_count = len(faces)
        status = density_data['status']
        
        print(f"[{timestamp}] {camera_id}: {people_count} people, "
              f"Motion: {motion:.3f}, Status: {status}")
        
        if behavior_data['is_anomalous']:
            print(f"  ⚠️  ANOMALY DETECTED - Variance: {behavior_data['motion_variance']:.3f}")
        
        if status in ["HIGH_DENSITY", "OVERCROWDED"]:
            print(f"  🚨 DENSITY ALERT - {density_data['capacity_ratio']:.1%} capacity")
    
    def start_monitoring(self):
        """Start the surveillance system"""
        self.is_running = True
        logger.info(f"Starting Urban Security System at {self.location}")
        
        for i, source in enumerate(self.camera_sources):
            camera_id = f"CAM_{i+1:02d}"
            thread = threading.Thread(
                target=self.process_camera_feed,
                args=(camera_id, source)
            )
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Monitoring {len(self.camera_sources)} camera feeds")
    
    def stop_monitoring(self):
        """Stop the surveillance system"""
        self.is_running = False
        logger.info("Stopping Urban Security System")
        
        for thread in self.processing_threads:
            thread.join()
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'active_cameras': len(self.camera_sources),
            'total_alerts': len(self.alert_system.alerts),
            'recent_alerts': [
                {
                    'type': alert.alert_type,
                    'priority': alert.priority,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alert_system.alerts[-5:]  # Last 5 alerts
            ]
        }

# Example usage and demonstration
def main():
    """Demonstrate the urban security system"""
    
    print("="*60)
    print("AI-DRIVEN URBAN SECURITY & SURVEILLANCE SYSTEM")
    print("="*60)
    print("Initializing system components...")
    
    # Initialize system with camera sources
    # Note: Use 0 for default webcam, or provide video file paths
    camera_sources = [0]  # Can add more sources: [0, 'camera2.mp4', 'rtsp://camera3']
    
    system = UrbanSecuritySystem(
        camera_sources=camera_sources,
        location="Downtown Plaza"
    )
    
    print(f"✓ Face Detection Module: Loaded")
    print(f"✓ Behavior Analysis Module: Loaded")
    print(f"✓ Crowd Density Monitor: Loaded")
    print(f"✓ Alert System: Loaded")
    print(f"✓ Data Encryption: Enabled")
    print(f"✓ Camera Sources: {len(camera_sources)} configured")
    print()
    
    try:
        # Start monitoring
        system.start_monitoring()
        
        # Run for demonstration (in practice, this would run continuously)
        print("🔴 SYSTEM ACTIVE - Monitoring in progress...")
        print("Press Ctrl+C to stop monitoring")
        print("-" * 60)
        
        # Show real-time status updates
        for i in range(30):  # Run for 30 seconds for demo
            time.sleep(1)
            if i % 5 == 0:  # Update every 5 seconds
                status = system.get_system_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Active: {status['active_cameras']} cameras | "
                      f"Alerts: {status['total_alerts']} total")
        
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("⚠️  SHUTDOWN INITIATED")
        print("="*60)
    finally:
        system.stop_monitoring()
        
        # Display final status
        status = system.get_system_status()
        print(f"📊 FINAL SYSTEM REPORT:")
        print(f"   Total Alerts Generated: {status['total_alerts']}")
        print(f"   Recent Alerts: {len(status['recent_alerts'])}")
        
        if status['recent_alerts']:
            print(f"   Latest Alert Types:")
            for alert in status['recent_alerts'][-3:]:
                print(f"   - {alert['type']} ({alert['priority']}) at {alert['timestamp'][:19]}")
        
        print(f"✓ System shutdown complete")
        print("="*60)

if __name__ == "__main__":
    main()