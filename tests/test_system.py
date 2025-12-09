"""
Test Suite for Aerial Threat Detection System
Tests core functionality of detection, server, and utilities
"""

import unittest
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aerial_threat_detector import AerialThreatDetector
from utils.evaluation_utils import EvaluationMetrics, calculate_iou, non_max_suppression


class TestAerialThreatDetector(unittest.TestCase):
    """Test cases for AerialThreatDetector class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_image_path = 'test_data/test_image.jpg'
        cls.model_path = 'best.pt'
        
        # Create test image if it doesn't exist
        os.makedirs('test_data', exist_ok=True)
        if not os.path.exists(cls.test_image_path):
            # Create a simple test image
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(cls.test_image_path, test_img)
    
    def test_detector_initialization(self):
        """Test detector initialization without model file"""
        # Test with non-existent model - should handle gracefully
        detector = AerialThreatDetector('nonexistent_model.pt', confidence_threshold=0.5)
        self.assertIsNone(detector.model)
    
    def test_confidence_threshold(self):
        """Test confidence threshold setting"""
        detector = AerialThreatDetector('test_model.pt', confidence_threshold=0.7)
        self.assertEqual(detector.confidence_threshold, 0.7)
    
    def test_color_setup(self):
        """Test color setup for classes"""
        detector = AerialThreatDetector('test_model.pt')
        detector.class_names = ['soldier', 'civilian']
        detector._setup_colors()
        
        # Check that colors are assigned
        self.assertIn('soldier', detector.colors)
        self.assertIn('civilian', detector.colors)
        
        # Soldier should be red
        self.assertEqual(detector.colors['soldier'], (0, 0, 255))
        # Civilian should be green
        self.assertEqual(detector.colors['civilian'], (0, 255, 0))
    
    def test_detection_stats(self):
        """Test detection statistics calculation"""
        detector = AerialThreatDetector('test_model.pt')
        
        # Create mock detections
        detections = [
            {'class_name': 'soldier', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]},
            {'class_name': 'soldier', 'confidence': 0.85, 'bbox': [60, 60, 100, 100]},
            {'class_name': 'civilian', 'confidence': 0.75, 'bbox': [120, 120, 160, 160]},
        ]
        
        stats = detector.get_detection_stats(detections)
        
        self.assertEqual(stats['total_detections'], 3)
        self.assertEqual(stats['class_counts']['soldier'], 2)
        self.assertEqual(stats['class_counts']['civilian'], 1)
        self.assertAlmostEqual(stats['avg_confidence'], 0.833, places=2)
        self.assertEqual(stats['high_confidence_count'], 2)  # 0.9 and 0.85 are > 0.8


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for EvaluationMetrics class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = EvaluationMetrics()
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        self.assertEqual(len(self.evaluator.true_positives), 0)
        self.assertEqual(len(self.evaluator.false_positives), 0)
        self.assertEqual(len(self.evaluator.false_negatives), 0)
    
    def test_add_detection(self):
        """Test adding detection results"""
        self.evaluator.add_detection('soldier', 'soldier', 0.9, 0.7)
        
        self.assertEqual(self.evaluator.true_positives['soldier'], 1)
        self.assertIn(0.9, self.evaluator.confidences['soldier'])
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Add some test detections
        self.evaluator.add_detection('soldier', 'soldier', 0.9, 0.7)  # TP
        self.evaluator.add_detection('soldier', 'soldier', 0.85, 0.6)  # TP
        self.evaluator.add_detection('soldier', 'civilian', 0.75, 0.6)  # FP
        self.evaluator.add_detection('civilian', 'civilian', 0.8, 0.7)  # TP
        
        metrics = self.evaluator.calculate_metrics()
        
        # Check soldier metrics
        self.assertIn('soldier', metrics)
        self.assertEqual(metrics['soldier']['true_positives'], 2)
        self.assertEqual(metrics['soldier']['false_positives'], 1)
        
        # Check that precision/recall are calculated
        self.assertGreater(metrics['soldier']['precision'], 0)
        self.assertGreater(metrics['soldier']['recall'], 0)
    
    def test_reset(self):
        """Test metrics reset"""
        self.evaluator.add_detection('soldier', 'soldier', 0.9, 0.7)
        self.evaluator.reset()
        
        self.assertEqual(len(self.evaluator.true_positives), 0)
        self.assertEqual(len(self.evaluator.all_classes), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        
        iou = calculate_iou(box1, box2)
        
        # Expected IoU for these overlapping boxes
        # Intersection area: 50x50 = 2500
        # Union area: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.1428...
        self.assertAlmostEqual(iou, 0.1428, places=3)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap"""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU with perfect overlap"""
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 1.0)
    
    def test_non_max_suppression(self):
        """Test non-maximum suppression"""
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class_name': 'soldier'},
            {'bbox': [15, 15, 55, 55], 'confidence': 0.7, 'class_name': 'soldier'},  # Overlaps with first
            {'bbox': [100, 100, 150, 150], 'confidence': 0.85, 'class_name': 'civilian'},
        ]
        
        filtered = non_max_suppression(detections, iou_threshold=0.5)
        
        # Should keep the highest confidence detection from overlapping boxes
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['confidence'], 0.9)
        self.assertEqual(filtered[1]['confidence'], 0.85)
    
    def test_non_max_suppression_empty(self):
        """Test NMS with empty input"""
        filtered = non_max_suppression([])
        self.assertEqual(len(filtered), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the system"""
    
    def test_end_to_end_detection(self):
        """Test end-to-end detection pipeline (without actual model)"""
        # This test validates the structure without requiring a trained model
        detector = AerialThreatDetector('nonexistent.pt', confidence_threshold=0.5)
        
        # Create a test frame
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Should handle gracefully when model is not loaded
        annotated_frame, detections = detector.detect_frame(test_frame)
        
        self.assertIsNotNone(annotated_frame)
        self.assertEqual(len(detections), 0)  # No model, so no detections
    
    def test_evaluation_pipeline(self):
        """Test evaluation metrics pipeline"""
        evaluator = EvaluationMetrics()
        
        # Simulate a detection pipeline
        test_results = [
            ('soldier', 'soldier', 0.95, 0.8),
            ('soldier', 'soldier', 0.88, 0.75),
            ('civilian', 'civilian', 0.92, 0.85),
            ('soldier', 'civilian', 0.65, 0.6),  # False positive
        ]
        
        for pred_class, true_class, conf, iou in test_results:
            evaluator.add_detection(pred_class, true_class, conf, iou)
        
        metrics = evaluator.calculate_metrics()
        
        # Verify basic metric structure
        self.assertIn('soldier', metrics)
        self.assertIn('civilian', metrics)
        self.assertIn('precision', metrics['soldier'])
        self.assertIn('recall', metrics['soldier'])
        self.assertIn('f1_score', metrics['soldier'])


def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING AERIAL THREAT DETECTION SYSTEM TESTS")
    print("=" * 80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAerialThreatDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
