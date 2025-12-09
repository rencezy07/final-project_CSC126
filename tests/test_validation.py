"""
Quick validation tests for Aerial Threat Detection System
Tests basic functionality without requiring heavy dependencies
"""

import unittest
import os
import sys
from pathlib import Path


class TestProjectStructure(unittest.TestCase):
    """Test project structure and file existence"""
    
    def setUp(self):
        self.project_root = Path(__file__).parent.parent
    
    def test_source_files_exist(self):
        """Test that core source files exist"""
        files = [
            'src/aerial_threat_detector.py',
            'src/detection_server.py',
            'src/utils/evaluation_utils.py',
            'src/__init__.py',
            'src/utils/__init__.py'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")
    
    def test_script_files_exist(self):
        """Test that training/preparation scripts exist"""
        files = [
            'scripts/prepare_dataset.py',
            'scripts/train_model.py',
            'scripts/__init__.py'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")
    
    def test_electron_app_files_exist(self):
        """Test that Electron app files exist"""
        files = [
            'electron-app/main.js',
            'electron-app/renderer.js',
            'electron-app/index.html',
            'electron-app/styles.css',
            'electron-app/package.json'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")
    
    def test_documentation_exists(self):
        """Test that documentation files exist"""
        files = [
            'README.md',
            'requirements.txt',
            'docs/Technical_Report.md',
            'docs/ETHICAL_CONSIDERATIONS.md',
            'docs/TRAINING_GUIDE.md',
            'docs/FINAL_PRESENTATION.md'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")
    
    def test_batch_files_exist(self):
        """Test that startup scripts exist"""
        files = [
            'start_app.bat',
            'launch.bat'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")


class TestPythonSyntax(unittest.TestCase):
    """Test Python files for syntax errors"""
    
    def setUp(self):
        self.project_root = Path(__file__).parent.parent
    
    def test_source_syntax(self):
        """Test that source files have valid Python syntax"""
        files = [
            'src/aerial_threat_detector.py',
            'src/detection_server.py',
            'src/utils/evaluation_utils.py'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            with open(full_path, 'r') as f:
                code = f.read()
                try:
                    compile(code, file_path, 'exec')
                except SyntaxError as e:
                    self.fail(f"Syntax error in {file_path}: {e}")
    
    def test_script_syntax(self):
        """Test that script files have valid Python syntax"""
        files = [
            'scripts/prepare_dataset.py',
            'scripts/train_model.py'
        ]
        
        for file_path in files:
            full_path = self.project_root / file_path
            with open(full_path, 'r') as f:
                code = f.read()
                try:
                    compile(code, file_path, 'exec')
                except SyntaxError as e:
                    self.fail(f"Syntax error in {file_path}: {e}")


class TestDocumentation(unittest.TestCase):
    """Test documentation completeness"""
    
    def setUp(self):
        self.project_root = Path(__file__).parent.parent
    
    def test_readme_content(self):
        """Test that README has essential content"""
        readme_path = self.project_root / 'README.md'
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check for essential sections
        essential_sections = [
            'Project Overview',
            'Quick Start',
            'Installation',
            'Usage',
            'Requirements'
        ]
        
        for section in essential_sections:
            self.assertIn(section, content, f"README missing section: {section}")
    
    def test_requirements_file(self):
        """Test that requirements.txt has essential packages"""
        req_path = self.project_root / 'requirements.txt'
        with open(req_path, 'r') as f:
            content = f.read()
        
        essential_packages = [
            'torch',
            'ultralytics',
            'opencv-python',
            'flask',
            'flask-socketio'
        ]
        
        for package in essential_packages:
            self.assertIn(package, content, f"Missing package in requirements: {package}")
    
    def test_training_guide_exists(self):
        """Test that training guide has content"""
        guide_path = self.project_root / 'docs' / 'TRAINING_GUIDE.md'
        with open(guide_path, 'r') as f:
            content = f.read()
        
        # Should have substantial content
        self.assertGreater(len(content), 5000, "Training guide seems incomplete")
        
        # Check for key sections
        key_sections = [
            'Prerequisites',
            'Dataset Preparation',
            'Model Training',
            'Evaluation'
        ]
        
        for section in key_sections:
            self.assertIn(section, content, f"Training guide missing: {section}")
    
    def test_ethical_considerations(self):
        """Test that ethical considerations document exists and has content"""
        ethics_path = self.project_root / 'docs' / 'ETHICAL_CONSIDERATIONS.md'
        with open(ethics_path, 'r') as f:
            content = f.read()
        
        # Should have substantial content
        self.assertGreater(len(content), 5000, "Ethical considerations seems incomplete")
        
        # Check for key ethical concepts
        key_concepts = [
            'ethical',
            'human rights',
            'transparency',
            'accountability',
            'prohibited'
        ]
        
        for concept in key_concepts:
            self.assertIn(concept.lower(), content.lower(), 
                         f"Ethical doc should discuss: {concept}")
    
    def test_final_presentation(self):
        """Test that final presentation exists and is comprehensive"""
        pres_path = self.project_root / 'docs' / 'FINAL_PRESENTATION.md'
        with open(pres_path, 'r') as f:
            content = f.read()
        
        # Should be comprehensive
        self.assertGreater(len(content), 20000, "Presentation seems incomplete")
        
        # Check for key sections
        key_sections = [
            'Model Design',
            'Performance Evaluation',
            'Deployment Recommendations',
            'Ethical Considerations',
            'Conclusion'
        ]
        
        for section in key_sections:
            self.assertIn(section, content, f"Presentation missing: {section}")


def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION TESTS")
    print("=" * 80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProjectStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestPythonSyntax))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentation))
    
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
