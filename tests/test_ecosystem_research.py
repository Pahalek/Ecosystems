"""
Tests for the Financial Ecosystem Research Environment
====================================================
"""

import unittest
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'visualization'))

from research_agent import FinancialEcosystemAgent, ResearchContext
from three_phase_model import ThreePhaseModel, EcosystemStakeholder, StakeholderType
from financial_data import FinancialDataCollector
from planner import VisualizationPlanner


class TestResearchAgent(unittest.TestCase):
    """Test the research agent functionality"""
    
    def setUp(self):
        self.context = ResearchContext(
            topic="Test Financial Inclusion",
            scope="Test scope",
            methodology="Test methodology",
            data_sources=["Test source"],
            timeline="2024"
        )
        self.agent = FinancialEcosystemAgent(self.context)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.context.topic, "Test Financial Inclusion")
        self.assertIsNotNone(self.agent.research_network)
        
    def test_chapter3_analysis(self):
        """Test Chapter 3 structure analysis"""
        analysis = self.agent.analyze_chapter3_structure()
        
        self.assertIn('structure_analysis', analysis)
        self.assertIn('research_gaps', analysis)
        self.assertIn('theoretical_analysis', analysis)
        self.assertIn('novel_aspects', analysis)
        
    def test_network_analysis(self):
        """Test network analysis functionality"""
        network_analysis = self.agent.conduct_network_analysis(["test_source"])
        
        self.assertIn('network_data', network_analysis)
        self.assertIn('metrics', network_analysis)
        self.assertIn('centrality', network_analysis)
        
    def test_expert_report_generation(self):
        """Test expert report generation"""
        report = self.agent.generate_expert_analysis_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("Expert Analysis Report", report)
        self.assertIn("Financial Inclusion", report)


class TestThreePhaseModel(unittest.TestCase):
    """Test the three-phase model functionality"""
    
    def setUp(self):
        self.model = ThreePhaseModel("Test Ecosystem")
        
        # Add test stakeholder
        stakeholder = EcosystemStakeholder(
            name="Test Corp",
            type=StakeholderType.CORE_MNC,
            financial_capacity=1000000,
            trust_score=0.8,
            digital_readiness=0.7
        )
        self.model.add_stakeholder(stakeholder)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.ecosystem_name, "Test Ecosystem")
        self.assertEqual(len(self.model.stakeholders), 1)
        
    def test_phase_execution_sequence(self):
        """Test phases execute in correct sequence"""
        # Phase 1
        phase_1 = self.model.execute_phase_1_define()
        self.assertEqual(phase_1['phase'], 'DEFINE')
        
        # Phase 2
        phase_2 = self.model.execute_phase_2_design()
        self.assertEqual(phase_2['phase'], 'DESIGN')
        
        # Phase 3
        phase_3 = self.model.execute_phase_3_build()
        self.assertEqual(phase_3['phase'], 'BUILD')
        
    def test_ecosystem_status(self):
        """Test ecosystem status reporting"""
        # Execute all phases first
        self.model.execute_phase_1_define()
        self.model.execute_phase_2_design()
        self.model.execute_phase_3_build()
        
        status = self.model.get_ecosystem_status()
        
        self.assertIn('ecosystem_name', status)
        self.assertIn('current_phase', status)
        self.assertIn('stakeholder_summary', status)


class TestFinancialDataCollector(unittest.TestCase):
    """Test financial data collection functionality"""
    
    def setUp(self):
        # Use temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_db.sqlite')
        self.collector = FinancialDataCollector(self.db_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Test collector initializes correctly"""
        self.assertIsNotNone(self.collector.data_sources)
        self.assertTrue(os.path.exists(self.db_path))
        
    def test_economic_indicators_collection(self):
        """Test economic indicators collection"""
        countries = ['US', 'EU']
        indicators = ['GDP_growth', 'inflation_rate']
        
        data = self.collector.collect_economic_indicators(countries, indicators)
        
        self.assertGreater(len(data), 0)
        self.assertIn('country', data.columns)
        self.assertIn('indicator', data.columns)
        
    def test_financial_inclusion_metrics(self):
        """Test financial inclusion metrics collection"""
        regions = ['North America', 'Europe']
        
        data = self.collector.collect_financial_inclusion_metrics(regions)
        
        self.assertGreater(len(data), 0)
        self.assertIn('region', data.columns)
        self.assertIn('metric', data.columns)


class TestVisualizationPlanner(unittest.TestCase):
    """Test visualization planning functionality"""
    
    def setUp(self):
        self.planner = VisualizationPlanner()
    
    def test_planner_initialization(self):
        """Test planner initializes correctly"""
        self.assertIsNotNone(self.planner.academic_standards)
        
    def test_chapter3_visualization_plan(self):
        """Test Chapter 3 visualization plan creation"""
        plan = self.planner.create_chapter3_visualization_plan()
        
        self.assertIn('section_3_1', plan)
        self.assertIn('theoretical_framework', plan)
        self.assertIn('empirical_analysis', plan)
        
    def test_visualization_plan_export(self):
        """Test visualization plan export"""
        # Create plan first
        plan = self.planner.create_chapter3_visualization_plan()
        
        # Export plan
        export_path = self.planner.export_visualization_plan()
        
        self.assertTrue(os.path.exists(export_path))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_system_integration(self):
        """Test that all components work together"""
        # Initialize components
        context = ResearchContext(
            topic="Integration Test",
            scope="Test scope",
            methodology="Test methodology",
            data_sources=["Test source"],
            timeline="2024"
        )
        
        agent = FinancialEcosystemAgent(context)
        model = ThreePhaseModel("Integration Test Ecosystem")
        
        # Add stakeholder to model
        stakeholder = EcosystemStakeholder(
            name="Integration Test Corp",
            type=StakeholderType.CORE_MNC,
            financial_capacity=1000000,
            trust_score=0.8,
            digital_readiness=0.7
        )
        model.add_stakeholder(stakeholder)
        
        # Test that components can work together
        agent_analysis = agent.analyze_chapter3_structure()
        model_phase_1 = model.execute_phase_1_define()
        
        self.assertIsNotNone(agent_analysis)
        self.assertIsNotNone(model_phase_1)


if __name__ == '__main__':
    # Create test directories if they don't exist
    os.makedirs('/home/runner/work/Ecosystems/Ecosystems/reports', exist_ok=True)
    os.makedirs('/home/runner/work/Ecosystems/Ecosystems/data', exist_ok=True)
    os.makedirs('/home/runner/work/Ecosystems/Ecosystems/visualizations', exist_ok=True)
    os.makedirs('/home/runner/work/Ecosystems/Ecosystems/config', exist_ok=True)
    
    unittest.main(verbosity=2)