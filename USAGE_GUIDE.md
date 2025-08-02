# Financial Ecosystem Research Usage Guide

## Quick Start

### 1. Run Complete Analysis
```bash
python src/main.py --mode full
```
This executes all research phases and generates comprehensive results.

### 2. Individual Component Testing
```bash
# Research agent only
python src/main.py --mode agent

# Three-phase model only  
python src/main.py --mode model

# Data collection only
python src/main.py --mode data

# Visualization planning only
python src/main.py --mode viz
```

### 3. Direct Module Usage

#### Research Agent
```python
from src.agents.research_agent import FinancialEcosystemAgent, ResearchContext

context = ResearchContext(
    topic="Financial Inclusion in Business Ecosystems",
    scope="Chapter 3 Analysis", 
    methodology="Mixed methods",
    data_sources=["Academic DB", "Financial APIs"],
    timeline="2024"
)

agent = FinancialEcosystemAgent(context)
report = agent.generate_expert_analysis_report()
```

#### Three-Phase Model
```python
from src.models.three_phase_model import ThreePhaseModel, create_sample_ecosystem

ecosystem = create_sample_ecosystem()
phase_1 = ecosystem.execute_phase_1_define()
phase_2 = ecosystem.execute_phase_2_design() 
phase_3 = ecosystem.execute_phase_3_build()
```

#### Visualization Generation
```python
from src.visualization.planner import VisualizationPlanner

planner = VisualizationPlanner()
plan = planner.create_chapter3_visualization_plan()
samples = planner.generate_sample_visualizations()
```

## Output Files

### Reports
- `reports/comprehensive_research_analysis.md` - Expert analysis report
- `reports/chapter3_detailed_outline.md` - Detailed chapter structure
- `reports/comprehensive_results.json` - Complete analysis results

### Data
- `data/three_phase_results.json` - Three-phase model outputs
- `data/financial_analysis_results.json` - Financial data analysis
- `data/ecosystem_data.db` - SQLite database with collected data

### Visualizations
- `visualizations/three_phase_diagram.png` - Three-phase process diagram
- `visualizations/ecosystem_network.png` - Network visualization  
- `visualizations/integration_heatmap.png` - Stakeholder integration matrix
- `visualizations/theoretical_comparison.png` - Theoretical framework comparison

### Configuration
- `config/visualization_plan.json` - Complete visualization planning
- `.vscode/` - VS Code copilot configuration

## Key Features

### Research Agent Capabilities
- ✅ Network analysis for research gap identification
- ✅ Novel research direction generation (3 directions identified)
- ✅ Theoretical framework extension (Moore 1993, Adner 2017)
- ✅ Expert-level analysis comparable to McKinsey standards

### Three-Phase Implementation
- ✅ **DEFINE**: Financial strategy, stakeholder analysis, risk assessment
- ✅ **DESIGN**: Architecture planning, API framework, integration matrix
- ✅ **BUILD**: Operational systems, investment vehicles, monitoring

### Data Analysis Engine
- ✅ MNC financial data collection (real or sample)
- ✅ Economic indicators across multiple countries
- ✅ Financial inclusion metrics by region
- ✅ Ecosystem health scoring and monitoring

### Academic Standards
- ✅ Top-tier journal visualization standards
- ✅ Consulting-quality analysis frameworks
- ✅ Dissertation-ready research structure
- ✅ Reproducible methodology

## VS Code Integration

The project includes full VS Code copilot configuration:

1. **Settings**: Optimized for Python research and analysis
2. **Extensions**: Recommended extensions for data science and research
3. **Launch configs**: Pre-configured debug sessions for each component
4. **Tasks**: Automated workflows for testing and analysis

## Dependencies

### Core Requirements
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
```

### Optional Enhancements
```
yfinance>=0.2.0  # For real financial data
plotly>=5.15.0   # For interactive visualizations
jupyter>=1.0.0   # For notebook analysis
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Or run individual module tests:
```bash
python tests/test_ecosystem_research.py
```

## Academic Applications

This research environment supports:

1. **Dissertation Research**: Complete Chapter 3 framework and analysis
2. **Academic Publications**: Novel theoretical contributions identified
3. **Consulting Projects**: McKinsey-level analysis capabilities  
4. **Teaching**: Comprehensive case study material
5. **Policy Research**: Real-world financial inclusion insights

## Next Steps

1. **Enhance Data Sources**: Configure real financial APIs
2. **Expand Visualizations**: Generate additional academic charts
3. **Academic Writing**: Use research outputs for dissertation writing
4. **Real-world Testing**: Apply to actual MNC ecosystem case studies
5. **Publication Preparation**: Develop papers from novel research directions