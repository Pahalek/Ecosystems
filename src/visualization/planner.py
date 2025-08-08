"""
Visualization Planning System for Financial Ecosystem Research
============================================================

This module creates comprehensive visualization plans for academic research
on financial inclusion in business ecosystems, following leading academic
and consulting visualization standards.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Optional plotly import
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class VisualizationSpec:
    """Specification for a visualization"""
    title: str
    type: str
    purpose: str
    data_requirements: List[str]
    academic_standards: List[str]
    complexity_level: str


class VisualizationPlanner:
    """
    Comprehensive visualization planning system for Chapter 3 research
    """
    
    def __init__(self):
        self.visualization_catalog = {}
        self.chapter3_plan = {}
        self.academic_standards = self._define_academic_standards()
    
    def _define_academic_standards(self) -> Dict[str, List[str]]:
        """Define academic visualization standards"""
        return {
            "top_tier_journals": [
                "Clear, professional typography",
                "Colorblind-friendly palettes",
                "High resolution (300+ DPI)",
                "Consistent styling across figures",
                "Comprehensive captions"
            ],
            "consulting_standards": [
                "Executive summary visuals",
                "Action-oriented insights",
                "Clear value proposition",
                "Stakeholder-specific views",
                "Implementation roadmaps"
            ],
            "dissertation_requirements": [
                "Theoretical framework diagrams",
                "Empirical evidence presentation",
                "Methodological clarity",
                "Reproducible results",
                "Academic citation integration"
            ]
        }
    
    def create_chapter3_visualization_plan(self) -> Dict[str, Any]:
        """
        Create comprehensive visualization plan for Chapter 3:
        "Шляхи формування і структурна композиція фінансів екосистем багатонаціонального бізнесу"
        """
        
        plan = {
            "section_3_1": self._plan_section_3_1_visuals(),
            "theoretical_framework": self._plan_theoretical_visuals(),
            "empirical_analysis": self._plan_empirical_visuals(),
            "implementation_roadmap": self._plan_implementation_visuals()
        }
        
        self.chapter3_plan = plan
        return plan
    
    def _plan_section_3_1_visuals(self) -> List[VisualizationSpec]:
        """Plan visualizations for Section 3.1"""
        
        return [
            VisualizationSpec(
                title="Three-Phase Ecosystem Financial Base Formation",
                type="process_diagram",
                purpose="Illustrate the Define→Design→Build progression",
                data_requirements=["Phase timelines", "Resource allocation", "Milestone markers"],
                academic_standards=["Process clarity", "Temporal visualization", "Resource flow"],
                complexity_level="Medium"
            ),
            VisualizationSpec(
                title="Concurrent Business Ecosystem Creation Model",
                type="network_diagram",
                purpose="Show simultaneous ecosystem development processes",
                data_requirements=["Process dependencies", "Resource sharing", "Timeline overlap"],
                academic_standards=["Network visualization", "Temporal dimension", "Complexity handling"],
                complexity_level="High"
            ),
            VisualizationSpec(
                title="Financial Architecture Evolution",
                type="evolutionary_diagram",
                purpose="Demonstrate financial structure development across phases",
                data_requirements=["Financial components", "Integration points", "Evolution stages"],
                academic_standards=["Evolutionary visualization", "System complexity", "Clear progression"],
                complexity_level="High"
            ),
            VisualizationSpec(
                title="Stakeholder Financial Integration Matrix",
                type="heatmap_matrix",
                purpose="Show financial integration levels between stakeholders",
                data_requirements=["Stakeholder types", "Integration metrics", "Financial flows"],
                academic_standards=["Matrix visualization", "Quantitative display", "Relationship clarity"],
                complexity_level="Medium"
            )
        ]
    
    def _plan_theoretical_visuals(self) -> List[VisualizationSpec]:
        """Plan theoretical framework visualizations"""
        
        return [
            VisualizationSpec(
                title="Moore (1993) vs. Modern Digital Ecosystem Evolution",
                type="comparative_framework",
                purpose="Compare traditional and digital ecosystem evolution",
                data_requirements=["Theoretical stages", "Digital adaptations", "Time comparisons"],
                academic_standards=["Theoretical comparison", "Clear differentiation", "Academic rigor"],
                complexity_level="Medium"
            ),
            VisualizationSpec(
                title="Adner (2017) Structure Applied to Financial Ecosystems",
                type="structural_diagram",
                purpose="Apply Adner's ecosystem structure to financial context",
                data_requirements=["Ecosystem components", "Structural relationships", "Financial flows"],
                academic_standards=["Theoretical application", "Structural clarity", "Financial focus"],
                complexity_level="High"
            ),
            VisualizationSpec(
                title="Integrated Theoretical Framework",
                type="conceptual_model",
                purpose="Present novel integrated theoretical framework",
                data_requirements=["Theory components", "Integration points", "Novel contributions"],
                academic_standards=["Theoretical novelty", "Integration clarity", "Academic contribution"],
                complexity_level="High"
            )
        ]
    
    def _plan_empirical_visuals(self) -> List[VisualizationSpec]:
        """Plan empirical analysis visualizations"""
        
        return [
            VisualizationSpec(
                title="Financial Inclusion Metrics Across Ecosystems",
                type="dashboard",
                purpose="Present comprehensive financial inclusion measurements",
                data_requirements=["Inclusion metrics", "Ecosystem types", "Performance data"],
                academic_standards=["Quantitative rigor", "Comparative analysis", "Clear metrics"],
                complexity_level="Medium"
            ),
            VisualizationSpec(
                title="Cross-Cultural Financial Ecosystem Comparison",
                type="comparative_analysis",
                purpose="Compare financial ecosystems across different cultures",
                data_requirements=["Cultural dimensions", "Financial metrics", "Regional data"],
                academic_standards=["Cross-cultural validity", "Comparative methodology", "Cultural sensitivity"],
                complexity_level="High"
            ),
            VisualizationSpec(
                title="Real-time Ecosystem Health Monitor",
                type="dynamic_dashboard",
                purpose="Show real-time ecosystem financial health indicators",
                data_requirements=["Real-time data feeds", "Health metrics", "Alert systems"],
                academic_standards=["Real-time visualization", "Health indicators", "Actionable insights"],
                complexity_level="High"
            )
        ]
    
    def _plan_implementation_visuals(self) -> List[VisualizationSpec]:
        """Plan implementation and practical visualizations"""
        
        return [
            VisualizationSpec(
                title="Implementation Roadmap for MNCs",
                type="gantt_roadmap",
                purpose="Provide practical implementation timeline",
                data_requirements=["Implementation phases", "Resource requirements", "Dependencies"],
                academic_standards=["Practical applicability", "Clear timelines", "Resource planning"],
                complexity_level="Medium"
            ),
            VisualizationSpec(
                title="Risk-Return Matrix for Ecosystem Phases",
                type="scatter_matrix",
                purpose="Show risk-return profiles across ecosystem phases",
                data_requirements=["Risk metrics", "Return measures", "Phase classifications"],
                academic_standards=["Risk-return analysis", "Financial visualization", "Decision support"],
                complexity_level="Medium"
            )
        ]
    
    def generate_sample_visualizations(self) -> Dict[str, Any]:
        """Generate sample visualizations to demonstrate capabilities"""
        
        samples = {}
        
        # 1. Three-Phase Process Diagram
        samples['three_phase_process'] = self._create_three_phase_diagram()
        
        # 2. Network Analysis Visualization
        samples['ecosystem_network'] = self._create_ecosystem_network()
        
        # 3. Financial Integration Heatmap
        samples['integration_heatmap'] = self._create_integration_heatmap()
        
        # 4. Comparative Framework
        samples['theoretical_comparison'] = self._create_theoretical_comparison()
        
        return samples
    
    def _create_three_phase_diagram(self) -> str:
        """Create three-phase ecosystem formation diagram"""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Define phases
        phases = THREE_PHASE_DIAGRAM_PHASES
        phase_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Create phase boxes
        for i, (phase, color) in enumerate(zip(phases, phase_colors)):
            rect = plt.Rectangle((i*3, 0), 2.5, 4, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i*3 + 1.25, 2, phase, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add arrows between phases
        for i in range(len(phases)-1):
            ax.arrow(i*3 + 2.6, 2, 0.3, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        # Add sub-components
        components = [
            ['Ідентифікація трендів', 'Формування резервів', 'Механізми партнерства'],
            ['Розподіл ризиків', 'Взаємне кредитування', 'Цифрові інтерфейси'],
            ['Платіжні системи', 'Інвестиційні фонди', 'Моніторинг']
        ]
        
        for i, component_list in enumerate(components):
            for j, component in enumerate(component_list):
                ax.text(i*3 + 1.25, -0.5 - j*0.4, f"• {component}", ha='center', va='center', fontsize=9)
        
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-2.5, 5)
        ax.set_title('Поетапне формування фінансової бази бізнес-екосистем БНК', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = '/home/runner/work/Ecosystems/Ecosystems/visualizations/three_phase_diagram.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_ecosystem_network(self) -> str:
        """Create ecosystem network visualization"""
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with different types
        financial_nodes = ['Центральний банк', 'Комерційні банки', 'Фінтех', 'Страхові']
        corporate_nodes = ['БНК-ядро', 'Партнери', 'Постачальники', 'Дистрибютори']
        tech_nodes = ['Платформи', 'API', 'Дані', 'Аналітика']
        
        all_nodes = financial_nodes + corporate_nodes + tech_nodes
        node_types = ['financial'] * len(financial_nodes) + ['corporate'] * len(corporate_nodes) + ['tech'] * len(tech_nodes)
        
        for node, node_type in zip(all_nodes, node_types):
            G.add_node(node, type=node_type)
        
        # Add edges (simplified network)
        edges = [
            ('БНК-ядро', 'Центральний банк'), ('БНК-ядро', 'Комерційні банки'),
            ('БНК-ядро', 'Партнери'), ('БНК-ядро', 'Платформи'),
            ('Комерційні банки', 'Фінтех'), ('Фінтех', 'Платформи'),
            ('Партнери', 'Постачальники'), ('Партнери', 'Дистрибютори'),
            ('Платформи', 'API'), ('API', 'Дані'), ('Дані', 'Аналітика'),
            ('Страхові', 'Комерційні банки'), ('Аналітика', 'БНК-ядро')
        ]
        
        G.add_edges_from(edges)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Define positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Define colors for different node types
        color_map = {'financial': '#FF6B6B', 'corporate': '#4ECDC4', 'tech': '#45B7D1'}
        node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Мережа фінансової екосистеми БНК', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=15, label='Фінансові інститути'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=15, label='Корпоративні учасники'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', markersize=15, label='Технологічні компоненти')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        output_path = '/home/runner/work/Ecosystems/Ecosystems/visualizations/ecosystem_network.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_integration_heatmap(self) -> str:
        """Create financial integration heatmap"""
        
        # Create sample data
        stakeholders = ['Центральний банк', 'Комерційні банки', 'Фінтех', 'БНК-ядро', 'Партнери', 'Регулятори']
        integration_data = np.random.uniform(0.2, 1.0, (len(stakeholders), len(stakeholders)))
        
        # Make symmetric and set diagonal to 1
        integration_data = (integration_data + integration_data.T) / 2
        np.fill_diagonal(integration_data, 1.0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(integration_data, 
                   xticklabels=stakeholders, 
                   yticklabels=stakeholders,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Рівень фінансової інтеграції'})
        
        plt.title('Матриця фінансової інтеграції стейкхолдерів', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        output_path = '/home/runner/work/Ecosystems/Ecosystems/visualizations/integration_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_theoretical_comparison(self) -> str:
        """Create theoretical framework comparison"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Moore (1993) Traditional Model
        ax1.set_title('Moore (1993) - Традиційна модель', fontsize=14, fontweight='bold')
        
        # Ecosystem stages
        stages = ['Зародження', 'Розширення', 'Влада', 'Оновлення']
        stage_colors = ['#FFE5B4', '#FFD700', '#FFA500', '#FF6B35']
        
        for i, (stage, color) in enumerate(zip(stages, stage_colors)):
            circle = plt.Circle((0.5, 0.8 - i*0.2), 0.15, facecolor=color, alpha=0.7, edgecolor='black')
            ax1.add_patch(circle)
            ax1.text(0.7, 0.8 - i*0.2, stage, va='center', fontsize=11)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Digital Ecosystem Model
        ax2.set_title('Цифрова екосистема (2024)', fontsize=14, fontweight='bold')
        
        digital_stages = ['Цифрова стратегія', 'Платформна архітектура', 'API інтеграція', 'AI оптимізація']
        digital_colors = ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2']
        
        for i, (stage, color) in enumerate(zip(digital_stages, digital_colors)):
            circle = plt.Circle((0.5, 0.8 - i*0.2), 0.15, facecolor=color, alpha=0.7, edgecolor='black')
            ax2.add_patch(circle)
            ax2.text(0.7, 0.8 - i*0.2, stage, va='center', fontsize=11)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.suptitle('Порівняння теоретичних моделей екосистемного розвитку', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = '/home/runner/work/Ecosystems/Ecosystems/visualizations/theoretical_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def export_visualization_plan(self) -> str:
        """Export complete visualization plan as JSON"""
        
        plan_data = {
            "chapter3_visualization_plan": self.chapter3_plan,
            "academic_standards": self.academic_standards,
            "generated_timestamp": datetime.now().isoformat(),
            "total_visualizations": sum(len(section) for section in self.chapter3_plan.values() if isinstance(section, list))
        }
        
        output_path = '/home/runner/work/Ecosystems/Ecosystems/config/visualization_plan.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2, default=str)
        
        return output_path


def main():
    """Main function to demonstrate visualization planning"""
    
    planner = VisualizationPlanner()
    
    # Create visualization plan
    plan = planner.create_chapter3_visualization_plan()
    
    # Generate sample visualizations
    samples = planner.generate_sample_visualizations()
    
    # Export plan
    plan_file = planner.export_visualization_plan()
    
    print("Visualization planning completed!")
    print(f"Plan exported to: {plan_file}")
    print(f"Sample visualizations created: {len(samples)}")
    
    for name, path in samples.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()