"""
Three-Phase Financial Ecosystem Model Implementation
==================================================

Implementation of the three-phase model for financial base creation
in multinational business ecosystems:
- Phase 1: Define (Визначення фінансової стратегії)
- Phase 2: Design (Проектування фінансової архітектури)  
- Phase 3: Build (Операційна інтеграція)

Based on Moore (1993), Adner (2017), and contemporary digital ecosystem theories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta


class EcosystemPhase(Enum):
    """Phases of ecosystem financial base development"""
    DEFINE = "define"
    DESIGN = "design"
    BUILD = "build"


class StakeholderType(Enum):
    """Types of ecosystem stakeholders"""
    CORE_MNC = "core_mnc"
    FINANCIAL_INSTITUTION = "financial_institution"
    TECHNOLOGY_PARTNER = "technology_partner"
    REGULATORY_BODY = "regulatory_body"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOMER_SEGMENT = "customer_segment"


@dataclass
class FinancialResource:
    """Financial resource in the ecosystem"""
    type: str
    amount: float
    currency: str
    allocation_phase: EcosystemPhase
    stakeholder: str
    risk_level: float = 0.0
    expected_return: float = 0.0


@dataclass
class EcosystemStakeholder:
    """Stakeholder in the business ecosystem"""
    name: str
    type: StakeholderType
    financial_capacity: float
    integration_level: float = 0.0
    trust_score: float = 0.5
    digital_readiness: float = 0.0
    resources: List[FinancialResource] = field(default_factory=list)


class ThreePhaseModel:
    """
    Implementation of the three-phase financial ecosystem development model
    """
    
    def __init__(self, ecosystem_name: str):
        self.ecosystem_name = ecosystem_name
        self.stakeholders: Dict[str, EcosystemStakeholder] = {}
        self.current_phase = EcosystemPhase.DEFINE
        self.phase_metrics = {}
        self.financial_flows = []
        self.integration_matrix = None
        
    def add_stakeholder(self, stakeholder: EcosystemStakeholder):
        """Add stakeholder to the ecosystem"""
        self.stakeholders[stakeholder.name] = stakeholder
        
    def execute_phase_1_define(self) -> Dict[str, Any]:
        """
        Phase 1: Define Financial Strategy
        - Identify relevant trends and plan value creation pools
        - Form strategic financial reserves for innovation
        - Develop partnership financing mechanisms
        - Create R&D and technology development funds
        """
        
        phase_1_results = {
            "phase": "DEFINE",
            "objectives": [
                "trend_identification",
                "value_pool_planning", 
                "strategic_reserves",
                "partnership_mechanisms",
                "rd_funds"
            ],
            "stakeholder_analysis": self._analyze_stakeholders(),
            "financial_strategy": self._develop_financial_strategy(),
            "risk_assessment": self._assess_phase_1_risks(),
            "success_metrics": self._define_phase_1_metrics()
        }
        
        self.phase_metrics["phase_1"] = phase_1_results
        return phase_1_results
    
    def execute_phase_2_design(self) -> Dict[str, Any]:
        """
        Phase 2: Design Financial Architecture
        - Develop risk distribution system among ecosystem participants
        - Create mutual lending and financial support mechanisms
        - Form unified financial reporting and control standards
        - Develop APIs and digital financial interfaces
        """
        
        if self.current_phase != EcosystemPhase.DEFINE:
            raise ValueError("Must complete Phase 1 (Define) before Phase 2")
            
        phase_2_results = {
            "phase": "DESIGN",
            "objectives": [
                "risk_distribution",
                "mutual_lending",
                "reporting_standards",
                "digital_interfaces"
            ],
            "architecture_design": self._design_financial_architecture(),
            "integration_matrix": self._create_integration_matrix(),
            "api_specifications": self._design_api_framework(),
            "risk_framework": self._design_risk_framework(),
            "success_metrics": self._define_phase_2_metrics()
        }
        
        self.current_phase = EcosystemPhase.DESIGN
        self.phase_metrics["phase_2"] = phase_2_results
        return phase_2_results
    
    def execute_phase_3_build(self) -> Dict[str, Any]:
        """
        Phase 3: Build Operational Integration
        - Implement unified payment and settlement systems
        - Create joint investment funds and development programs
        - Form operational financial monitoring and control
        """
        
        if self.current_phase != EcosystemPhase.DESIGN:
            raise ValueError("Must complete Phase 2 (Design) before Phase 3")
            
        phase_3_results = {
            "phase": "BUILD",
            "objectives": [
                "payment_systems",
                "investment_funds",
                "monitoring_control"
            ],
            "operational_systems": self._build_operational_systems(),
            "investment_vehicles": self._create_investment_vehicles(),
            "monitoring_framework": self._build_monitoring_framework(),
            "integration_completion": self._finalize_integration(),
            "success_metrics": self._define_phase_3_metrics()
        }
        
        self.current_phase = EcosystemPhase.BUILD
        self.phase_metrics["phase_3"] = phase_3_results
        return phase_3_results
    
    def _analyze_stakeholders(self) -> Dict[str, Any]:
        """Analyze current stakeholder landscape"""
        
        analysis = {
            "total_stakeholders": len(self.stakeholders),
            "stakeholder_types": {},
            "financial_capacity_distribution": {},
            "readiness_assessment": {}
        }
        
        for stakeholder in self.stakeholders.values():
            # Count by type
            type_name = stakeholder.type.value
            analysis["stakeholder_types"][type_name] = analysis["stakeholder_types"].get(type_name, 0) + 1
            
            # Financial capacity analysis
            capacity_tier = self._categorize_financial_capacity(stakeholder.financial_capacity)
            analysis["financial_capacity_distribution"][capacity_tier] = analysis["financial_capacity_distribution"].get(capacity_tier, 0) + 1
            
            # Readiness assessment
            analysis["readiness_assessment"][stakeholder.name] = {
                "digital_readiness": stakeholder.digital_readiness,
                "trust_score": stakeholder.trust_score,
                "integration_potential": (stakeholder.digital_readiness + stakeholder.trust_score) / 2
            }
        
        return analysis
    
    def _develop_financial_strategy(self) -> Dict[str, Any]:
        """Develop comprehensive financial strategy for ecosystem"""
        
        total_capacity = sum(s.financial_capacity for s in self.stakeholders.values())
        
        strategy = {
            "total_ecosystem_capacity": total_capacity,
            "strategic_allocation": {
                "innovation_fund": total_capacity * 0.15,
                "risk_reserve": total_capacity * 0.10,
                "partnership_development": total_capacity * 0.08,
                "technology_infrastructure": total_capacity * 0.12,
                "operational_capital": total_capacity * 0.55
            },
            "value_creation_pools": [
                {
                    "name": "Digital Transformation Pool",
                    "target_value": total_capacity * 0.20,
                    "participants": [s.name for s in self.stakeholders.values() if s.digital_readiness > 0.6]
                },
                {
                    "name": "Financial Inclusion Pool",
                    "target_value": total_capacity * 0.25,
                    "participants": [s.name for s in self.stakeholders.values() if s.type in [StakeholderType.FINANCIAL_INSTITUTION, StakeholderType.CORE_MNC]]
                },
                {
                    "name": "Innovation Development Pool",
                    "target_value": total_capacity * 0.15,
                    "participants": [s.name for s in self.stakeholders.values() if s.type == StakeholderType.TECHNOLOGY_PARTNER]
                }
            ]
        }
        
        return strategy
    
    def _assess_phase_1_risks(self) -> Dict[str, Any]:
        """Assess risks in Phase 1"""
        
        return {
            "stakeholder_alignment_risk": self._calculate_alignment_risk(),
            "financial_commitment_risk": self._calculate_commitment_risk(),
            "regulatory_compliance_risk": self._calculate_regulatory_risk(),
            "technology_integration_risk": self._calculate_technology_risk(),
            "overall_risk_score": 0.0  # Will be calculated based on above
        }
    
    def _design_financial_architecture(self) -> Dict[str, Any]:
        """Design financial architecture for Phase 2"""
        
        return {
            "core_components": [
                "Central Financial Hub",
                "Distributed Ledger System", 
                "Risk Management Engine",
                "Compliance Monitoring System",
                "Analytics and Reporting Platform"
            ],
            "integration_layers": {
                "data_layer": "Unified data standards and APIs",
                "service_layer": "Financial services integration",
                "presentation_layer": "Stakeholder interfaces",
                "security_layer": "Authentication and authorization"
            },
            "financial_flows": self._model_financial_flows(),
            "governance_structure": self._design_governance()
        }
    
    def _model_financial_flows(self) -> Dict[str, Any]:
        """Model financial flows in the ecosystem"""
        return {
            "flow_types": ["Investment", "Revenue", "Cost sharing", "Risk transfer"],
            "flow_volume": sum(s.financial_capacity for s in self.stakeholders.values()),
            "flow_frequency": "Real-time with batch settlement"
        }
    
    def _design_api_framework(self) -> Dict[str, Any]:
        """Design API framework for ecosystem integration"""
        return {
            "api_standards": "RESTful with GraphQL support",
            "authentication": "OAuth 2.0 with JWT tokens",
            "rate_limiting": "Tier-based with SLA guarantees",
            "documentation": "OpenAPI 3.0 specifications"
        }
    
    def _design_risk_framework(self) -> Dict[str, Any]:
        """Design risk management framework"""
        return {
            "risk_types": ["Credit", "Operational", "Market", "Liquidity", "Regulatory"],
            "assessment_methodology": "Monte Carlo simulation with stress testing",
            "mitigation_strategies": "Diversification and hedging",
            "monitoring_frequency": "Real-time with daily reporting"
        }
    
    def _build_operational_systems(self) -> Dict[str, Any]:
        """Build operational systems for Phase 3"""
        return {
            "payment_rails": "Multi-currency with real-time settlement",
            "clearing_system": "Automated with exception handling",
            "reconciliation": "Daily automated with variance reporting",
            "customer_onboarding": "Digital KYC with biometric verification"
        }
    
    def _create_investment_vehicles(self) -> Dict[str, Any]:
        """Create investment vehicles"""
        return {
            "ecosystem_fund": "Joint investment vehicle for shared initiatives",
            "innovation_fund": "R&D and technology development",
            "risk_pool": "Shared insurance and guarantee mechanisms",
            "liquidity_facility": "Emergency funding and credit lines"
        }
    
    def _design_governance(self) -> Dict[str, Any]:
        """Design governance structure"""
        return {
            "decision_making": "Consensus-based with weighted voting",
            "oversight_committee": "Multi-stakeholder board",
            "compliance_framework": "International standards alignment"
        }
    
    def _build_monitoring_framework(self) -> Dict[str, Any]:
        """Build monitoring and control framework"""
        return {
            "dashboards": "Real-time performance and risk monitoring",
            "alerts": "Automated threshold-based notifications",
            "reporting": "Regulatory and management reporting",
            "analytics": "Predictive modeling and trend analysis"
        }
    
    def _finalize_integration(self) -> Dict[str, Any]:
        """Finalize ecosystem integration"""
        # Update integration levels for all stakeholders
        for stakeholder in self.stakeholders.values():
            stakeholder.integration_level = min(1.0, stakeholder.trust_score * stakeholder.digital_readiness * 1.2)
        
        return {
            "integration_completion": True,
            "average_integration_level": np.mean([s.integration_level for s in self.stakeholders.values()]),
            "ecosystem_maturity": "Operational",
            "next_phase": "Optimization and scaling"
        }
    
    def _create_integration_matrix(self) -> np.ndarray:
        """Create stakeholder integration matrix"""
        
        n_stakeholders = len(self.stakeholders)
        matrix = np.zeros((n_stakeholders, n_stakeholders))
        
        stakeholder_list = list(self.stakeholders.keys())
        
        for i, stakeholder_a in enumerate(stakeholder_list):
            for j, stakeholder_b in enumerate(stakeholder_list):
                if i != j:
                    # Calculate integration potential based on trust and compatibility
                    integration_score = self._calculate_integration_potential(
                        self.stakeholders[stakeholder_a],
                        self.stakeholders[stakeholder_b]
                    )
                    matrix[i, j] = integration_score
                else:
                    matrix[i, j] = 1.0  # Self-integration is perfect
        
        self.integration_matrix = matrix
        return matrix
    
    def _calculate_integration_potential(self, stakeholder_a: EcosystemStakeholder, stakeholder_b: EcosystemStakeholder) -> float:
        """Calculate integration potential between two stakeholders"""
        
        # Base compatibility by type
        type_compatibility = self._get_type_compatibility(stakeholder_a.type, stakeholder_b.type)
        
        # Trust factor
        trust_factor = (stakeholder_a.trust_score + stakeholder_b.trust_score) / 2
        
        # Digital readiness alignment
        readiness_alignment = 1 - abs(stakeholder_a.digital_readiness - stakeholder_b.digital_readiness)
        
        # Financial capacity balance (avoid too large imbalances)
        capacity_ratio = min(stakeholder_a.financial_capacity, stakeholder_b.financial_capacity) / max(stakeholder_a.financial_capacity, stakeholder_b.financial_capacity)
        
        # Weighted integration score
        integration_score = (
            type_compatibility * 0.3 +
            trust_factor * 0.3 +
            readiness_alignment * 0.2 +
            capacity_ratio * 0.2
        )
        
        return min(1.0, max(0.0, integration_score))
    
    def _get_type_compatibility(self, type_a: StakeholderType, type_b: StakeholderType) -> float:
        """Get compatibility score between stakeholder types"""
        
        compatibility_matrix = {
            (StakeholderType.CORE_MNC, StakeholderType.FINANCIAL_INSTITUTION): 0.9,
            (StakeholderType.CORE_MNC, StakeholderType.TECHNOLOGY_PARTNER): 0.8,
            (StakeholderType.CORE_MNC, StakeholderType.SUPPLY_CHAIN): 0.7,
            (StakeholderType.FINANCIAL_INSTITUTION, StakeholderType.TECHNOLOGY_PARTNER): 0.8,
            (StakeholderType.FINANCIAL_INSTITUTION, StakeholderType.REGULATORY_BODY): 0.9,
            (StakeholderType.TECHNOLOGY_PARTNER, StakeholderType.SUPPLY_CHAIN): 0.6,
            (StakeholderType.REGULATORY_BODY, StakeholderType.FINANCIAL_INSTITUTION): 0.9,
            (StakeholderType.SUPPLY_CHAIN, StakeholderType.CUSTOMER_SEGMENT): 0.8
        }
        
        # Check both directions
        key1 = (type_a, type_b)
        key2 = (type_b, type_a)
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.5))
    
    def _categorize_financial_capacity(self, capacity: float) -> str:
        """Categorize financial capacity into tiers"""
        
        if capacity >= 1000000000:  # 1B+
            return "Tier_1_Large"
        elif capacity >= 100000000:  # 100M+
            return "Tier_2_Medium"
        elif capacity >= 10000000:   # 10M+
            return "Tier_3_Small"
        else:
            return "Tier_4_Micro"
    
    def _calculate_alignment_risk(self) -> float:
        """Calculate stakeholder alignment risk"""
        
        if not self.stakeholders:
            return 1.0
            
        trust_scores = [s.trust_score for s in self.stakeholders.values()]
        trust_variance = np.var(trust_scores)
        
        # Higher variance = higher risk
        return min(1.0, trust_variance * 2)
    
    def _calculate_commitment_risk(self) -> float:
        """Calculate financial commitment risk"""
        
        capacities = [s.financial_capacity for s in self.stakeholders.values()]
        if not capacities:
            return 1.0
            
        # Risk based on capacity concentration
        total_capacity = sum(capacities)
        largest_capacity = max(capacities)
        concentration_ratio = largest_capacity / total_capacity
        
        # Higher concentration = higher risk
        return min(1.0, concentration_ratio * 1.2)
    
    def _calculate_regulatory_risk(self) -> float:
        """Calculate regulatory compliance risk"""
        
        regulatory_stakeholders = [s for s in self.stakeholders.values() if s.type == StakeholderType.REGULATORY_BODY]
        
        if not regulatory_stakeholders:
            return 0.8  # High risk if no regulatory stakeholders
        
        # Risk decreases with regulatory stakeholder integration
        avg_trust = sum(s.trust_score for s in regulatory_stakeholders) / len(regulatory_stakeholders)
        return 1.0 - avg_trust
    
    def _calculate_technology_risk(self) -> float:
        """Calculate technology integration risk"""
        
        readiness_scores = [s.digital_readiness for s in self.stakeholders.values()]
        if not readiness_scores:
            return 1.0
            
        avg_readiness = np.mean(readiness_scores)
        readiness_variance = np.var(readiness_scores)
        
        # Risk from low average readiness and high variance
        return min(1.0, (1 - avg_readiness) * 0.7 + readiness_variance * 0.3)
    
    def _define_phase_1_metrics(self) -> List[str]:
        """Define success metrics for Phase 1"""
        
        return [
            "stakeholder_commitment_rate",
            "financial_resource_mobilization",
            "strategic_alignment_score",
            "innovation_fund_establishment",
            "partnership_agreements_signed"
        ]
    
    def _define_phase_2_metrics(self) -> List[str]:
        """Define success metrics for Phase 2"""
        
        return [
            "architecture_component_completion",
            "api_integration_success_rate",
            "risk_framework_implementation", 
            "reporting_standard_adoption",
            "stakeholder_system_readiness"
        ]
    
    def _define_phase_3_metrics(self) -> List[str]:
        """Define success metrics for Phase 3"""
        
        return [
            "payment_system_transaction_volume",
            "investment_fund_utilization_rate",
            "monitoring_system_coverage",
            "stakeholder_satisfaction_score",
            "ecosystem_financial_performance"
        ]
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem status"""
        
        return {
            "ecosystem_name": self.ecosystem_name,
            "current_phase": self.current_phase.value,
            "total_stakeholders": len(self.stakeholders),
            "phase_completion": self.phase_metrics,
            "stakeholder_summary": {
                name: {
                    "type": s.type.value,
                    "capacity": s.financial_capacity,
                    "integration_level": s.integration_level,
                    "trust_score": s.trust_score,
                    "digital_readiness": s.digital_readiness
                }
                for name, s in self.stakeholders.items()
            }
        }


def create_sample_ecosystem() -> ThreePhaseModel:
    """Create a sample multinational business ecosystem"""
    
    ecosystem = ThreePhaseModel("Global Financial Inclusion Ecosystem")
    
    # Add sample stakeholders
    stakeholders = [
        EcosystemStakeholder("GlobalBank Corp", StakeholderType.CORE_MNC, 5000000000, 0.0, 0.8, 0.7),
        EcosystemStakeholder("Central Bank", StakeholderType.REGULATORY_BODY, 1000000000, 0.0, 0.9, 0.6),
        EcosystemStakeholder("FinTech Partner", StakeholderType.TECHNOLOGY_PARTNER, 100000000, 0.0, 0.7, 0.9),
        EcosystemStakeholder("Commercial Bank A", StakeholderType.FINANCIAL_INSTITUTION, 2000000000, 0.0, 0.8, 0.5),
        EcosystemStakeholder("Supply Chain Partner", StakeholderType.SUPPLY_CHAIN, 500000000, 0.0, 0.6, 0.4),
        EcosystemStakeholder("Customer Segment Rep", StakeholderType.CUSTOMER_SEGMENT, 50000000, 0.0, 0.7, 0.3)
    ]
    
    for stakeholder in stakeholders:
        ecosystem.add_stakeholder(stakeholder)
    
    return ecosystem


def main():
    """Demonstrate the three-phase model"""
    
    # Create sample ecosystem
    ecosystem = create_sample_ecosystem()
    
    print("=== Three-Phase Financial Ecosystem Development Model ===\n")
    
    # Execute Phase 1
    print("Executing Phase 1: DEFINE")
    phase_1_results = ecosystem.execute_phase_1_define()
    print(f"Phase 1 completed with {len(phase_1_results['objectives'])} objectives")
    
    # Execute Phase 2  
    print("\nExecuting Phase 2: DESIGN")
    phase_2_results = ecosystem.execute_phase_2_design()
    print(f"Phase 2 completed with {len(phase_2_results['objectives'])} objectives")
    
    # Execute Phase 3
    print("\nExecuting Phase 3: BUILD")
    phase_3_results = ecosystem.execute_phase_3_build()
    print(f"Phase 3 completed with {len(phase_3_results['objectives'])} objectives")
    
    # Get final status
    status = ecosystem.get_ecosystem_status()
    print(f"\nEcosystem '{status['ecosystem_name']}' development completed!")
    print(f"Current phase: {status['current_phase']}")
    print(f"Total stakeholders: {status['total_stakeholders']}")
    
    # Save results
    data_dir = os.environ.get("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)
    results_path = os.path.join(data_dir, "three_phase_results.json")
    with open(results_path, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()