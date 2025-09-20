"""
Amazon Financial Inclusion Model Visualization & Analysis
Візуалізація та аналіз результатів економетричної моделі фінансової інклюзії

Створює:
- Графіки Financial Inclusion Index
- Кореляційний аналіз факторів
- Діагностичні графіки моделі
- Інтерактивні дашборди
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinancialInclusionVisualizer:
    """Клас для візуалізації результатів моделі фінансової інклюзії"""
    
    def __init__(self, db_path='D:/Econometric/econometric_results.db'):
        self.db_path = db_path
        self.data = None
        self.load_data_from_db()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("📊 Financial Inclusion Visualizer Initialized")
        
    def load_data_from_db(self):
        """Завантаження даних з бази даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            self.data = pd.read_sql_query("""
                SELECT * FROM financial_inclusion_index 
                ORDER BY date
            """, conn)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
            conn.close()
            print(f"✅ Завантажено {len(self.data)} спостережень")
        except Exception as e:
            print(f"❌ Помилка завантаження даних: {e}")
            
    def plot_financial_inclusion_index(self):
        """Графік Financial Inclusion Index"""
        if self.data is None:
            print("❌ Дані не завантажені")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Amazon Financial Inclusion Index Analysis', fontsize=16, fontweight='bold')
        
        # 1. FI Index over time
        axes[0,0].plot(self.data.index, self.data['fi_index'], linewidth=2, color='navy')
        axes[0,0].set_title('Financial Inclusion Index Over Time')
        axes[0,0].set_ylabel('FI Index')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. FI Index distribution
        axes[0,1].hist(self.data['fi_index'], bins=30, alpha=0.7, color='darkblue')
        axes[0,1].set_title('FI Index Distribution')
        axes[0,1].set_xlabel('FI Index Value')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Rolling correlation with key factors
        window = 12
        correlations = []
        dates = []
        
        for i in range(window, len(self.data)):
            subset = self.data.iloc[i-window:i]
            corr_ps = subset['fi_index'].corr(subset['platform_scale'])
            corr_ne = subset['fi_index'].corr(subset['network_effects'])
            correlations.append([corr_ps, corr_ne])
            dates.append(subset.index[-1])
        
        correlations = np.array(correlations)
        axes[1,0].plot(dates, correlations[:, 0], label='Platform Scale', linewidth=2)
        axes[1,0].plot(dates, correlations[:, 1], label='Network Effects', linewidth=2)
        axes[1,0].set_title('Rolling Correlations with FI Index')
        axes[1,0].set_ylabel('Correlation Coefficient')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Decomposition: trend vs cycles
        from scipy import signal
        
        # Simple trend extraction using rolling mean
        trend = self.data['fi_index'].rolling(window=12).mean()
        cycle = self.data['fi_index'] - trend
        
        axes[1,1].plot(self.data.index, trend, label='Trend', linewidth=2)
        axes[1,1].plot(self.data.index, cycle, label='Cycle', alpha=0.7)
        axes[1,1].set_title('FI Index: Trend vs Cyclical Components')
        axes[1,1].set_ylabel('Index Value')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('D:/Econometric/results/fi_index_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ FI Index аналіз збережений: fi_index_analysis.png")
        
    def plot_factors_correlation_matrix(self):
        """Кореляційна матриця всіх факторів"""
        if self.data is None:
            print("❌ Дані не завантажені")
            return
            
        # Select relevant columns
        factor_columns = [
            'fi_index', 'platform_scale', 'network_effects', 'digital_infrastructure',
            'partner_integration', 'financial_services_diversity', 'regulatory_environment',
            'amazon_flywheel_effect', 'competitive_pressure', 'innovation_index'
        ]
        
        correlation_matrix = self.data[factor_columns].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix: Financial Inclusion Factors', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('D:/Econometric/results/factors_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Кореляційна матриця збережена: factors_correlation_matrix.png")
        
        # Print key correlations
        print("\n🔍 КЛЮЧОВІ КОРЕЛЯЦІЇ З FI INDEX:")
        fi_correlations = correlation_matrix['fi_index'].sort_values(key=abs, ascending=False)
        for factor, corr in fi_correlations.items():
            if factor != 'fi_index':
                print(f"{factor:30s}: {corr:6.3f}")
                
    def plot_factor_dynamics(self):
        """Динаміка всіх факторів у часі"""
        if self.data is None:
            print("❌ Дані не завантажені")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Amazon Ecosystem Factors Dynamics', fontsize=16, fontweight='bold')
        
        factors = [
            'platform_scale', 'network_effects', 'digital_infrastructure',
            'partner_integration', 'financial_services_diversity', 'regulatory_environment',
            'amazon_flywheel_effect', 'competitive_pressure', 'innovation_index'
        ]
        
        factor_names = [
            'Platform Scale', 'Network Effects', 'Digital Infrastructure',
            'Partner Integration', 'Financial Services Diversity', 'Regulatory Environment',
            'Amazon Flywheel Effect', 'Competitive Pressure', 'Innovation Index'
        ]
        
        for i, (factor, name) in enumerate(zip(factors, factor_names)):
            row = i // 3
            col = i % 3
            
            axes[row, col].plot(self.data.index, self.data[factor], linewidth=2)
            axes[row, col].set_title(name)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(self.data)), self.data[factor], 1)
            p = np.poly1d(z)
            axes[row, col].plot(self.data.index, p(range(len(self.data))), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('D:/Econometric/results/factors_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Динаміка факторів збережена: factors_dynamics.png")
        
    def create_interactive_dashboard(self):
        """Створення інтерактивного дашборду"""
        if self.data is None:
            print("❌ Дані не завантажені")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Financial Inclusion Index Timeline',
                'FI Index vs Platform Scale', 
                'Key Factors Comparison',
                'Flywheel Effect Analysis',
                'Competitive vs Innovation Dynamics',
                'Regulatory Environment Impact'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # 1. FI Index Timeline
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['fi_index'],
                      mode='lines', name='FI Index', line=dict(width=3)),
            row=1, col=1
        )
        
        # 2. FI Index vs Platform Scale scatter
        fig.add_trace(
            go.Scatter(x=self.data['platform_scale'], y=self.data['fi_index'],
                      mode='markers', name='FI vs Platform Scale',
                      marker=dict(size=8, color=self.data.index.year, colorscale='viridis')),
            row=1, col=2
        )
        
        # 3. Key Factors Comparison
        key_factors = ['platform_scale', 'network_effects', 'digital_infrastructure', 'amazon_flywheel_effect']
        for factor in key_factors:
            fig.add_trace(
                go.Scatter(x=self.data.index, y=self.data[factor],
                          mode='lines', name=factor.replace('_', ' ').title()),
                row=2, col=1
            )
        
        # 4. Flywheel Effect Analysis
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['amazon_flywheel_effect'],
                      mode='lines', name='Flywheel Effect', line=dict(color='red', width=3)),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['fi_index'],
                      mode='lines', name='FI Index', line=dict(color='blue', width=2)),
            row=3, col=1, secondary_y=True
        )
        
        # 5. Competition vs Innovation
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['competitive_pressure'],
                      mode='lines', name='Competitive Pressure', line=dict(color='orange')),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['innovation_index'],
                      mode='lines', name='Innovation Index', line=dict(color='green')),
            row=3, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Amazon Financial Inclusion Ecosystem Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive dashboard
        fig.write_html('D:/Econometric/results/amazon_fi_dashboard.html')
        fig.show()
        
        print("✅ Інтерактивний дашборд збережений: amazon_fi_dashboard.html")
        
    def analyze_model_performance(self):
        """Аналіз якості моделі"""
        try:
            conn = sqlite3.connect(self.db_path)
            model_results = pd.read_sql_query("""
                SELECT * FROM fi_model_results 
                ORDER BY date_created DESC
            """, conn)
            conn.close()
            
            if len(model_results) > 0:
                print("\n📊 РЕЗУЛЬТАТИ МОДЕЛІ:")
                print("=" * 40)
                
                for _, row in model_results.iterrows():
                    print(f"Модель: {row['model_name']}")
                    print(f"R²: {row['r_squared']:.4f}")
                    print(f"Adj. R²: {row['adj_r_squared']:.4f}")
                    print(f"F-статистика: {row['f_statistic']:.4f}")
                    print(f"Durbin-Watson: {row['durbin_watson']:.4f}")
                    print("-" * 40)
                    
        except Exception as e:
            print(f"❌ Помилка аналізу моделі: {e}")
            
    def generate_summary_statistics(self):
        """Генерація зведеної статистики"""
        if self.data is None:
            print("❌ Дані не завантажені")
            return
            
        print("\n📈 ЗВЕДЕНА СТАТИСТИКА AMAZON FINANCIAL INCLUSION MODEL")
        print("=" * 65)
        
        # Basic statistics
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        summary_stats = self.data[numeric_columns].describe()
        
        print("\n📊 Описова статистика:")
        print(summary_stats.round(4))
        
        # Correlation with FI Index
        print(f"\n🎯 Кореляції з Financial Inclusion Index:")
        fi_correlations = self.data.corr()['fi_index'].sort_values(key=abs, ascending=False)
        
        for factor, corr in fi_correlations.items():
            if factor != 'fi_index' and abs(corr) > 0.1:
                direction = "📈" if corr > 0 else "📉"
                print(f"{direction} {factor:30s}: {corr:6.3f}")
        
        # Time period analysis
        print(f"\n📅 Період аналізу:")
        print(f"   Початок: {self.data.index.min().strftime('%Y-%m-%d')}")
        print(f"   Кінець: {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"   Тривалість: {(self.data.index.max() - self.data.index.min()).days} днів")
        print(f"   Спостережень: {len(self.data)}")
        
        # Volatility analysis
        print(f"\n📊 Аналіз волатильності:")
        volatilities = self.data[numeric_columns].std().sort_values(ascending=False)
        for factor, vol in volatilities.head(5).items():
            print(f"   {factor:30s}: {vol:.4f}")
            
        return summary_stats

def run_complete_analysis():
    """Запуск повного аналізу та візуалізації"""
    print("🚀 ЗАПУСК ПОВНОГО АНАЛІЗУ AMAZON FINANCIAL INCLUSION MODEL")
    print("=" * 70)
    
    # Initialize visualizer
    viz = FinancialInclusionVisualizer()
    
    if viz.data is not None:
        print("\n1️⃣ Генерація зведеної статистики...")
        viz.generate_summary_statistics()
        
        print("\n2️⃣ Створення графіків FI Index...")
        viz.plot_financial_inclusion_index()
        
        print("\n3️⃣ Аналіз кореляцій факторів...")
        viz.plot_factors_correlation_matrix()
        
        print("\n4️⃣ Візуалізація динаміки факторів...")
        viz.plot_factor_dynamics()
        
        print("\n5️⃣ Створення інтерактивного дашборду...")
        viz.create_interactive_dashboard()
        
        print("\n6️⃣ Аналіз якості моделі...")
        viz.analyze_model_performance()
        
        print("\n🎉 АНАЛІЗ ЗАВЕРШЕНО!")
        print("✅ Всі графіки та звіти збережені в D:/Econometric/results/")
        
    else:
        print("❌ Спочатку запустіть amazon_financial_inclusion_model.py")

if __name__ == "__main__":
    run_complete_analysis()
