"""
Amazon Financial Inclusion Model з РЕАЛЬНИМИ ДАНИМИ
Інтеграція вилучених даних з PDF звітів Amazon

Ця модель поєднує:
1. Реальні фінансові дані з офіційних звітів Amazon (PDF)
2. Ринкові дані з Yahoo Finance
3. Теорію Adner & Kapoor для бізнес-екосистем
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
import json
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class AmazonRealDataFinancialInclusionModel:
    """
    Економетрична модель фінансової інклюзії з використанням реальних даних Amazon
    """
    
    def __init__(self):
        self.real_amazon_data = None
        self.market_data = {}
        self.financial_inclusion_index = None
        self.explanatory_variables = {}
        self.models = {}
        
        print("🎯 AMAZON REAL DATA FINANCIAL INCLUSION MODEL")
        print("=" * 55)
        print("📊 Використовує РЕАЛЬНІ дані з PDF звітів Amazon")
        print("🔬 Базована на теорії Adner & Kapoor (2016)")
        
    def load_real_amazon_data(self):
        """Завантаження реальних даних з PDF звітів"""
        print("\n📄 ЗАВАНТАЖЕННЯ РЕАЛЬНИХ ДАНИХ З PDF")
        print("=" * 45)
        
        # Load CSV with extracted real data
        try:
            self.real_amazon_data = pd.read_csv("D:/Econometric/amazon_real_financial_data.csv")
            self.real_amazon_data['date'] = pd.to_datetime(self.real_amazon_data['date'])
            self.real_amazon_data.set_index('date', inplace=True)
            
            print(f"✅ Завантажено {len(self.real_amazon_data)} реальних спостережень")
            print(f"📅 Період: {self.real_amazon_data.index.min()} - {self.real_amazon_data.index.max()}")
            
            # Display sample of real data
            print("\n📊 Зразок реальних даних Amazon:")
            print(self.real_amazon_data[['revenue', 'operating_income', 'aws_revenue', 'free_cash_flow']].head())
            
            return True
            
        except Exception as e:
            print(f"❌ Помилка завантаження реальних даних: {e}")
            return False
    
    def load_json_data(self):
        """Завантаження детальних даних з JSON"""
        try:
            with open("D:/Econometric/amazon_real_data.json", 'r') as f:
                self.detailed_data = json.load(f)
            
            print(f"📋 Завантажено детальні дані з {len(self.detailed_data)} файлів")
            return True
        except Exception as e:
            print(f"❌ Помилка завантаження JSON: {e}")
            return False
    
    def construct_real_financial_inclusion_index(self):
        """
        Побудова індексу фінансової інклюзії на основі РЕАЛЬНИХ даних Amazon
        """
        print("\n🏗️ ПОБУДОВА REAL FINANCIAL INCLUSION INDEX")
        print("=" * 50)
        
        data = self.real_amazon_data.copy()
        
        # 1. Revenue Growth Rate (платформне зростання)
        print("1️⃣ Revenue Growth - зростання доходів")
        data['revenue_growth'] = data['revenue'].pct_change()
        
        # 2. AWS Revenue Share (цифрова інфраструктура)
        print("2️⃣ AWS Share - частка хмарних сервісів")
        data['aws_share'] = data['aws_revenue'] / data['revenue']
        data['aws_share'] = data['aws_share'].fillna(data['aws_share'].mean())
        
        # 3. Operating Efficiency (операційна ефективність)
        print("3️⃣ Operating Margin - операційна маржа")
        data['operating_margin'] = data['operating_income'] / data['revenue']
        
        # 4. Cash Generation Capability (фінансова стабільність)
        print("4️⃣ FCF Yield - рентабельність грошових потоків")
        data['fcf_yield'] = data['free_cash_flow'] / data['revenue']
        data['fcf_yield'] = data['fcf_yield'].fillna(0)
        
        # 5. Business Diversification (різноманітність бізнесу)
        print("5️⃣ Business Diversification - диверсифікація")
        # Проксі через стабільність операційної маржі
        data['margin_stability'] = 1 / (data['operating_margin'].rolling(window=4).std() + 0.001)
        
        # Normalize all components
        components = ['revenue_growth', 'aws_share', 'operating_margin', 'fcf_yield', 'margin_stability']
        
        for comp in components:
            if comp in data.columns:
                data[f'{comp}_norm'] = (data[comp] - data[comp].mean()) / data[comp].std()
                data[f'{comp}_norm'] = data[f'{comp}_norm'].fillna(0)
        
        # Create composite Financial Inclusion Index
        weights = {
            'revenue_growth_norm': 0.25,  # Платформне зростання
            'aws_share_norm': 0.25,       # Цифровізація
            'operating_margin_norm': 0.20, # Ефективність
            'fcf_yield_norm': 0.20,       # Фінансова стійкість
            'margin_stability_norm': 0.10  # Стабільність бізнесу
        }
        
        self.financial_inclusion_index = pd.Series(0, index=data.index)
        
        for component, weight in weights.items():
            if component in data.columns:
                self.financial_inclusion_index += weight * data[component]
        
        print("✅ Real Financial Inclusion Index створено!")
        print(f"📈 Індекс варіює від {self.financial_inclusion_index.min():.3f} до {self.financial_inclusion_index.max():.3f}")
        
        return self.financial_inclusion_index
    
    def construct_real_explanatory_variables(self):
        """
        Побудова пояснювальних змінних на основі реальних даних
        """
        print("\n🔧 ПОБУДОВА РЕАЛЬНИХ ПОЯСНЮВАЛЬНИХ ЗМІННИХ")
        print("=" * 50)
        
        data = self.real_amazon_data.copy()
        
        # 1. Platform Scale (масштаб платформи) - на основі доходів
        print("1️⃣ Real Platform Scale")
        platform_scale = np.log(data['revenue'])  # Log transformation for growth
        platform_scale = (platform_scale - platform_scale.mean()) / platform_scale.std()
        
        # 2. Network Effects (мережеві ефекти) - на основі зростання доходів
        print("2️⃣ Real Network Effects")
        revenue_growth = data['revenue'].pct_change()
        network_effects = revenue_growth.rolling(window=4).mean()
        network_effects = (network_effects - network_effects.mean()) / network_effects.std()
        network_effects = network_effects.fillna(0)
        
        # 3. Digital Infrastructure (цифрова інфраструктура) - AWS зростання
        print("3️⃣ Real Digital Infrastructure")
        aws_growth = data['aws_revenue'].pct_change()
        digital_infrastructure = aws_growth.rolling(window=3).mean()
        digital_infrastructure = (digital_infrastructure - digital_infrastructure.mean()) / digital_infrastructure.std()
        digital_infrastructure = digital_infrastructure.fillna(0)
        
        # 4. Financial Services Quality - операційна ефективність
        print("4️⃣ Real Financial Services Quality")
        operating_efficiency = data['operating_income'] / data['revenue']
        financial_services_quality = operating_efficiency.rolling(window=3).mean()
        financial_services_quality = (financial_services_quality - financial_services_quality.mean()) / financial_services_quality.std()
        financial_services_quality = financial_services_quality.fillna(0)
        
        # 5. Business Ecosystem Maturity - стабільність FCF
        print("5️⃣ Real Ecosystem Maturity")
        fcf_stability = data['free_cash_flow'].rolling(window=4).std()
        ecosystem_maturity = -fcf_stability  # Lower volatility = higher maturity
        ecosystem_maturity = (ecosystem_maturity - ecosystem_maturity.mean()) / ecosystem_maturity.std()
        ecosystem_maturity = ecosystem_maturity.fillna(0)
        
        # 6. Innovation Intensity - R&D проксі через зростання AWS
        print("6️⃣ Real Innovation Intensity")
        aws_share = data['aws_revenue'] / data['revenue']
        aws_share = aws_share.fillna(aws_share.mean())
        innovation_intensity = aws_share.rolling(window=3).mean()
        innovation_intensity = (innovation_intensity - innovation_intensity.mean()) / innovation_intensity.std()
        innovation_intensity = innovation_intensity.fillna(0)
        
        self.explanatory_variables = {
            'platform_scale': platform_scale,
            'network_effects': network_effects,
            'digital_infrastructure': digital_infrastructure,
            'financial_services_quality': financial_services_quality,
            'ecosystem_maturity': ecosystem_maturity,
            'innovation_intensity': innovation_intensity
        }
        
        print("✅ Всі реальні змінні побудовані!")
        return self.explanatory_variables
    
    def estimate_real_data_model(self):
        """
        Оцінка моделі на основі реальних даних Amazon
        """
        print("\n🎯 ОЦІНКА МОДЕЛІ З РЕАЛЬНИМИ ДАНИМИ")
        print("=" * 45)
        
        # Prepare dataset
        model_data = pd.DataFrame({
            'fi_index': self.financial_inclusion_index
        })
        
        # Add explanatory variables
        for var_name, var_data in self.explanatory_variables.items():
            model_data[var_name] = var_data
        
        # Remove missing values
        model_data = model_data.dropna()
        print(f"📊 Кількість спостережень: {len(model_data)}")
        
        if len(model_data) < 10:
            print("⚠️ Недостатньо спостережень для оцінки моделі")
            return None
        
        # Dependent variable
        y = model_data['fi_index']
        
        # Independent variables
        X = model_data[list(self.explanatory_variables.keys())]
        X = sm.add_constant(X)
        
        # Estimate model
        print("🔬 Оцінка параметрів з реальними даними...")
        model = sm.OLS(y, X).fit()
        
        # Store results
        self.models['real_data_model'] = model
        
        # Print results
        print("\n📊 РЕЗУЛЬТАТИ МОДЕЛІ З РЕАЛЬНИМИ ДАНИМИ:")
        print("=" * 50)
        print(model.summary())
        
        # Model statistics
        print("\n📈 СТАТИСТИКИ МОДЕЛІ:")
        print(f"R² = {model.rsquared:.4f}")
        print(f"Adj. R² = {model.rsquared_adj:.4f}")
        print(f"F-статистика = {model.fvalue:.4f}")
        print(f"Durbin-Watson = {durbin_watson(model.resid):.4f}")
        
        # Significance test
        significant_vars = []
        for var in X.columns[1:]:  # Skip constant
            p_value = model.pvalues[var]
            if p_value < 0.05:
                significant_vars.append(f"{var} (p={p_value:.3f})")
        
        print("\n⭐ Статистично значущі змінні:")
        for var in significant_vars:
            print(f"  • {var}")
        
        return model
    
    def create_real_data_visualization(self):
        """
        Візуалізація результатів з реальними даними
        """
        print("\n📊 ВІЗУАЛІЗАЦІЯ РЕАЛЬНИХ ДАНИХ")
        print("=" * 35)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Amazon Financial Inclusion Model - РЕАЛЬНІ ДАНІ', fontsize=16, fontweight='bold')
        
        # 1. Financial Inclusion Index timeline
        axes[0,0].plot(self.financial_inclusion_index.index, self.financial_inclusion_index.values, 
                      linewidth=3, color='blue', marker='o')
        axes[0,0].set_title('Financial Inclusion Index (Real Data)')
        axes[0,0].set_ylabel('FI Index')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Revenue Growth
        revenue_data = self.real_amazon_data['revenue']
        axes[0,1].plot(revenue_data.index, revenue_data.values, 
                      linewidth=2, color='green', marker='s')
        axes[0,1].set_title('Amazon Revenue (Billions USD)')
        axes[0,1].set_ylabel('Revenue')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. AWS Revenue Growth
        aws_data = self.real_amazon_data['aws_revenue'].dropna()
        if len(aws_data) > 0:
            axes[0,2].plot(aws_data.index, aws_data.values, 
                          linewidth=2, color='orange', marker='^')
            axes[0,2].set_title('AWS Revenue (Billions USD)')
            axes[0,2].set_ylabel('AWS Revenue')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Operating Margin
        op_margin = self.real_amazon_data['operating_income'] / self.real_amazon_data['revenue']
        axes[1,0].plot(op_margin.index, op_margin.values, 
                      linewidth=2, color='red', marker='d')
        axes[1,0].set_title('Operating Margin')
        axes[1,0].set_ylabel('Margin %')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Free Cash Flow
        fcf_data = self.real_amazon_data['free_cash_flow'].dropna()
        if len(fcf_data) > 0:
            axes[1,1].plot(fcf_data.index, fcf_data.values, 
                          linewidth=2, color='purple', marker='*')
            axes[1,1].set_title('Free Cash Flow (Billions USD)')
            axes[1,1].set_ylabel('FCF')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Model Residuals (if model exists)
        if 'real_data_model' in self.models:
            model = self.models['real_data_model']
            axes[1,2].scatter(model.fittedvalues, model.resid, alpha=0.7)
            axes[1,2].axhline(y=0, color='red', linestyle='--')
            axes[1,2].set_title('Model Residuals')
            axes[1,2].set_xlabel('Fitted Values')
            axes[1,2].set_ylabel('Residuals')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('D:/Econometric/amazon_real_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Візуалізація збережена: amazon_real_data_analysis.png")
    
    def save_real_results(self):
        """
        Збереження результатів в базу даних
        """
        print("\n💾 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
        print("=" * 30)
        
        if 'real_data_model' not in self.models:
            print("⚠️ Модель не оцінена")
            return
        
        model = self.models['real_data_model']
        
        # Connect to database
        conn = sqlite3.connect('D:/Econometric/econometric_results.db')
        
        # Save model results
        model_data = {
            'model_name': 'amazon_real_data_financial_inclusion',
            'date_created': datetime.now().isoformat(),
            'coefficients': model.params.to_json(),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'durbin_watson': durbin_watson(model.resid),
            'model_specification': 'FI_Real = f(Platform_Scale, Network_Effects, Digital_Infrastructure, Financial_Services_Quality, Ecosystem_Maturity, Innovation_Intensity)'
        }
        
        # Insert into database
        placeholders = ', '.join(['?' for _ in model_data])
        columns = ', '.join(model_data.keys())
        
        conn.execute(f'''
            INSERT OR REPLACE INTO fi_model_results ({columns})
            VALUES ({placeholders})
        ''', list(model_data.values()))
        
        conn.commit()
        conn.close()
        
        print("✅ Результати збережено в базу даних")
        
    def run_complete_analysis(self):
        """
        Повний аналіз з реальними даними
        """
        print("\n🚀 ЗАПУСК ПОВНОГО АНАЛІЗУ З РЕАЛЬНИМИ ДАНИМИ")
        print("=" * 55)
        
        # Step 1: Load real data
        if not self.load_real_amazon_data():
            return False
        
        # Step 2: Load detailed JSON data
        self.load_json_data()
        
        # Step 3: Construct Financial Inclusion Index
        self.construct_real_financial_inclusion_index()
        
        # Step 4: Construct explanatory variables
        self.construct_real_explanatory_variables()
        
        # Step 5: Estimate model
        model = self.estimate_real_data_model()
        
        if model is None:
            return False
        
        # Step 6: Visualize results
        self.create_real_data_visualization()
        
        # Step 7: Save results
        self.save_real_results()
        
        print("\n🎯 АНАЛІЗ ЗАВЕРШЕНО!")
        print("✅ Модель використовує РЕАЛЬНІ дані з PDF звітів Amazon")
        print("📊 Результати збережено та візуалізовано")
        
        return True

if __name__ == "__main__":
    # Initialize and run analysis with real Amazon data
    model = AmazonRealDataFinancialInclusionModel()
    success = model.run_complete_analysis()
    
    if success:
        print("\n🏆 УСПІХ! Економетрична модель з реальними даними готова!")
    else:
        print("\n❌ Помилка в аналізі")
