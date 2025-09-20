"""
Amazon Financial Inclusion Econometric Model
Інноваційна економетрична модель фінансової інклюзії бізнес-екосистеми Amazon

Базована на:
- Структуралістському підході Adner (2017)
- Моделі Adner & Kapoor (2010, 2016) 
- Платформній моделі Jacobides et al. (2018)
- Теорії мережевих ефектів

Автор: Розроблено для дослідження конкурентного проекту фінансової інклюзії
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Econometric libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sqlite3

# Load environment variables
load_dotenv()

class AmazonFinancialInclusionModel:
    """
    Економетрична модель фінансової інклюзії бізнес-екосистеми Amazon
    
    Базова модель:
    FI_Amazon(t) = α + β1·Platform_Scale(t) + β2·Network_Effects(t) + 
                   β3·Digital_Infrastructure(t) + β4·Partner_Integration(t) + 
                   β5·Financial_Services_Diversity(t) + β6·Regulatory_Environment(t) + ε(t)
    """
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.results = {}
        self.financial_inclusion_index = None
        self.explanatory_variables = {}
        
        # Database connection
        self.db_path = 'D:/Econometric/econometric_results.db'
        self.init_database()
        
        print("🏛️ Amazon Financial Inclusion Econometric Model Initialized")
        print("=" * 70)
        print("📊 Базована на теорії Adner & Kapoor (2016)")
        print("🔬 Перше застосування до фінансової інклюзії платформ")
        
    def init_database(self):
        """Initialize database for storing model results"""
        conn = sqlite3.connect(self.db_path)
        
        # Table for Financial Inclusion Index
        conn.execute('''
            CREATE TABLE IF NOT EXISTS financial_inclusion_index (
                date TEXT PRIMARY KEY,
                fi_index REAL,
                platform_scale REAL,
                network_effects REAL,
                digital_infrastructure REAL,
                partner_integration REAL,
                financial_services_diversity REAL,
                regulatory_environment REAL,
                amazon_flywheel_effect REAL,
                competitive_pressure REAL,
                innovation_index REAL
            )
        ''')
        
        # Table for model results
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fi_model_results (
                model_name TEXT PRIMARY KEY,
                date_created TEXT,
                coefficients TEXT,
                r_squared REAL,
                adj_r_squared REAL,
                f_statistic REAL,
                durbin_watson REAL,
                white_test_pvalue REAL,
                model_specification TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def fetch_amazon_ecosystem_data(self):
        """
        Збір даних для побудови компонентів моделі фінансової інклюзії
        """
        print("\n🔍 ЗБІР ДАНИХ ДЛЯ AMAZON FINANCIAL INCLUSION MODEL")
        print("=" * 60)
        
        # Amazon stock and financial data
        print("📈 Завантаження даних Amazon...")
        amzn = yf.Ticker("AMZN")
        self.data['amzn_stock'] = amzn.history(period="10y", interval="1mo")
        self.data['amzn_info'] = amzn.info
        
        # AWS growth proxy (cloud infrastructure)
        print("☁️ Завантаження даних хмарних сервісів...")
        cloud_etf = yf.Ticker("SKYY")  # Cloud Computing ETF as proxy
        self.data['cloud_growth'] = cloud_etf.history(period="10y", interval="1mo")
        
        # E-commerce and digital payments growth
        print("💳 Завантаження даних цифрових платежів...")
        fintech_etf = yf.Ticker("FINX")  # Global X FinTech ETF
        self.data['fintech_growth'] = fintech_etf.history(period="10y", interval="1mo")
        
        # Competition data
        print("🏪 Завантаження даних конкурентів...")
        competitors = ["GOOGL", "MSFT", "AAPL", "META"]  # Major tech platforms
        self.data['competitors'] = {}
        for ticker in competitors:
            self.data['competitors'][ticker] = yf.Ticker(ticker).history(period="10y", interval="1mo")
            
        # Market benchmarks
        print("📊 Завантаження ринкових індексів...")
        sp500 = yf.Ticker("^GSPC")
        nasdaq = yf.Ticker("^IXIC")
        self.data['sp500'] = sp500.history(period="10y", interval="1mo")
        self.data['nasdaq'] = nasdaq.history(period="10y", interval="1mo")
        
        # Economic indicators (using ETFs as proxies)
        print("🌍 Завантаження економічних індикаторів...")
        vix = yf.Ticker("^VIX")  # Market volatility
        dxy = yf.Ticker("DX-Y.NYB")  # Dollar index
        self.data['vix'] = vix.history(period="10y", interval="1mo")
        self.data['dxy'] = dxy.history(period="10y", interval="1mo")
        
        print("✅ Дані успішно завантажені!")
        return self.data
        
    def construct_financial_inclusion_variables(self):
        """
        Побудова змінних для моделі фінансової інклюзії
        """
        print("\n🏗️ ПОБУДОВА ЗМІННИХ МОДЕЛІ")
        print("=" * 45)
        
        # Get common date range
        dates = self.data['amzn_stock'].index
        
        # 1. Platform_Scale(t) - масштаби платформи
        print("1️⃣ Platform Scale - масштаби платформи")
        amzn_close = self.data['amzn_stock']['Close']
        amzn_volume = self.data['amzn_stock']['Volume']
        
        # Market cap proxy and trading activity
        platform_scale = (amzn_close * amzn_volume).rolling(window=3).mean()
        platform_scale = (platform_scale - platform_scale.mean()) / platform_scale.std()
        
        # 2. Network_Effects(t) - мережеві ефекти
        print("2️⃣ Network Effects - мережеві ефекти")
        # Використовуємо кореляцію з ростом e-commerce сектору
        amzn_returns = amzn_close.pct_change()
        sp500_returns = self.data['sp500']['Close'].pct_change()
        
        # Rolling correlation as proxy for network effects
        network_effects = amzn_returns.rolling(window=12).corr(sp500_returns)
        network_effects = network_effects.fillna(network_effects.mean())
        
        # 3. Digital_Infrastructure(t) - цифрова інфраструктура
        print("3️⃣ Digital Infrastructure - цифрова інфраструктура")
        cloud_close = self.data['cloud_growth']['Close'].reindex(dates, method='nearest')
        digital_infrastructure = cloud_close.pct_change().rolling(window=6).mean()
        digital_infrastructure = (digital_infrastructure - digital_infrastructure.mean()) / digital_infrastructure.std()
        
        # 4. Partner_Integration(t) - інтеграція партнерів
        print("4️⃣ Partner Integration - інтеграція партнерів")
        # Proxy через волатильність - більша інтеграція = менша волатільність
        amzn_volatility = amzn_returns.rolling(window=12).std()
        partner_integration = -amzn_volatility  # Inverse relationship
        partner_integration = (partner_integration - partner_integration.mean()) / partner_integration.std()
        
        # 5. Financial_Services_Diversity(t) - різноманітність фінпослуг
        print("5️⃣ Financial Services Diversity - різноманітність фінпослуг")
        fintech_close = self.data['fintech_growth']['Close'].reindex(dates, method='nearest')
        fintech_growth = fintech_close.pct_change().rolling(window=3).mean()
        financial_services_diversity = (fintech_growth - fintech_growth.mean()) / fintech_growth.std()
        
        # 6. Regulatory_Environment(t) - регуляторне середовище
        print("6️⃣ Regulatory Environment - регуляторне середовище")
        # Proxy через VIX - високий VIX = напружене регуляторне середовище
        vix_data = self.data['vix']['Close'].reindex(dates, method='nearest')
        regulatory_environment = -vix_data.rolling(window=6).mean()  # Inverse relationship
        regulatory_environment = (regulatory_environment - regulatory_environment.mean()) / regulatory_environment.std()
        
        # 7. Amazon_Flywheel_Effect(t) - ефект маховика Amazon
        print("7️⃣ Amazon Flywheel Effect - ефект маховика")
        # Composite indicator combining multiple Amazon metrics
        price_momentum = amzn_close.rolling(window=6).mean() / amzn_close.rolling(window=12).mean()
        volume_trend = amzn_volume.rolling(window=3).mean()
        amazon_flywheel_effect = (price_momentum * np.log(volume_trend))
        amazon_flywheel_effect = (amazon_flywheel_effect - amazon_flywheel_effect.mean()) / amazon_flywheel_effect.std()
        
        # 8. Competitive_Pressure(t) - конкурентний тиск
        print("8️⃣ Competitive Pressure - конкурентний тиск")
        # Average performance of competitors relative to Amazon
        competitor_performance = []
        for ticker, data in self.data['competitors'].items():
            comp_close = data['Close'].reindex(dates, method='nearest')
            comp_returns = comp_close.pct_change()
            competitor_performance.append(comp_returns)
        
        avg_competitor_returns = pd.concat(competitor_performance, axis=1).mean(axis=1)
        competitive_pressure = avg_competitor_returns - amzn_returns
        competitive_pressure = competitive_pressure.rolling(window=6).mean()
        competitive_pressure = (competitive_pressure - competitive_pressure.mean()) / competitive_pressure.std()
        
        # 9. Innovation_Index(t) - індекс інноваційності
        print("9️⃣ Innovation Index - індекс інноваційності")
        # Proxy through technology sector performance
        nasdaq_returns = self.data['nasdaq']['Close'].pct_change()
        innovation_index = nasdaq_returns.rolling(window=12).mean()
        innovation_index = (innovation_index - innovation_index.mean()) / innovation_index.std()
        
        # Store all variables
        self.explanatory_variables = {
            'platform_scale': platform_scale,
            'network_effects': network_effects,
            'digital_infrastructure': digital_infrastructure,
            'partner_integration': partner_integration,
            'financial_services_diversity': financial_services_diversity,
            'regulatory_environment': regulatory_environment,
            'amazon_flywheel_effect': amazon_flywheel_effect,
            'competitive_pressure': competitive_pressure,
            'innovation_index': innovation_index
        }
        
        print("✅ Всі змінні успішно побудовані!")
        return self.explanatory_variables
        
    def construct_financial_inclusion_index(self):
        """
        Побудова композитного індексу фінансової інклюзії Amazon
        """
        print("\n📊 ПОБУДОВА FINANCIAL INCLUSION INDEX")
        print("=" * 50)
        
        # Combine multiple proxies for financial inclusion
        amzn_close = self.data['amzn_stock']['Close']
        dates = amzn_close.index
        
        # Components of Financial Inclusion Index
        print("🔧 Компоненти індексу:")
        
        # 1. Market penetration (stock performance relative to market)
        print("   • Ринкове проникнення")
        sp500_close = self.data['sp500']['Close'].reindex(dates, method='nearest')
        market_penetration = (amzn_close / sp500_close).rolling(window=12).mean()
        
        # 2. Service accessibility (inverse of volatility)
        print("   • Доступність сервісів")
        amzn_returns = amzn_close.pct_change()
        service_accessibility = 1 / (amzn_returns.rolling(window=12).std() + 0.001)
        
        # 3. Digital adoption (correlation with fintech growth)
        print("   • Цифрове впровадження")
        fintech_close = self.data['fintech_growth']['Close'].reindex(dates, method='nearest')
        fintech_returns = fintech_close.pct_change()
        digital_adoption = amzn_returns.rolling(window=12).corr(fintech_returns)
        digital_adoption = digital_adoption.fillna(digital_adoption.mean())
        
        # 4. Ecosystem integration (volume-price relationship)
        print("   • Інтеграція екосистеми")
        amzn_volume = self.data['amzn_stock']['Volume']
        ecosystem_integration = np.log(amzn_volume).rolling(window=6).mean()
        
        # Combine components using PCA-like approach
        print("🔢 Комбінування компонентів...")
        
        # Standardize all components
        components = pd.DataFrame({
            'market_penetration': (market_penetration - market_penetration.mean()) / market_penetration.std(),
            'service_accessibility': (service_accessibility - service_accessibility.mean()) / service_accessibility.std(),
            'digital_adoption': (digital_adoption - digital_adoption.mean()) / digital_adoption.std(),
            'ecosystem_integration': (ecosystem_integration - ecosystem_integration.mean()) / ecosystem_integration.std()
        }).fillna(0)
        
        # Create composite index with equal weights (can be optimized)
        weights = [0.3, 0.25, 0.25, 0.2]  # Based on theoretical importance
        
        self.financial_inclusion_index = (
            weights[0] * components['market_penetration'] +
            weights[1] * components['service_accessibility'] +
            weights[2] * components['digital_adoption'] +
            weights[3] * components['ecosystem_integration']
        )
        
        print("✅ Financial Inclusion Index успішно створений!")
        print(f"📈 Період: {self.financial_inclusion_index.index[0]} - {self.financial_inclusion_index.index[-1]}")
        print(f"📊 Спостережень: {len(self.financial_inclusion_index)}")
        
        return self.financial_inclusion_index
        
    def estimate_base_model(self):
        """
        Оцінка базової моделі фінансової інклюзії
        
        FI_Amazon(t) = α + β1·Platform_Scale(t) + β2·Network_Effects(t) + 
                       β3·Digital_Infrastructure(t) + β4·Partner_Integration(t) + 
                       β5·Financial_Services_Diversity(t) + β6·Regulatory_Environment(t) + ε(t)
        """
        print("\n🎯 ОЦІНКА БАЗОВОЇ МОДЕЛІ")
        print("=" * 40)
        
        # Prepare data
        data_df = pd.DataFrame({
            'fi_index': self.financial_inclusion_index,
            'platform_scale': self.explanatory_variables['platform_scale'],
            'network_effects': self.explanatory_variables['network_effects'],
            'digital_infrastructure': self.explanatory_variables['digital_infrastructure'],
            'partner_integration': self.explanatory_variables['partner_integration'],
            'financial_services_diversity': self.explanatory_variables['financial_services_diversity'],
            'regulatory_environment': self.explanatory_variables['regulatory_environment']
        }).dropna()
        
        print(f"📊 Кількість спостережень: {len(data_df)}")
        
        # Dependent variable
        y = data_df['fi_index']
        
        # Independent variables
        X = data_df[['platform_scale', 'network_effects', 'digital_infrastructure',
                    'partner_integration', 'financial_services_diversity', 'regulatory_environment']]
        X = sm.add_constant(X)  # Add intercept
        
        # Estimate model
        print("🔬 Оцінка параметрів...")
        model = sm.OLS(y, X).fit()
        
        # Store results
        self.models['base_model'] = model
        
        # Print results
        print("\n📋 РЕЗУЛЬТАТИ БАЗОВОЇ МОДЕЛІ:")
        print("=" * 50)
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}")
        print(f"Prob (F-statistic): {model.f_pvalue:.6f}")
        
        print("\n📊 КОЕФІЦІЄНТИ:")
        for var, coef, pval in zip(X.columns, model.params, model.pvalues):
            significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"{var:25s}: {coef:8.4f} {significance:3s} (p={pval:.3f})")
        
        # Diagnostic tests
        print("\n🔍 ДІАГНОСТИЧНІ ТЕСТИ:")
        dw_stat = durbin_watson(model.resid)
        print(f"Durbin-Watson: {dw_stat:.4f}")
        
        # White test for heteroscedasticity
        try:
            white_test = het_white(model.resid, model.model.exog)
            print(f"White test p-value: {white_test[1]:.4f}")
        except:
            print("White test: не вдалося обчислити")
        
        return model
        
    def estimate_extended_model(self):
        """
        Оцінка розширеної моделі з лаговими змінними та додатковими факторами
        
        FI_Amazon(t) = α + Σβi·X(t-i) + γ·Amazon_Flywheel_Effect(t) + 
                       δ·Competitive_Pressure(t) + λ·Innovation_Index(t) + ε(t)
        """
        print("\n🚀 ОЦІНКА РОЗШИРЕНОЇ МОДЕЛІ")
        print("=" * 45)
        
        # Prepare data with lags
        data_df = pd.DataFrame({
            'fi_index': self.financial_inclusion_index,
            'platform_scale': self.explanatory_variables['platform_scale'],
            'network_effects': self.explanatory_variables['network_effects'],
            'digital_infrastructure': self.explanatory_variables['digital_infrastructure'],
            'partner_integration': self.explanatory_variables['partner_integration'],
            'financial_services_diversity': self.explanatory_variables['financial_services_diversity'],
            'regulatory_environment': self.explanatory_variables['regulatory_environment'],
            'amazon_flywheel_effect': self.explanatory_variables['amazon_flywheel_effect'],
            'competitive_pressure': self.explanatory_variables['competitive_pressure'],
            'innovation_index': self.explanatory_variables['innovation_index']
        })
        
        # Add lagged variables
        print("⏰ Додавання лагових змінних...")
        for lag in [1, 3, 6]:  # 1, 3, 6 month lags
            for var in ['platform_scale', 'network_effects', 'digital_infrastructure']:
                data_df[f'{var}_lag{lag}'] = data_df[var].shift(lag)
        
        data_df = data_df.dropna()
        print(f"📊 Кількість спостережень: {len(data_df)}")
        
        # Dependent variable
        y = data_df['fi_index']
        
        # Independent variables (including lags and new factors)
        independent_vars = [
            'platform_scale', 'network_effects', 'digital_infrastructure',
            'partner_integration', 'financial_services_diversity', 'regulatory_environment',
            'amazon_flywheel_effect', 'competitive_pressure', 'innovation_index',
            'platform_scale_lag1', 'network_effects_lag1', 'digital_infrastructure_lag1',
            'platform_scale_lag3', 'network_effects_lag3', 'digital_infrastructure_lag3'
        ]
        
        X = data_df[independent_vars]
        X = sm.add_constant(X)  # Add intercept
        
        # Estimate model
        print("🔬 Оцінка параметрів розширеної моделі...")
        model = sm.OLS(y, X).fit()
        
        # Store results
        self.models['extended_model'] = model
        
        # Print results
        print("\n📋 РЕЗУЛЬТАТИ РОЗШИРЕНОЇ МОДЕЛІ:")
        print("=" * 55)
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}")
        print(f"Prob (F-statistic): {model.f_pvalue:.6f}")
        
        # Show most significant coefficients
        print("\n📊 НАЙБІЛЬШ ЗНАЧУЩІ КОЕФІЦІЄНТИ:")
        coef_df = pd.DataFrame({
            'Variable': X.columns,
            'Coefficient': model.params,
            'P-value': model.pvalues,
            'Significance': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' 
                           for p in model.pvalues]
        }).sort_values('P-value')
        
        for _, row in coef_df.head(10).iterrows():
            print(f"{row['Variable']:30s}: {row['Coefficient']:8.4f} {row['Significance']:3s}")
        
        return model
        
    def save_results_to_database(self):
        """Збереження результатів в базу даних"""
        print("\n💾 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
        print("=" * 35)
        
        conn = sqlite3.connect(self.db_path)
        
        # Save Financial Inclusion Index and variables
        data_to_save = pd.DataFrame({
            'date': self.financial_inclusion_index.index.strftime('%Y-%m-%d'),
            'fi_index': self.financial_inclusion_index.values,
            'platform_scale': self.explanatory_variables['platform_scale'].reindex(self.financial_inclusion_index.index).values,
            'network_effects': self.explanatory_variables['network_effects'].reindex(self.financial_inclusion_index.index).values,
            'digital_infrastructure': self.explanatory_variables['digital_infrastructure'].reindex(self.financial_inclusion_index.index).values,
            'partner_integration': self.explanatory_variables['partner_integration'].reindex(self.financial_inclusion_index.index).values,
            'financial_services_diversity': self.explanatory_variables['financial_services_diversity'].reindex(self.financial_inclusion_index.index).values,
            'regulatory_environment': self.explanatory_variables['regulatory_environment'].reindex(self.financial_inclusion_index.index).values,
            'amazon_flywheel_effect': self.explanatory_variables['amazon_flywheel_effect'].reindex(self.financial_inclusion_index.index).values,
            'competitive_pressure': self.explanatory_variables['competitive_pressure'].reindex(self.financial_inclusion_index.index).values,
            'innovation_index': self.explanatory_variables['innovation_index'].reindex(self.financial_inclusion_index.index).values
        })
        
        data_to_save.to_sql('financial_inclusion_index', conn, if_exists='replace', index=False)
        
        # Save model results
        if 'base_model' in self.models:
            model = self.models['base_model']
            model_results = {
                'model_name': 'base_financial_inclusion_model',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'coefficients': str(model.params.to_dict()),
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'durbin_watson': durbin_watson(model.resid),
                'white_test_pvalue': None,  # Will add if calculated
                'model_specification': 'FI_Amazon(t) = α + β1·Platform_Scale(t) + β2·Network_Effects(t) + β3·Digital_Infrastructure(t) + β4·Partner_Integration(t) + β5·Financial_Services_Diversity(t) + β6·Regulatory_Environment(t) + ε(t)'
            }
            
            pd.DataFrame([model_results]).to_sql('fi_model_results', conn, if_exists='append', index=False)
        
        conn.close()
        print("✅ Результати збережені в базу даних!")
        
    def generate_comprehensive_report(self):
        """Генерація комплексного звіту"""
        print("\n📄 ГЕНЕРАЦІЯ КОМПЛЕКСНОГО ЗВІТУ")
        print("=" * 45)
        
        report = []
        report.append("# AMAZON FINANCIAL INCLUSION ECONOMETRIC MODEL")
        report.append("## Інноваційна економетрична модель фінансової інклюзії бізнес-екосистеми Amazon")
        report.append("")
        report.append("### Теоретична база:")
        report.append("- Структуралістський підхід Adner (2017)")
        report.append("- Модель Adner & Kapoor (2010, 2016)")
        report.append("- Платформна модель Jacobides et al. (2018)")
        report.append("")
        
        if self.financial_inclusion_index is not None:
            report.append("### Financial Inclusion Index Statistics:")
            report.append(f"- Період аналізу: {self.financial_inclusion_index.index[0]} - {self.financial_inclusion_index.index[-1]}")
            report.append(f"- Кількість спостережень: {len(self.financial_inclusion_index)}")
            report.append(f"- Середнє значення: {self.financial_inclusion_index.mean():.4f}")
            report.append(f"- Стандартне відхилення: {self.financial_inclusion_index.std():.4f}")
            report.append("")
        
        if 'base_model' in self.models:
            model = self.models['base_model']
            report.append("### Базова модель результати:")
            report.append(f"- R-squared: {model.rsquared:.4f}")
            report.append(f"- Adj. R-squared: {model.rsquared_adj:.4f}")
            report.append(f"- F-statistic: {model.fvalue:.4f}")
            report.append("")
            
            report.append("### Ключові висновки:")
            # Analyze coefficients
            significant_vars = []
            for var, pval in zip(model.params.index, model.pvalues):
                if pval < 0.05 and var != 'const':
                    significant_vars.append(var)
            
            report.append(f"- Статистично значущі фактори: {', '.join(significant_vars)}")
            report.append("")
        
        if 'extended_model' in self.models:
            model = self.models['extended_model']
            report.append("### Розширена модель результати:")
            report.append(f"- R-squared: {model.rsquared:.4f}")
            report.append(f"- Adj. R-squared: {model.rsquared_adj:.4f}")
            report.append("")
        
        report.append("### Наукова цінність:")
        report.append("- Перше комплексне економетричне моделювання фінансової інклюзії великої платформної екосистеми")
        report.append("- Інноваційне поєднання класичних факторів бізнес-екосистем із специфічними змінними фінансової інклюзії")
        report.append("- Розробка моделі з урахуванням часової динаміки через лагові змінні")
        report.append("- Практична значущість для розробки стратегій фінансових послуг у глобальних корпораціях")
        
        # Save report
        report_text = "\n".join(report)
        
        with open('D:/Econometric/results/amazon_financial_inclusion_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✅ Звіт збережений: D:/Econometric/results/amazon_financial_inclusion_report.md")
        
        return report_text

if __name__ == "__main__":
    # Initialize the model
    model = AmazonFinancialInclusionModel()
    
    print("🚀 Запуск повного аналізу Amazon Financial Inclusion Model...")
    
    # Step 1: Fetch data
    model.fetch_amazon_ecosystem_data()
    
    # Step 2: Construct variables
    model.construct_financial_inclusion_variables()
    
    # Step 3: Construct Financial Inclusion Index
    model.construct_financial_inclusion_index()
    
    # Step 4: Estimate models
    model.estimate_base_model()
    model.estimate_extended_model()
    
    # Step 5: Save results
    model.save_results_to_database()
    
    # Step 6: Generate report
    model.generate_comprehensive_report()
    
    print("\n🎉 AMAZON FINANCIAL INCLUSION MODEL - ЗАВЕРШЕНО!")
    print("=" * 55)
    print("✅ Інноваційна модель успішно реалізована")
    print("📊 Результати збережені в базу даних")
    print("📄 Звіт згенеровано")
    print("🔬 Готово для наукового аналізу!")
