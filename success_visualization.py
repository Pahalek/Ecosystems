"""
🎯 ВІЗУАЛІЗАЦІЯ ДОСЯГНЕННЯ 89.3% РЕАЛЬНИХ ДАНИХ
Amazon Financial Inclusion Model - Enhanced Version

Створення графіків та порівняльного аналізу
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_achievement_visualizations():
    """Створення візуалізацій досягнення цілі"""
    
    # Read results
    try:
        with open('D:/Econometric/enhanced_80_percent_model_results.json', 'r') as f:
            enhanced_results = json.load(f)
        
        with open('D:/Econometric/final_model_results.json', 'r') as f:
            base_results = json.load(f)
    except:
        print("❌ Файли результатів не знайдено")
        return
    
    # Create comprehensive dashboard with larger size to prevent overlapping
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Data Composition Comparison
    ax1 = plt.subplot(3, 4, 1)
    
    models = ['Базова модель\n(71.6%)', 'Enhanced модель\n(89.3%)']
    real_data = [71.6, 89.3]
    synthetic_data = [28.4, 10.7]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x, real_data, width, label='Реальні дані', color='#2E8B57', alpha=0.8)
    bars2 = ax1.bar(x, synthetic_data, width, bottom=real_data, label='Синтетичні дані', color='#CD5C5C', alpha=0.8)
    
    ax1.set_ylabel('Відсоток (%)', fontsize=10)
    ax1.set_title('🎯 ДОСЯГНЕННЯ ЦІЛІ:\n80%+ РЕАЛЬНИХ ДАНИХ', fontweight='bold', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Ціль: 80%')
    
    # Add percentage labels
    for i, (real, synthetic) in enumerate(zip(real_data, synthetic_data)):
        ax1.text(i, real/2, f'{real}%', ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax1.text(i, real + synthetic/2, f'{synthetic}%', ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    
    # 2. Model Quality Comparison
    ax2 = plt.subplot(3, 4, 2)
    
    metrics = ['R²', 'Adj. R²', 'F-stat/10']
    base_values = [base_results['model_statistics']['r_squared'], 
                   base_results['model_statistics']['adj_r_squared'],
                   base_results['model_statistics']['f_statistic']/10]
    enhanced_values = [enhanced_results['model_statistics']['r_squared'],
                       enhanced_results['model_statistics']['adj_r_squared'], 
                       enhanced_results['model_statistics']['f_statistic']/10]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, base_values, width, label='Базова', color='#4682B4', alpha=0.7)
    ax2.bar(x + width/2, enhanced_values, width, label='Enhanced', color='#32CD32', alpha=0.7)
    
    ax2.set_ylabel('Значення', fontsize=10)
    ax2.set_title('📊 ЯКІСТЬ МОДЕЛІ', fontweight='bold', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.legend(fontsize=8)
    
    # 3. Data Sources Growth
    ax3 = plt.subplot(3, 4, 3)
    
    sources = ['Amazon\nPDF', 'Market\nData', 'FRED\nEconomic', 'International', 'Alternative\nSectors']
    base_components = [4, 48, 6, 0, 0]  # Estimated from base model
    enhanced_components = [4, 64, 22, 20, 36]
    
    x = np.arange(len(sources))
    width = 0.35
    
    ax3.bar(x - width/2, base_components, width, label='Базова', color='#4682B4', alpha=0.7)
    ax3.bar(x + width/2, enhanced_components, width, label='Enhanced', color='#32CD32', alpha=0.7)
    
    ax3.set_ylabel('Кількість\nкомпонентів', fontsize=10)
    ax3.set_title('📈 РОЗШИРЕННЯ\nДЖЕРЕЛ ДАНИХ', fontweight='bold', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
    ax3.legend(fontsize=8)
    
    # 4. Achievement Timeline
    ax4 = plt.subplot(3, 4, 4)
    
    timeline_dates = ['Початкова\nмодель', 'Реальні дані\nз PDF', 'Фінальна\nмодель', 'Enhanced\nмодель']
    timeline_percentages = [50, 73.1, 71.6, 89.3]
    colors = ['#CD5C5C', '#FFA500', '#4682B4', '#32CD32']
    
    bars = ax4.bar(timeline_dates, timeline_percentages, color=colors, alpha=0.8)
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Ціль: 80%')
    ax4.set_ylabel('% Реальних\nданих', fontsize=10)
    ax4.set_title('⏰ ПРОГРЕС\nДОСЯГНЕННЯ', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=8)
    
    for bar, pct in zip(bars, timeline_percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Significant Variables - Base Model
    ax5 = plt.subplot(3, 4, 5)
    
    base_vars = ['financial_services_quality', 'competitive_position', 'platform_scale']
    base_coeffs = [0.2027, 0.1630, 0.0617]
    base_p_values = [0.000, 0.000, 0.053]
    
    colors_base = ['green' if p < 0.05 else 'orange' for p in base_p_values]
    bars = ax5.barh(base_vars, base_coeffs, color=colors_base, alpha=0.7)
    ax5.set_xlabel('Коефіцієнт', fontsize=10)
    ax5.set_title('📊 БАЗОВА МОДЕЛЬ\nЗначущі змінні', fontweight='bold', fontsize=11)
    ax5.tick_params(axis='both', which='major', labelsize=9)
    
    # 6. Significant Variables - Enhanced Model  
    ax6 = plt.subplot(3, 4, 6)
    
    enh_vars = ['network_effects', 'financial_services_quality', 'competitive_position']
    enh_coeffs = [-0.2307, 0.1694, 0.0898]
    enh_p_values = [0.000, 0.000, 0.013]
    
    colors_enh = ['green' if p < 0.05 else 'orange' for p in enh_p_values]
    bars = ax6.barh(enh_vars, enh_coeffs, color=colors_enh, alpha=0.7)
    ax6.set_xlabel('Коефіцієнт', fontsize=10)
    ax6.set_title('🚀 ENHANCED МОДЕЛЬ\nЗначущі змінні', fontweight='bold', fontsize=11)
    ax6.tick_params(axis='both', which='major', labelsize=9)
    
    # 7. Data Type Distribution - Enhanced
    ax7 = plt.subplot(3, 4, 7)
    
    data_types = ['Amazon\nCore', 'Market\nData', 'FRED/Economic', 'International', 'Alternative', 'Synthetic\nOps']
    components = [4, 64, 22, 20, 36, 17]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    ax7.pie(components, labels=data_types, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90, textprops={'fontsize': 8})
    ax7.set_title('📊 РОЗПОДІЛ КОМПОНЕНТІВ\nEnhanced Model', fontweight='bold', fontsize=11)
    
    # 8. Real vs Synthetic Evolution
    ax8 = plt.subplot(3, 4, 8)
    
    models_evo = ['Початкова', 'PDF модель', 'Фінальна', 'Enhanced']
    real_evo = [50, 73.1, 71.6, 89.3]
    synthetic_evo = [50, 26.9, 28.4, 10.7]
    
    ax8.plot(models_evo, real_evo, marker='o', linewidth=3, markersize=8, 
             color='green', label='Реальні дані')
    ax8.plot(models_evo, synthetic_evo, marker='s', linewidth=3, markersize=8, 
             color='red', label='Синтетичні дані')
    ax8.axhline(y=80, color='blue', linestyle='--', alpha=0.7, label='Ціль: 80%')
    
    ax8.set_ylabel('Відсоток (%)', fontsize=10)
    ax8.set_title('📈 ЕВОЛЮЦІЯ\nСПІВВІДНОШЕННЯ', fontweight='bold', fontsize=11)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='both', which='major', labelsize=9)
    
    # 9. Statistical Quality Metrics
    ax9 = plt.subplot(3, 4, 9)
    
    metrics_detailed = ['R²', 'Adj. R²', 'AIC', 'BIC', 'D-W']
    base_detailed = [0.6389, 0.5947, -4.94, 9.23, 1.014]
    enhanced_detailed = [0.6400, 0.5960, 5.97, 20.15, 1.116]
    
    # Normalize for better visualization
    base_norm = [x/max(abs(x), 1) for x in base_detailed]
    enh_norm = [x/max(abs(x), 1) for x in enhanced_detailed]
    
    x = np.arange(len(metrics_detailed))
    width = 0.35
    
    ax9.bar(x - width/2, base_norm, width, label='Базова', color='#4682B4', alpha=0.7)
    ax9.bar(x + width/2, enh_norm, width, label='Enhanced', color='#32CD32', alpha=0.7)
    
    ax9.set_ylabel('Нормалізоване\nзначення', fontsize=10)
    ax9.set_title('📊 СТАТИСТИЧНІ\nМЕТРИКИ', fontweight='bold', fontsize=11)
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_detailed, fontsize=9)
    ax9.legend(fontsize=8)
    
    # 10. Components Growth by Category
    ax10 = plt.subplot(3, 4, 10)
    
    categories = ['Core\nFinancial', 'Market\nIndices', 'Economic\nIndicators', 'Global\nData', 'Innovation\nSectors']
    growth = [0, 20, 22, 20, 36]  # New components added
    
    bars = ax10.bar(categories, growth, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'], 
                    alpha=0.8)
    ax10.set_ylabel('Додані\nкомпоненти', fontsize=10)
    ax10.set_title('📈 РОЗШИРЕННЯ\nПО КАТЕГОРІЯХ', fontweight='bold', fontsize=11)
    
    for bar, growth_val in zip(bars, growth):
        if growth_val > 0:
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'+{growth_val}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax10.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    
    # 11. Success Metrics Dashboard - Fixed overlapping text
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    success_text = """🏆 ДОСЯГНЕННЯ УСПІХУ

✅ ЦІЛЬ ПЕРЕВИЩЕНО
   89.3% vs 80% (+9.3%)

📊 ЯКІСТЬ ЗБЕРЕЖЕНО  
   R² = 0.6400 (+0.0011)

🔬 СТАТИСТИЧНА ЗНАЧУЩІСТЬ
   F-stat = 14.52***

📈 РОЗШИРЕННЯ ДАНИХ
   +84 реальних компонентів
   
⚡ СИНТЕТИЧНІ МІНІМІЗОВАНО
   28.4% → 10.7% (-17.7%)
   
🌍 МІЖНАРОДНА ІНТЕГРАЦІЯ
   +20 глобальних компонентів
   
🔍 ІННОВАЦІЇ ДОДАНО
   +36 альтернативних секторів"""
    
    ax11.text(0.05, 0.95, success_text, transform=ax11.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgreen", alpha=0.7), family='monospace')
    
    # 12. Target Achievement Gauge - Fixed layout
    ax12 = plt.subplot(3, 4, 12)
    
    # Create gauge chart
    achieved = 89.3
    target = 80.0
    
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Background gauge
    ax12.plot(theta, r, 'lightgray', linewidth=8)
    
    # Target line
    target_angle = (target/100) * np.pi
    ax12.plot([target_angle, target_angle], [0, 1], 'red', linewidth=3, label=f'Ціль: {target}%')
    
    # Achievement arc
    achieved_angle = (achieved/100) * np.pi
    theta_achieved = np.linspace(0, achieved_angle, 100)
    ax12.plot(theta_achieved, r[:len(theta_achieved)], 'green', linewidth=8, label=f'Досягнуто: {achieved}%')
    
    ax12.set_xlim(-0.2, np.pi + 0.2)
    ax12.set_ylim(-0.2, 1.2)
    ax12.set_aspect('equal')
    ax12.axis('off')
    ax12.set_title('🎯 GAUGE ДОСЯГНЕННЯ', fontweight='bold', fontsize=11)
    
    # Add center text with better positioning
    ax12.text(np.pi/2, 0.4, f'{achieved}%\nDONE!', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='green')
    
    plt.tight_layout(pad=2.0)
    plt.suptitle('🏆 AMAZON FINANCIAL INCLUSION MODEL: УСПІШНЕ ДОСЯГНЕННЯ 89.3% РЕАЛЬНИХ ДАНИХ', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save plot
    plt.savefig('D:/Econometric/Ecosystems/SUCCESS_ACHIEVEMENT_DASHBOARD.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Візуалізація успіху збережена: SUCCESS_ACHIEVEMENT_DASHBOARD.png")
    
    plt.show()

def create_summary_table():
    """Створення підсумкової таблиці результатів"""
    
    summary_data = {
        'Метрика': [
            'Реальні дані (%)',
            'Синтетичні дані (%)', 
            'Кількість реальних компонентів',
            'R²',
            'Adj. R²',
            'F-statistic',
            'P-value (F-test)',
            'AIC',
            'BIC',
            'Durbin-Watson',
            'Спостережень'
        ],
        'Базова модель': [
            '71.6%',
            '28.4%',
            '58',
            '0.6389',
            '0.5947', 
            '14.45',
            '< 0.001',
            '-4.94',
            '9.23',
            '1.014',
            '56'
        ],
        'Enhanced модель': [
            '89.3%',
            '10.7%',
            '142',
            '0.6400',
            '0.5960',
            '14.52',
            '< 0.001',
            '5.97',
            '20.15',
            '1.116',
            '56'
        ],
        'Покращення': [
            '+17.7%',
            '-17.7%',
            '+84',
            '+0.0011',
            '+0.0013',
            '+0.07',
            'Stable',
            '+10.91',
            '+10.92',
            '+0.102',
            '0'
        ],
        'Статус': [
            '✅ ЦІЛЬ ДОСЯГНУТО',
            '✅ МІНІМІЗОВАНО',
            '✅ РОЗШИРЕНО',
            '✅ ЗБЕРЕЖЕНО',
            '✅ ЗБЕРЕЖЕНО',
            '✅ ЗБЕРЕЖЕНО',
            '✅ ЗНАЧУЩИЙ',
            '⚠️ Підвищено',
            '⚠️ Підвищено', 
            '✅ НОРМАЛЬНО',
            '✅ СТАБІЛЬНО'
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    print("\n📊 ПІДСУМКОВА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('D:/Econometric/Ecosystems/ENHANCED_MODEL_COMPARISON.csv', index=False)
    print("\n💾 Таблиця збережена: ENHANCED_MODEL_COMPARISON.csv")

if __name__ == "__main__":
    print("🎯 СТВОРЕННЯ ВІЗУАЛІЗАЦІЇ ДОСЯГНЕННЯ 89.3% РЕАЛЬНИХ ДАНИХ")
    print("=" * 60)
    
    create_achievement_visualizations()
    create_summary_table()
    
    print("\n🏆 ВІЗУАЛІЗАЦІЯ УСПІХУ ЗАВЕРШЕНА!")
    print("📊 Всі графіки та таблиці створено")
    print("✅ ЦІЛЬ 80%+ РЕАЛЬНИХ ДАНИХ ПЕРЕВИЩЕНО: 89.3%")
