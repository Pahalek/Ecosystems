"""
PDF Data Extractor for Amazon Financial Reports
Витяг реальних даних з PDF звітів Amazon для економетричної моделі

Обробляє:
- Earnings releases
- Shareholder letters  
- 10-Q quarterly reports
- Annual reports
"""

import os
import re
import pandas as pd
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
import numpy as np
from datetime import datetime
import json

class AmazonPDFDataExtractor:
    """
    Клас для витягу структурованих даних з PDF звітів Amazon
    """
    
    def __init__(self, pdf_directory="D:/Econometric/investor_reports"):
        self.pdf_dir = Path(pdf_directory)
        self.extracted_data = {}
        self.financial_metrics = {}
        
        print("📄 Amazon PDF Data Extractor Initialized")
        print(f"📁 PDF Directory: {self.pdf_dir}")
        
    def scan_pdf_files(self):
        """Сканування всіх PDF файлів у директорії"""
        pdf_files = []
        for root, dirs, files in os.walk(self.pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(Path(root) / file)
        
        print(f"📊 Знайдено {len(pdf_files)} PDF файлів")
        return pdf_files
    
    def extract_earnings_data(self, pdf_path):
        """
        Витяг даних з earnings releases
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Парсинг ключових метрик
            metrics = {}
            
            # Net Sales / Revenue
            revenue_pattern = r'Net sales.*?(\$[\d,]+\.?\d*)\s*(?:million|billion)?'
            revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
            if revenue_match:
                metrics['revenue'] = self._parse_financial_amount(revenue_match.group(1))
            
            # Operating Income
            op_income_pattern = r'Operating income.*?(\$[\d,]+\.?\d*)\s*(?:million|billion)?'
            op_income_match = re.search(op_income_pattern, text, re.IGNORECASE)
            if op_income_match:
                metrics['operating_income'] = self._parse_financial_amount(op_income_match.group(1))
            
            # AWS Revenue
            aws_pattern = r'AWS.*?(\$[\d,]+\.?\d*)\s*(?:million|billion)?'
            aws_match = re.search(aws_pattern, text, re.IGNORECASE)
            if aws_match:
                metrics['aws_revenue'] = self._parse_financial_amount(aws_match.group(1))
            
            # Free Cash Flow
            fcf_pattern = r'Free cash flow.*?(\$[\d,]+\.?\d*)\s*(?:million|billion)?'
            fcf_match = re.search(fcf_pattern, text, re.IGNORECASE)
            if fcf_match:
                metrics['free_cash_flow'] = self._parse_financial_amount(fcf_match.group(1))
            
            # Extract quarter/year
            date_pattern = r'Q([1-4])\s*(\d{4})|(\d{4})\s*Q([1-4])'
            date_match = re.search(date_pattern, text)
            if date_match:
                quarter = date_match.group(1) or date_match.group(4)
                year = date_match.group(2) or date_match.group(3)
                metrics['period'] = f"Q{quarter}-{year}"
            
            return metrics
            
        except Exception as e:
            print(f"❌ Помилка обробки {pdf_path}: {e}")
            return {}
    
    def extract_10q_data(self, pdf_path):
        """
        Витяг даних з 10-Q quarterly reports
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages[:10]:  # Перші 10 сторінок зазвичай містять фін. дані
                    text += page.extract_text() or ""
            
            metrics = {}
            
            # Total Revenue
            revenue_patterns = [
                r'Total net sales.*?(\$[\d,]+)',
                r'Net product sales.*?(\$[\d,]+)',
                r'Revenue.*?(\$[\d,]+)'
            ]
            
            for pattern in revenue_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics['total_revenue'] = self._parse_financial_amount(match.group(1))
                    break
            
            # Segment revenues
            segments = {
                'North America': r'North America.*?(\$[\d,]+)',
                'International': r'International.*?(\$[\d,]+)',
                'AWS': r'Amazon Web Services.*?(\$[\d,]+)'
            }
            
            for segment, pattern in segments.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics[f'{segment.lower()}_revenue'] = self._parse_financial_amount(match.group(1))
            
            # Operating expenses
            expense_patterns = {
                'cost_of_sales': r'Cost of sales.*?(\$[\d,]+)',
                'fulfillment': r'Fulfillment.*?(\$[\d,]+)',
                'technology_content': r'Technology and content.*?(\$[\d,]+)',
                'marketing': r'Sales and marketing.*?(\$[\d,]+)'
            }
            
            for expense_type, pattern in expense_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics[expense_type] = self._parse_financial_amount(match.group(1))
            
            return metrics
            
        except Exception as e:
            print(f"❌ Помилка обробки 10-Q {pdf_path}: {e}")
            return {}
    
    def extract_shareholder_letter_data(self, pdf_path):
        """
        Витяг стратегічної інформації з листів акціонерам
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            insights = {}
            
            # Key strategic metrics mentions
            strategic_keywords = {
                'prime_members': r'Prime members.*?(\d+\.?\d*)\s*million',
                'aws_customers': r'AWS.*?customers.*?(\d+)',
                'fulfillment_centers': r'fulfillment centers.*?(\d+)',
                'employees': r'employees.*?(\d+,?\d*)',
                'countries': r'countries.*?(\d+)',
                'marketplace_sellers': r'sellers.*?(\d+\.?\d*)\s*million'
            }
            
            for metric, pattern in strategic_keywords.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    insights[metric] = float(match.group(1).replace(',', ''))
            
            # Innovation themes count
            innovation_themes = [
                'machine learning', 'artificial intelligence', 'alexa',
                'drone', 'autonomous', 'sustainability', 'climate',
                'innovation', 'invention', 'customer obsession'
            ]
            
            innovation_count = 0
            for theme in innovation_themes:
                innovation_count += len(re.findall(theme, text, re.IGNORECASE))
            
            insights['innovation_mentions'] = innovation_count
            
            return insights
            
        except Exception as e:
            print(f"❌ Помилка обробки shareholder letter {pdf_path}: {e}")
            return {}
    
    def _parse_financial_amount(self, amount_str):
        """Парсинг фінансових сум з тексту"""
        # Remove $ and commas
        clean_amount = re.sub(r'[$,]', '', amount_str)
        try:
            return float(clean_amount)
        except (ValueError, TypeError):
            return 0.0
    
    def process_all_pdfs(self):
        """
        Обробка всіх PDF файлів та створення структурованого датасету
        """
        print("\n🔄 ОБРОБКА ВСІХ PDF ФАЙЛІВ")
        print("=" * 40)
        
        pdf_files = self.scan_pdf_files()
        all_data = {}
        
        for pdf_path in pdf_files:
            print(f"📄 Обробка: {pdf_path.name}")
            
            # Визначення типу звіту
            filename = pdf_path.name.lower()
            
            if 'earnings' in filename or 'q1' in filename or 'q2' in filename or 'q3' in filename or 'q4' in filename:
                data = self.extract_earnings_data(pdf_path)
                report_type = 'earnings'
            elif '10q' in filename:
                data = self.extract_10q_data(pdf_path)
                report_type = '10q'
            elif 'shareholder' in filename or 'letter' in filename:
                data = self.extract_shareholder_letter_data(pdf_path)
                report_type = 'shareholder_letter'
            else:
                continue
            
            if data:
                all_data[pdf_path.name] = {
                    'type': report_type,
                    'data': data,
                    'file_path': str(pdf_path)
                }
        
        self.extracted_data = all_data
        print(f"✅ Обробано {len(all_data)} файлів")
        
        return all_data
    
    def create_time_series_dataset(self):
        """
        Створення часового ряду з вилучених даних
        """
        print("\n📊 СТВОРЕННЯ ЧАСОВОГО РЯДУ")
        print("=" * 35)
        
        time_series_data = []
        
        for filename, file_data in self.extracted_data.items():
            if file_data['type'] in ['earnings', '10q']:
                # Extract date from filename or content
                date_match = re.search(r'(\d{4})', filename)
                quarter_match = re.search(r'q([1-4])', filename.lower())
                
                if date_match and quarter_match:
                    year = int(date_match.group(1))
                    quarter = int(quarter_match.group(1))
                    
                    # Create date (end of quarter)
                    quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
                    date = pd.Timestamp(year, quarter_end_months[quarter], 28)
                    
                    row = {'date': date, 'quarter': f"Q{quarter}-{year}"}
                    row.update(file_data['data'])
                    time_series_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(time_series_data)
        if not df.empty:
            df = df.sort_values('date').set_index('date')
        
        print(f"📈 Створено часовий ряд з {len(df)} спостережень")
        return df
    
    def save_extracted_data(self, output_path="D:/Econometric/amazon_real_data.json"):
        """Збереження вилучених даних"""
        with open(output_path, 'w') as f:
            # Convert any datetime objects to strings for JSON serialization
            serializable_data = {}
            for key, value in self.extracted_data.items():
                serializable_data[key] = value
            
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"💾 Дані збережено в {output_path}")

if __name__ == "__main__":
    # Ініціалізація та обробка
    extractor = AmazonPDFDataExtractor()
    
    # Обробка всіх PDF
    extracted_data = extractor.process_all_pdfs()
    
    # Створення часового ряду
    time_series = extractor.create_time_series_dataset()
    
    # Збереження результатів
    extractor.save_extracted_data()
    
    if not time_series.empty:
        time_series.to_csv("D:/Econometric/amazon_real_financial_data.csv")
        print("📊 Часовий ряд збережено в amazon_real_financial_data.csv")
    
    print("\n🎯 ГОТОВО! Реальні дані з PDF вилучено та структуровано.")
