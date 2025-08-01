"""
Web scraping utilities for the Data Analyst Agent
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import aiohttp
import logging
from typing import Optional, List, Dict, Any
import re
import time
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class WebScraper:
    """Handles web scraping tasks"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    async def scrape_wikipedia_table(self, url: str, table_index: int = 0) -> Optional[pd.DataFrame]:
        """Scrape tables from Wikipedia pages"""
        try:
            logger.info(f"Scraping Wikipedia table from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all tables
            tables = soup.find_all('table', class_='wikitable')
            
            if not tables:
                # Try other table classes
                tables = soup.find_all('table')
                tables = [t for t in tables if t.get('class') and any('table' in cls.lower() for cls in t.get('class'))]
            
            if not tables:
                logger.error("No tables found on the page")
                return None
            
            if table_index >= len(tables):
                logger.error(f"Table index {table_index} out of range. Found {len(tables)} tables.")
                table_index = 0
            
            table = tables[table_index]
            
            # Extract table data
            headers = []
            rows = []
            
            # Get headers
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    header_text = th.get_text(strip=True)
                    headers.append(header_text)
            
            # Get data rows
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text(strip=True)
                    # Clean up cell text
                    cell_text = re.sub(r'\[.*?\]', '', cell_text)  # Remove citation brackets
                    cell_text = re.sub(r'\n+', ' ', cell_text)  # Replace newlines with spaces
                    row.append(cell_text)
                
                if row:  # Only add non-empty rows
                    rows.append(row)
            
            if not headers or not rows:
                logger.error("No valid table data found")
                return None
            
            # Create DataFrame
            # Ensure all rows have the same length as headers
            max_cols = len(headers)
            cleaned_rows = []
            for row in rows:
                # Pad or truncate row to match header length
                if len(row) < max_cols:
                    row.extend([''] * (max_cols - len(row)))
                elif len(row) > max_cols:
                    row = row[:max_cols]
                cleaned_rows.append(row)
            
            df = pd.DataFrame(cleaned_rows, columns=headers)
            
            logger.info(f"Successfully scraped table with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia table: {e}")
            return None
    
    async def scrape_website_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape general website content"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic content
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            # Extract main content (try various selectors)
            content_selectors = [
                'main', '[role="main"]', '.content', '#content', 
                '.main-content', 'article', '.article'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    main_content = content_element.get_text(strip=True)
                    break
            
            if not main_content:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    main_content = body.get_text(strip=True)
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                if href and text:
                    full_url = urljoin(url, href)
                    links.append({'text': text, 'url': full_url})
            
            return {
                'url': url,
                'title': title_text,
                'content': main_content[:5000],  # Limit content length
                'links': links[:50]  # Limit number of links
            }
            
        except Exception as e:
            logger.error(f"Error scraping website content: {e}")
            return None
    
    async def scrape_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                task = self._scrape_url_async(session, url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        return valid_results
    
    async def _scrape_url_async(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Async helper for scraping a single URL"""
        try:
            async with session.get(url, timeout=30) as response:
                content = await response.text()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                title = soup.find('title')
                title_text = title.get_text(strip=True) if title else ""
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text_content = soup.get_text()
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': text_content[:2000],  # Limit content
                    'status': response.status
                }
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {'url': url, 'error': str(e)}
    
    def extract_financial_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract financial data from financial websites"""
        try:
            financial_data = {}
            
            # Look for common financial data patterns
            price_patterns = [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*USD',
                r'Price:\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
            ]
            
            page_text = soup.get_text()
            
            for pattern in price_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    financial_data['prices'] = matches[:10]  # Limit to first 10 matches
                    break
            
            # Look for percentage changes
            pct_pattern = r'([+-]?\d+(?:\.\d+)?%)'
            pct_matches = re.findall(pct_pattern, page_text)
            if pct_matches:
                financial_data['percentage_changes'] = pct_matches[:10]
            
            # Look for market cap, volume, etc.
            volume_pattern = r'Volume:\s*(\d+(?:,\d{3})*(?:\.\d+)?[KMB]?)'
            volume_matches = re.findall(volume_pattern, page_text)
            if volume_matches:
                financial_data['volumes'] = volume_matches[:5]
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")
            return {}
    
    def extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, microdata, etc.)"""
        try:
            structured_data = {}
            
            # Extract JSON-LD
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            if json_ld_scripts:
                import json
                json_data = []
                for script in json_ld_scripts:
                    try:
                        data = json.loads(script.string)
                        json_data.append(data)
                    except json.JSONDecodeError:
                        continue
                structured_data['json_ld'] = json_data
            
            # Extract meta tags
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_tags[name] = content
            
            if meta_tags:
                structured_data['meta_tags'] = meta_tags
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {}
    
    def clean_scraped_text(self, text: str) -> str:
        """Clean and normalize scraped text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\$\%\(\)]', ' ', text)
        
        # Remove citation markers
        text = re.sub(r'\[\d+\]', '', text)
        
        # Normalize currency symbols
        text = re.sub(r'\$\s+', '$', text)
        
        return text.strip()
