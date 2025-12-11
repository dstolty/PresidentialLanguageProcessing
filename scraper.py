from bs4 import BeautifulSoup
import requests
from lxml import html
import os
import re
import sys

def extract_document_links(file_path):
    """
    Parses an HTML file and extracts links from the 'about' attribute of div tags.

    Args:
        url of target

    Returns:
        list: A list of full URLs extracted from the 'about' attributes.
              Returns an empty list if the file cannot be read or parsed,
              or if no matching tags are found.
    """
    base = "https://www.presidency.ucsb.edu"
    links = []

    try:
        # Read the HTML file content
        page = requests.get(url, verify=False)

        # Parse the HTML content
        soup = BeautifulSoup(page.content, 'html.parser')

        # Find all div tags with the 'about' attribute
        # The selector looks for div tags that have an 'about' attribute defined
        div_tags = soup.find_all('div', attrs={'about': True}) # Using attrs={'about': True} checks for the presence of the attribute

        # Extract the 'about' attribute value and construct the full URL
        for tag in div_tags:
            relative_link = tag.get('about')
            if relative_link:
                # Ensure the link starts with '/' before joining
                if not relative_link.startswith('/'):
                    relative_link = '/' + relative_link
                full_link = base + relative_link
                links.append(full_link)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return []

    print(f'Found {len(links)} links')
    return links

base_url = 'https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders?items_per_page=60&page='


OUTPUT_FILE = './executive_orders.tsv'

## 3304 pages ( / 60 per page 551) ## 472 - 542

## for exec orders - 179
with open(OUTPUT_FILE,'a') as dataset:
    for i in range(42,180):
        print(f"Page {i}")
        url = f'{base_url}{i}'
        links = extract_document_links(url)

        #new_base = "https://www.presidency.ucsb.edu"

        if links:
            for link in links:
                
                data = {
                'president': None,
                'date': None,
                'text': None 
                }
                
                url = link

                page = requests.get(url)
                soup = BeautifulSoup(page.content,'html.parser')
                
                president_container = soup.find('div', class_='field-docs-person')
                if president_container:
                    president_link = president_container.find('h3', class_='diet-title').find('a')
                    if president_link:
                        data['president'] = president_link.get_text(strip=True)
                        
                # 4. Extract the Date
                # Based on SPEECHex.html structure: field-docs-start-date-time -> span.date-display-single
                date_container = soup.find('div', class_='field-docs-start-date-time')
                if date_container:
                    date_span = date_container.find('span', class_='date-display-single')
                    if date_span:
                        data['date'] = date_span.get_text(strip=True)


                # 5. Extract the Document Text
                # Based on SPEECHex.html structure: field-docs-content -> all <p> tags inside
                content_div = soup.find('div', class_='field-docs-content')
                if content_div:
                    paragraphs = content_div.find_all('p')
                    # Join the text from all paragraphs, separated by double newlines
                    data['text'] = ''.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)) # Ensure we don't add empty paragraphs
                    
                
                ## used to ignore the documents in the current admin that do not have transcripts
                if not data['text'].startswith('Unlike previous administrations,'):

                    content = '\t'.join(list(data.values())) + '\n'
                    dataset.write(content)

