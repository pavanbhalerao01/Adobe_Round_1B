# main.py (DEFINITIVE FINAL BATCH PROCESSING VERSION)

import os
import json
import fitz
from sentence_transformers import SentenceTransformer, util
import torch
import datetime
import sys

# Make sure your Round 1A logic file is in the same directory
from round1a_logic import BalancedPDFOutlineExtractor

def format_output_json(ranked_sections, input_data, collection_path):
    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": input_data['persona']['role'],
            "job_to_be_done": input_data['job_to_be_done']['task'],
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    for i, section in enumerate(ranked_sections[:5]):
        output["extracted_sections"].append({
            "document": section['details']['document'],
            "section_title": section['details']['text'] if 'is_heading' in section['details'] else f"Relevant Content from Page {section['details']['page_number']}",
            "importance_rank": i + 1,
            "page_number": section['details']['page_number']
        })

    for section in ranked_sections[:5]:
        refined_text_snippet = ""
        heading_text = section['details']['text']
        
        # Correctly builds the path to the PDF inside the collection's 'PDFs' folder
        pdf_path = os.path.join(collection_path, 'PDFs', section['details']['document'])
        
        with fitz.open(pdf_path) as pdf_document:
            page = pdf_document.load_page(section['details']['page_number'] - 1)
            full_page_text = page.get_text()
            
            start_index = full_page_text.find(heading_text)
            if start_index != -1:
                end_index = start_index + 500
                refined_text_snippet = full_page_text[start_index:end_index].strip().replace('\n', ' ')
                if len(full_page_text) > end_index:
                    refined_text_snippet += "..."
            else:
                refined_text_snippet = section['details'].get('text', page.get_text()[:500].replace('\n', ' '))

        output["subsection_analysis"].append({
            "document": section['details']['document'],
            "refined_text": refined_text_snippet,
            "page_number": section['details']['page_number']
        })
    return output

def find_relevant_sections(collection_path, extractor_1a, relevance_model):
    input_path = os.path.join(collection_path, 'challenge1b_input.json')
    if not os.path.exists(input_path):
        print(f"  - Skipping: 'challenge1b_input.json' not found in {collection_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    enhanced_query = f"{input_data['job_to_be_done']['task']}. Focus on fun activities, social gatherings, nightlife, entertainment, beaches, and unique culinary experiences suitable for young adults."
    
    all_sections = []
    print("  - Analyzing structure with BALANCED Round 1A logic...")
    for doc in input_data['documents']:
        pdf_path = os.path.join(collection_path, 'PDFs', doc['filename'])
        if not os.path.exists(pdf_path):
            print(f"  - WARNING: PDF file not found: {pdf_path}")
            continue
        
        outline_data = extractor_1a.predict_outline(pdf_path)
        for heading in outline_data['outline']:
            all_sections.append({
                'document': doc['filename'],
                'page_number': heading['page'],
                'text': heading['text'],
                'is_heading': True
            })

    if not all_sections:
        print("  - WARNING: Round 1A logic did not detect headings. Switching to paragraph-based analysis.")
        for doc in input_data['documents']:
            pdf_path = os.path.join(collection_path, 'PDFs', doc['filename'])
            if not os.path.exists(pdf_path): continue
            with fitz.open(pdf_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    paragraphs = page.get_text().split('\n\n')
                    for para_text in paragraphs:
                        if len(para_text.strip()) > 50:
                            all_sections.append({
                                'document': doc['filename'],
                                'page_number': page_num + 1,
                                'text': para_text.strip().replace('\n', ' ')
                            })

    if not all_sections:
        print("  - ERROR: No text could be extracted from any documents. Exiting.")
        return

    print("  - Generating embeddings...")
    job_embedding = relevance_model.encode(enhanced_query, convert_to_tensor=True)
    section_texts = [sec['text'] for sec in all_sections]
    section_embeddings = relevance_model.encode(section_texts, convert_to_tensor=True)

    print("  - Calculating similarity scores...")
    cosine_scores = util.cos_sim(job_embedding, section_embeddings)[0]
    ranked_sections = sorted(
        [{'score': score, 'details': details} for score, details in zip(cosine_scores, all_sections)],
        key=lambda x: x['score'],
        reverse=True
    )

    final_output = format_output_json(ranked_sections, input_data, collection_path)
    
    # --- FILENAME CHANGE IS HERE ---
    output_path = os.path.join(collection_path, 'challenge1b_output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"  - ✅ Successfully created 'challenge1b_output.json' in '{collection_path}'.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <path_to_root_directory_of_collections>")
        print("Example: python main.py ./my_test_cases")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    if not os.path.isdir(root_directory):
        print(f"Error: Directory not found at '{root_directory}'")
        sys.exit(1)

    print("Loading models...")
    extractor_1a = BalancedPDFOutlineExtractor()
    extractor_1a.load_model(model_path='models/balanced_outline_extractor.pkl')
    relevance_model = SentenceTransformer('./model')
    print("Models loaded successfully.\n")

    for folder_name in sorted(os.listdir(root_directory)):
        collection_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(collection_path):
            print(f"--- Processing Collection: {folder_name} ---")
            try:
                find_relevant_sections(collection_path, extractor_1a, relevance_model)
            except Exception as e:
                print(f"  - 🚨 An error occurred while processing {folder_name}: {e}")
            print(f"--- Finished Collection: {folder_name} ---\n")