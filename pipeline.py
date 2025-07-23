# src/challenge_1b/pipeline.py
import fitz
import json
from datetime import datetime

# Import your 1A module
from src.challenge_1a.analyzer import PDFAnalyzer

# Import ML/NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Import summarization libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# --- Load the model from the local path ---
# This ensures it works offline inside the Docker container
MODEL_PATH = './models/all-MiniLM-L6-v2'
SENTENCE_MODEL = SentenceTransformer(MODEL_PATH)

def extract_section_content(pdf_path: str, outline: list) -> list:
    # ... (Paste the extract_section_content function from the previous answer) ...
    # This function takes the 1A outline and extracts full text for each section.

def filter_with_tfidf(query: str, sections: list, top_k: int = 50) -> list:
    # ... (Paste the filter_with_tfidf function) ...

def rerank_with_minilm(query: str, candidate_sections: list) -> list:
    # ... (Paste the rerank_with_minilm function, but make sure it uses SENTENCE_MODEL) ...
    # Example change:
    # query_embedding = SENTENCE_MODEL.encode(...)
    # corpus_embeddings = SENTENCE_MODEL.encode(...)

def generate_refined_text(section_full_text: str, num_sentences: int = 5) -> str:
    # ... (Paste the generate_refined_text function using sumy) ...

def run_1b_pipeline(input_data: dict, input_pdf_dir: str):
    """The main orchestrator for the 1B challenge."""
    
    # 1. Parse input and create query
    persona = input_data['persona']['role']
    job = input_data['job_to_be_done']['task']
    query = f"As a {persona}, I need to {job}"
    
    document_files = [doc['filename'] for doc in input_data['documents']]
    
    # 2. Extract sections from all documents
    all_sections = []
    analyzer_1a = PDFAnalyzer()
    for filename in document_files:
        pdf_path = f"{input_pdf_dir}/{filename}"
        
        # Run 1A logic
        result_1a = analyzer_1a.analyze_pdf(pdf_path)
        outline = result_1a['outline']
        
        # Extract full content for each section
        doc_sections = extract_section_content(pdf_path, outline)
        for section in doc_sections:
            section['document'] = filename # Store filename, not full path
        all_sections.extend(doc_sections)

    # 3. Filter and Rank
    candidate_sections = filter_with_tfidf(query, all_sections, top_k=50)
    final_ranked_sections = rerank_with_minilm(query, candidate_sections)
    
    # 4. Generate Subsection Analysis for top 5
    subsection_analysis = []
    top_5_sections = final_ranked_sections[:5]
    for section in top_5_sections:
        refined_text = generate_refined_text(section['full_text'])
        subsection_analysis.append({
            "document": section['document'],
            "refined_text": refined_text,
            "page_number": section['page_number']
        })

    # 5. Format the final JSON output
    output_json = {
        "metadata": {
            "input_documents": document_files,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": s['document'],
                "section_title": s['section_title'],
                "importance_rank": i + 1,
                "page_number": s['page_number']
            } for i, s in enumerate(top_5_sections)
        ],
        "subsection_analysis": subsection_analysis
    }
    
    return output_json