"""
Lightweight ML classifier for PDF text structure analysis.
Uses TF-IDF features with scikit-learn for heading classification and title detection.
Designed to work offline within 200MB constraint.
"""

import re
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DocumentStructureClassifier:
    """
    Lightweight ML classifier for document structure analysis.
    Classifies text blocks as title, heading (H1/H2/H3), or body text.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.title_classifier = None
        self.heading_classifier = None
        self.level_classifier = None
        
        # Feature extractors
        self.title_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        self.heading_vectorizer = TfidfVectorizer(
            max_features=800,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.models_trained = False
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)

    def extract_features(self, text_block: Dict) -> Dict:
        """Extract features from a text block for classification."""
        text = text_block.get("text", "").strip()
        size = text_block.get("size", 12)
        bbox = text_block.get("bbox", (0, 0, 0, 0))
        flags = text_block.get("flags", 0)
        page = text_block.get("page", 1)
        
        features = {
            # Text features
            'length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            
            # Formatting features
            'font_size': size,
            'is_bold': bool(flags & 2**4),
            'is_italic': bool(flags & 2**5),
            
            # Position features
            'x_position': bbox[0] if bbox else 0,
            'y_position': bbox[1] if bbox else 0,
            'width': bbox[2] - bbox[0] if bbox else 0,
            'height': bbox[3] - bbox[1] if bbox else 0,
            'page_number': page,
            'is_first_page': page == 1,
            'is_early_page': page <= 2,
            
            # Text pattern features
            'starts_with_number': bool(re.match(r'^\d+', text)),
            'starts_with_capital': text and text[0].isupper(),
            'is_all_caps': text.isupper() and len(text) > 3,
            'has_colon': ':' in text,
            'ends_with_period': text.endswith('.'),
            'has_parentheses': '(' in text or ')' in text,
            
            # Semantic patterns
            'is_numbered_heading': bool(re.match(r'^\d+\.?\s+[A-Z]', text)),
            'is_subsection': bool(re.match(r'^\d+\.\d+', text)),
            'is_chapter': bool(re.search(r'\b(chapter|section|part)\s+\d+\b', text, re.I)),
            'has_heading_words': bool(re.search(
                r'\b(introduction|overview|summary|conclusion|background|methodology|results|discussion)\b', 
                text, re.I
            )),
            
            # Title-specific features
            'is_title_position': bbox[1] < 200 if bbox else False,  # Top of page
            'title_length_range': 20 <= len(text) <= 100,
            'has_title_words': bool(re.search(
                r'\b(report|document|study|analysis|proposal|plan|guide|manual|application|request)\b',
                text, re.I
            )),
        }
        
        return features

    def generate_training_data(self) -> Tuple[List[Dict], List[str], List[str]]:
        """Generate synthetic training data based on common document patterns."""
        training_samples = []
        title_labels = []
        heading_labels = []
        
        # Title examples
        title_examples = [
            {"text": "Annual Financial Report 2024", "size": 18, "bbox": (100, 50, 400, 80), "flags": 16, "page": 1},
            {"text": "Project Implementation Guide", "size": 16, "bbox": (120, 60, 350, 85), "flags": 16, "page": 1},
            {"text": "Technical Documentation Overview", "size": 20, "bbox": (80, 40, 420, 75), "flags": 16, "page": 1},
            {"text": "Research Study on Market Trends", "size": 17, "bbox": (90, 55, 380, 82), "flags": 16, "page": 1},
            {"text": "Application Form for Grant Proposal", "size": 15, "bbox": (100, 70, 400, 95), "flags": 0, "page": 1},
        ]
        
        for example in title_examples:
            training_samples.append(example)
            title_labels.append("title")
            heading_labels.append("none")
        
        # H1 Heading examples
        h1_examples = [
            {"text": "1. Introduction", "size": 14, "bbox": (50, 150, 200, 170), "flags": 16, "page": 1},
            {"text": "Executive Summary", "size": 15, "bbox": (50, 200, 250, 220), "flags": 16, "page": 1},
            {"text": "Chapter 1: Overview", "size": 16, "bbox": (50, 180, 280, 205), "flags": 16, "page": 2},
            {"text": "METHODOLOGY", "size": 14, "bbox": (50, 160, 200, 180), "flags": 16, "page": 3},
            {"text": "Background and Context", "size": 14, "bbox": (50, 170, 300, 190), "flags": 16, "page": 2},
        ]
        
        for example in h1_examples:
            training_samples.append(example)
            title_labels.append("not_title")
            heading_labels.append("H1")
        
        # H2 Heading examples
        h2_examples = [
            {"text": "1.1 Project Scope", "size": 13, "bbox": (70, 220, 250, 240), "flags": 16, "page": 2},
            {"text": "Data Collection Methods", "size": 13, "bbox": (70, 200, 280, 220), "flags": 0, "page": 3},
            {"text": "2.1 Technical Requirements", "size": 12, "bbox": (70, 180, 300, 200), "flags": 16, "page": 4},
            {"text": "Risk Assessment", "size": 13, "bbox": (70, 190, 220, 210), "flags": 16, "page": 3},
            {"text": "Implementation Timeline", "size": 12, "bbox": (70, 210, 280, 230), "flags": 0, "page": 4},
        ]
        
        for example in h2_examples:
            training_samples.append(example)
            title_labels.append("not_title")
            heading_labels.append("H2")
        
        # H3 Heading examples
        h3_examples = [
            {"text": "1.1.1 Specific Objectives", "size": 12, "bbox": (90, 240, 280, 260), "flags": 0, "page": 2},
            {"text": "Data Sources", "size": 11, "bbox": (90, 220, 200, 240), "flags": 16, "page": 3},
            {"text": "Quality Assurance", "size": 12, "bbox": (90, 230, 240, 250), "flags": 0, "page": 4},
            {"text": "Testing Procedures", "size": 11, "bbox": (90, 210, 250, 230), "flags": 16, "page": 5},
            {"text": "Performance Metrics", "size": 12, "bbox": (90, 200, 270, 220), "flags": 0, "page": 5},
        ]
        
        for example in h3_examples:
            training_samples.append(example)
            title_labels.append("not_title")
            heading_labels.append("H3")
        
        # Body text examples (negative samples)
        body_examples = [
            {"text": "This document provides a comprehensive overview of the project implementation strategy and methodology.", "size": 11, "bbox": (50, 300, 500, 320), "flags": 0, "page": 2},
            {"text": "The research methodology employed in this study follows established academic standards.", "size": 11, "bbox": (50, 320, 480, 340), "flags": 0, "page": 3},
            {"text": "Figure 1 shows the relationship between various components of the system architecture.", "size": 10, "bbox": (50, 280, 450, 300), "flags": 0, "page": 4},
            {"text": "Page 12", "size": 9, "bbox": (250, 750, 300, 770), "flags": 0, "page": 12},
            {"text": "© 2024 All Rights Reserved", "size": 8, "bbox": (50, 780, 200, 800), "flags": 0, "page": 1},
        ]
        
        for example in body_examples:
            training_samples.append(example)
            title_labels.append("not_title")
            heading_labels.append("body")
        
        return training_samples, title_labels, heading_labels

    def train_models(self, save_models: bool = True) -> Dict[str, float]:
        """Train the classification models using synthetic data."""
        self.logger.info("Generating training data...")
        samples, title_labels, heading_labels = self.generate_training_data()
        
        # Extract features
        feature_vectors = []
        text_features = []
        
        for sample in samples:
            features = self.extract_features(sample)
            feature_vectors.append([
                features['length'], features['word_count'], features['font_size'],
                int(features['is_bold']), int(features['is_italic']), 
                features['y_position'], int(features['is_first_page']),
                int(features['starts_with_number']), int(features['starts_with_capital']),
                int(features['is_all_caps']), int(features['has_colon']),
                int(features['is_numbered_heading']), int(features['is_chapter']),
                int(features['has_heading_words']), int(features['has_title_words']),
                int(features['is_title_position']), int(features['title_length_range'])
            ])
            text_features.append(sample['text'])
        
        X_features = np.array(feature_vectors)
        X_text = text_features
        
        # Train title classifier
        self.logger.info("Training title classifier...")
        title_X_train, title_X_test, title_y_train, title_y_test = train_test_split(
            X_features, title_labels, test_size=0.2, random_state=42, stratify=title_labels
        )
        
        self.title_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        )
        self.title_classifier.fit(title_X_train, title_y_train)
        title_score = self.title_classifier.score(title_X_test, title_y_test)
        
        # Train heading classifier (filters out body text)
        self.logger.info("Training heading classifier...")
        heading_X_train, heading_X_test, heading_y_train, heading_y_test = train_test_split(
            X_features, heading_labels, test_size=0.2, random_state=42, stratify=heading_labels
        )
        
        self.heading_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        )
        self.heading_classifier.fit(heading_X_train, heading_y_train)
        heading_score = self.heading_classifier.score(heading_X_test, heading_y_test)
        
        # Fit vectorizers on text data
        self.title_vectorizer.fit(X_text)
        self.heading_vectorizer.fit(X_text)
        
        self.models_trained = True
        
        if save_models:
            self.save_models()
        
        scores = {
            "title_accuracy": title_score,
            "heading_accuracy": heading_score
        }
        
        self.logger.info(f"Model training completed. Scores: {scores}")
        return scores

    def predict_title(self, text_block: Dict) -> Tuple[bool, float]:
        """Predict if a text block is likely a document title."""
        if not self.models_trained or self.title_classifier is None:
            return False, 0.0
        
        features = self.extract_features(text_block)
        feature_vector = np.array([[
            features['length'], features['word_count'], features['font_size'],
            int(features['is_bold']), int(features['is_italic']), 
            features['y_position'], int(features['is_first_page']),
            int(features['starts_with_number']), int(features['starts_with_capital']),
            int(features['is_all_caps']), int(features['has_colon']),
            int(features['is_numbered_heading']), int(features['is_chapter']),
            int(features['has_heading_words']), int(features['has_title_words']),
            int(features['is_title_position']), int(features['title_length_range'])
        ]])
        
        prediction = self.title_classifier.predict(feature_vector)[0]
        confidence = self.title_classifier.predict_proba(feature_vector)[0].max()
        
        return prediction == "title", confidence

    def predict_heading(self, text_block: Dict) -> Tuple[Optional[str], float]:
        """Predict if a text block is a heading and its level."""
        if not self.models_trained or self.heading_classifier is None:
            return None, 0.0
        
        features = self.extract_features(text_block)
        feature_vector = np.array([[
            features['length'], features['word_count'], features['font_size'],
            int(features['is_bold']), int(features['is_italic']), 
            features['y_position'], int(features['is_first_page']),
            int(features['starts_with_number']), int(features['starts_with_capital']),
            int(features['is_all_caps']), int(features['has_colon']),
            int(features['is_numbered_heading']), int(features['is_chapter']),
            int(features['has_heading_words']), int(features['has_title_words']),
            int(features['is_title_position']), int(features['title_length_range'])
        ]])
        
        prediction = self.heading_classifier.predict(feature_vector)[0]
        confidence = self.heading_classifier.predict_proba(feature_vector)[0].max()
        
        if prediction in ["H1", "H2", "H3"]:
            return prediction, confidence
        else:
            return None, confidence

    def save_models(self):
        """Save trained models to disk."""
        try:
            with open(self.model_path / "title_classifier.pkl", "wb") as f:
                pickle.dump(self.title_classifier, f)
            
            with open(self.model_path / "heading_classifier.pkl", "wb") as f:
                pickle.dump(self.heading_classifier, f)
            
            with open(self.model_path / "title_vectorizer.pkl", "wb") as f:
                pickle.dump(self.title_vectorizer, f)
            
            with open(self.model_path / "heading_vectorizer.pkl", "wb") as f:
                pickle.dump(self.heading_vectorizer, f)
            
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_models(self) -> bool:
        """Load pre-trained models from disk."""
        try:
            with open(self.model_path / "title_classifier.pkl", "rb") as f:
                self.title_classifier = pickle.load(f)
            
            with open(self.model_path / "heading_classifier.pkl", "rb") as f:
                self.heading_classifier = pickle.load(f)
            
            with open(self.model_path / "title_vectorizer.pkl", "rb") as f:
                self.title_vectorizer = pickle.load(f)
            
            with open(self.model_path / "heading_vectorizer.pkl", "rb") as f:
                self.heading_vectorizer = pickle.load(f)
            
            self.models_trained = True
            self.logger.info("Models loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False


# Initialize and train models on first import
def initialize_classifier() -> DocumentStructureClassifier:
    """Initialize and train the classifier if needed."""
    classifier = DocumentStructureClassifier()
    
    # Try to load existing models first
    if not classifier.load_models():
        # Train new models if loading fails
        logging.info("Training new ML models...")
        classifier.train_models()
    
    return classifier


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    classifier = initialize_classifier()
    
    # Test samples
    test_title = {
        "text": "Annual Report on Software Development", 
        "size": 18, 
        "bbox": (100, 50, 400, 80), 
        "flags": 16, 
        "page": 1
    }
    
    test_heading = {
        "text": "1. Introduction", 
        "size": 14, 
        "bbox": (50, 150, 200, 170), 
        "flags": 16, 
        "page": 1
    }
    
    is_title, title_conf = classifier.predict_title(test_title)
    heading_level, heading_conf = classifier.predict_heading(test_heading)
    
    print(f"Title prediction: {is_title} (confidence: {title_conf:.2f})")
    print(f"Heading prediction: {heading_level} (confidence: {heading_conf:.2f})")