import fitz  # PyMuPDF
import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

class PDFAnalyzer:
    """
    PDF analyzer that extracts titles and hierarchical headings from PDF documents.
    Uses font size, style, and positioning analysis for intelligent heading detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Analyze a PDF file and extract title and headings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with title and outline, or None if analysis fails
        """
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Check page limit
            if len(doc) > 50:
                raise ValueError("PDF has more than 50 pages. Please upload a smaller document.")
            
            # Extract all text blocks with formatting information
            text_blocks = self._extract_text_blocks(doc)
            
            if not text_blocks:
                self.logger.warning("No text blocks found in PDF")
                doc.close()
                return None
            
            # Analyze font characteristics across the document
            font_stats = self._analyze_font_characteristics(text_blocks)
            
            # Extract title
            title = self._extract_title(text_blocks, font_stats)
            
            # Extract headings
            headings = self._extract_headings(text_blocks, font_stats)
            
            doc.close()
            
            # Format result
            result = {
                "title": title,
                "outline": headings
            }
            
            self.logger.info(f"Successfully analyzed PDF: {len(headings)} headings found")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing PDF: {str(e)}")
            raise

    def _extract_text_blocks(self, doc: fitz.Document) -> List[Dict]:
        """Extract text blocks with formatting information from all pages."""
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with detailed formatting information
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 1:  # Filter out very short text
                                text_blocks.append({
                                    "text": text,
                                    "page": page_num + 1,
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "bbox": span["bbox"],
                                    "color": span.get("color", 0)
                                })
        
        return text_blocks

    def _analyze_font_characteristics(self, text_blocks: List[Dict]) -> Dict:
        """Analyze font characteristics to identify typical patterns."""
        font_sizes = []
        font_styles = defaultdict(int)
        size_frequency = defaultdict(int)
        
        for block in text_blocks:
            font_sizes.append(block["size"])
            font_styles[block["font"]] += 1
            size_frequency[round(block["size"], 1)] += 1
        
        # Calculate statistics
        font_sizes.sort(reverse=True)
        avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        max_size = max(font_sizes) if font_sizes else 12
        
        # Find most common font size (likely body text)
        most_common_size = max(size_frequency.items(), key=lambda x: x[1])[0] if size_frequency else 12
        
        # Calculate size thresholds for heading levels
        size_diff = max_size - most_common_size
        
        return {
            "avg_size": avg_size,
            "max_size": max_size,
            "most_common_size": most_common_size,
            "size_diff": size_diff,
            "font_styles": font_styles,
            "size_frequency": size_frequency
        }

    def _extract_title(self, text_blocks: List[Dict], font_stats: Dict) -> str:
        """Extract document title using various heuristics."""
        candidates = []
        
        # Look for title candidates in the first few pages
        first_page_blocks = [block for block in text_blocks if block["page"] <= 2]
        
        for block in first_page_blocks:
            text = block["text"].strip()
            
            # Skip very short or very long text
            if len(text) < 3 or len(text) > 200:
                continue
                
            # Skip text that looks like headers, footers, or metadata
            if self._is_likely_metadata(text):
                continue
            
            # Calculate title score based on various factors
            score = self._calculate_title_score(block, font_stats)
            
            if score > 0:
                candidates.append({
                    "text": text,
                    "score": score,
                    "page": block["page"]
                })
        
        if candidates:
            # Sort by score and return best candidate
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[0]["text"]
        
        # Fallback: use first substantial text from first page
        for block in text_blocks:
            if block["page"] == 1:
                text = block["text"].strip()
                if len(text) > 10 and not self._is_likely_metadata(text):
                    return text
        
        return "Document"

    def _calculate_title_score(self, block: Dict, font_stats: Dict) -> float:
        """Calculate a score for how likely a text block is to be the title."""
        score = 0.0
        text = block["text"]
        
        # Font size score (larger = more likely to be title)
        size_ratio = block["size"] / font_stats["most_common_size"]
        if size_ratio > 1.5:
            score += size_ratio * 2
        
        # Position score (higher on page = more likely to be title)
        if block["bbox"][1] < 200:  # Top portion of page
            score += 3
        elif block["bbox"][1] < 400:  # Upper middle
            score += 1
        
        # Page score (first page is more likely)
        if block["page"] == 1:
            score += 2
        
        # Bold/italic score
        if block["flags"] & 2**4:  # Bold
            score += 1
        
        # Length score (reasonable title length)
        if 10 <= len(text) <= 100:
            score += 1
        elif 5 <= len(text) <= 150:
            score += 0.5
        
        # Penalize very common words that are unlikely to be titles
        common_words = ["the", "and", "or", "in", "on", "at", "to", "for", "of", "with"]
        if text.lower() in common_words:
            score -= 5
        
        return score

    def _is_likely_metadata(self, text: str) -> bool:
        """Check if text is likely to be metadata (headers, footers, page numbers, etc.)."""
        text_lower = text.lower().strip()
        
        # Page numbers
        if re.match(r'^\d+$', text_lower):
            return True
        
        # Common metadata patterns
        metadata_patterns = [
            r'^page \d+',
            r'^\d+\s*of\s*\d+',
            r'^chapter \d+',
            r'^section \d+',
            r'©\s*\d{4}',
            r'copyright',
            r'^www\.',
            r'@',
            r'\.com$',
            r'\.pdf$'
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Very short text
        if len(text) < 3:
            return True
            
        return False

    def _extract_headings(self, text_blocks: List[Dict], font_stats: Dict) -> List[Dict]:
        """Extract headings with hierarchical levels."""
        headings = []
        
        # Define size thresholds for heading levels
        base_size = font_stats["most_common_size"]
        
        # Calculate dynamic thresholds based on document characteristics
        h1_threshold = base_size + (font_stats["size_diff"] * 0.6)
        h2_threshold = base_size + (font_stats["size_diff"] * 0.3)
        h3_threshold = base_size + (font_stats["size_diff"] * 0.1)
        
        for block in text_blocks:
            text = block["text"].strip()
            
            # Skip very short text or metadata
            if len(text) < 3 or self._is_likely_metadata(text):
                continue
            
            # Check if this looks like a heading
            if self._is_likely_heading(block, font_stats):
                # Determine heading level based on font size
                level = self._determine_heading_level(
                    block["size"], h1_threshold, h2_threshold, h3_threshold
                )
                
                if level:
                    # Clean up the heading text
                    clean_text = self._clean_heading_text(text)
                    
                    if clean_text:
                        headings.append({
                            "level": level,
                            "text": clean_text,
                            "page": block["page"]
                        })
        
        # Post-process headings to ensure logical hierarchy
        headings = self._post_process_headings(headings)
        
        return headings

    def _is_likely_heading(self, block: Dict, font_stats: Dict) -> bool:
        """Determine if a text block is likely to be a heading."""
        text = block["text"].strip()
        
        # Size-based check
        size_ratio = block["size"] / font_stats["most_common_size"]
        if size_ratio < 1.05:  # Not significantly larger than body text
            return False
        
        # Length check (headings are usually not too long)
        if len(text) > 200:
            return False
        
        # Style checks
        is_bold = bool(block["flags"] & 2**4)
        
        # Common heading patterns
        heading_patterns = [
            r'^\d+\.\s*\w+',  # "1. Introduction"
            r'^\d+\s+\w+',    # "1 Introduction"
            r'^[A-Z][a-z]+\s+\d+',  # "Chapter 1"
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\w+\s*:$',     # "Introduction:"
        ]
        
        pattern_match = any(re.search(pattern, text) for pattern in heading_patterns)
        
        # Scoring system
        score = 0
        
        if size_ratio > 1.2:
            score += 2
        elif size_ratio > 1.1:
            score += 1
        
        if is_bold:
            score += 1
        
        if pattern_match:
            score += 2
        
        if len(text) <= 100:
            score += 1
        
        # Position-based scoring (headings often start at left margin)
        if block["bbox"][0] < 100:  # Left-aligned
            score += 1
        
        return score >= 2

    def _determine_heading_level(self, font_size: float, h1_thresh: float, 
                                h2_thresh: float, h3_thresh: float) -> Optional[str]:
        """Determine heading level based on font size."""
        if font_size >= h1_thresh:
            return "H1"
        elif font_size >= h2_thresh:
            return "H2"
        elif font_size >= h3_thresh:
            return "H3"
        return None

    def _clean_heading_text(self, text: str) -> str:
        """Clean and normalize heading text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove trailing punctuation that's not meaningful
        text = re.sub(r'[.]{2,}$', '', text)  # Remove trailing dots
        
        # Handle common heading numbering
        text = re.sub(r'^(\d+\.?\d*\.?)\s*', r'\1 ', text)  # Normalize numbering
        
        return text

    def _post_process_headings(self, headings: List[Dict]) -> List[Dict]:
        """Post-process headings to ensure logical hierarchy and remove duplicates."""
        if not headings:
            return headings
        
        # Remove near-duplicates (same text, similar page)
        cleaned_headings = []
        seen_texts = set()
        
        for heading in headings:
            text_key = heading["text"].lower().strip()
            if text_key not in seen_texts:
                cleaned_headings.append(heading)
                seen_texts.add(text_key)
        
        # Sort by page number
        cleaned_headings.sort(key=lambda x: (x["page"], x["text"]))
        
        return cleaned_headings
