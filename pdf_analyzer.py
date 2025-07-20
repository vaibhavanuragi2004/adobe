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
                    # Group spans within the same line that have similar formatting
                    block_lines = []
                    
                    for line in block["lines"]:
                        line_text_parts = []
                        line_bbox = None
                        line_font = None
                        line_size = None
                        line_flags = None
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text_parts.append(text)
                                # Use the dominant formatting of the line
                                if line_font is None or span["size"] > (line_size or 0):
                                    line_font = span["font"]
                                    line_size = span["size"]
                                    line_flags = span["flags"]
                                    line_bbox = span["bbox"]
                        
                        if line_text_parts:
                            # Combine text parts into a single line
                            full_line_text = " ".join(line_text_parts).strip()
                            if full_line_text and len(full_line_text) > 1:
                                block_lines.append({
                                    "text": full_line_text,
                                    "page": page_num + 1,
                                    "font": line_font,
                                    "size": line_size,
                                    "flags": line_flags,
                                    "bbox": line_bbox,
                                    "color": 0,
                                    "line_y": line_bbox[1] if line_bbox else 0
                                })
                    
                    # Group consecutive lines with same formatting into semantic blocks
                    if block_lines:
                        current_group = [block_lines[0]]
                        
                        for i in range(1, len(block_lines)):
                            current_line = block_lines[i]
                            prev_line = current_group[-1]
                            
                            # Check if lines should be grouped (same font, size, close Y position)
                            size_similar = abs(current_line["size"] - prev_line["size"]) < 1
                            font_similar = current_line["font"] == prev_line["font"]
                            y_close = abs(current_line["line_y"] - prev_line["line_y"]) < 50
                            
                            if size_similar and font_similar and y_close:
                                current_group.append(current_line)
                            else:
                                # Finalize current group
                                if current_group:
                                    self._add_grouped_text_block(current_group, text_blocks)
                                current_group = [current_line]
                        
                        # Add final group
                        if current_group:
                            self._add_grouped_text_block(current_group, text_blocks)
        
        return text_blocks

    def _add_grouped_text_block(self, line_group: List[Dict], text_blocks: List[Dict]):
        """Add a grouped text block from multiple lines."""
        if not line_group:
            return
            
        # Combine text from all lines in the group
        combined_text = " ".join(line["text"] for line in line_group).strip()
        
        # Use formatting from the first line (they should be similar)
        first_line = line_group[0]
        
        # Calculate combined bounding box
        min_x = min(line["bbox"][0] for line in line_group)
        min_y = min(line["bbox"][1] for line in line_group)  
        max_x = max(line["bbox"][2] for line in line_group)
        max_y = max(line["bbox"][3] for line in line_group)
        
        text_blocks.append({
            "text": combined_text,
            "page": first_line["page"],
            "font": first_line["font"],
            "size": first_line["size"],
            "flags": first_line["flags"],
            "bbox": (min_x, min_y, max_x, max_y),
            "color": first_line["color"],
            "line_count": len(line_group)
        })

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
        """Extract document title using semantic analysis and multi-line detection."""
        candidates = []
        
        # Look for title candidates primarily on the first page
        first_page_blocks = [block for block in text_blocks if block["page"] == 1]
        
        # Sort blocks by Y position to process from top to bottom
        first_page_blocks.sort(key=lambda x: x["bbox"][1])
        
        for i, block in enumerate(first_page_blocks):
            text = block["text"].strip()
            
            # Skip very short text or obvious metadata
            if len(text) < 3 or self._is_likely_metadata(text):
                continue
            
            # Calculate title score with enhanced semantic understanding
            score = self._calculate_semantic_title_score(block, font_stats, first_page_blocks, i)
            
            if score > 0:
                # Check for multi-line titles by looking at nearby blocks
                full_title = self._reconstruct_multiline_title(block, first_page_blocks, i)
                
                candidates.append({
                    "text": full_title,
                    "score": score,
                    "page": block["page"],
                    "original_text": text
                })
        
        if candidates:
            # Sort by score and return best candidate
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Clean up the title
            best_title = candidates[0]["text"].strip()
            # Remove common title artifacts
            best_title = re.sub(r'^(title|document|report):\s*', '', best_title, flags=re.IGNORECASE)
            best_title = re.sub(r'\s+', ' ', best_title)  # Normalize whitespace
            
            return best_title if len(best_title) > 3 else candidates[0]["original_text"]
        
        # Fallback: use first substantial text from first page
        for block in first_page_blocks:
            text = block["text"].strip()
            if len(text) > 5 and not self._is_likely_metadata(text):
                return text
        
        return "Document"

    def _reconstruct_multiline_title(self, main_block: Dict, all_blocks: List[Dict], block_index: int) -> str:
        """Reconstruct a title that might span multiple lines."""
        title_parts = [main_block["text"]]
        main_size = main_block["size"]
        main_font = main_block["font"]
        main_y = main_block["bbox"][1]
        
        # Look at nearby blocks (before and after)
        for i in range(max(0, block_index - 2), min(len(all_blocks), block_index + 3)):
            if i == block_index:
                continue
                
            candidate = all_blocks[i]
            
            # Skip if too far away vertically
            if abs(candidate["bbox"][1] - main_y) > 100:
                continue
            
            # Check if it's part of the same title (similar formatting and position)
            size_similar = abs(candidate["size"] - main_size) < 2
            font_similar = candidate["font"] == main_font or self._fonts_are_similar(candidate["font"], main_font)
            
            if size_similar and font_similar and not self._is_likely_metadata(candidate["text"]):
                # Determine if it should come before or after main text
                if candidate["bbox"][1] < main_y:  # Above main block
                    title_parts.insert(0, candidate["text"])
                else:  # Below main block
                    title_parts.append(candidate["text"])
        
        return " ".join(title_parts).strip()

    def _fonts_are_similar(self, font1: str, font2: str) -> bool:
        """Check if two fonts are similar (same family, different style)."""
        if not font1 or not font2:
            return False
        
        # Remove style suffixes to compare font families
        font1_base = re.sub(r'[-,+](bold|italic|regular|light|medium|heavy).*$', '', font1.lower())
        font2_base = re.sub(r'[-,+](bold|italic|regular|light|medium|heavy).*$', '', font2.lower())
        
        return font1_base == font2_base

    def _calculate_semantic_title_score(self, block: Dict, font_stats: Dict, page_blocks: List[Dict], block_index: int) -> float:
        """Calculate title score with enhanced semantic understanding."""
        score = 0.0
        text = block["text"]
        
        # Font size score - more nuanced
        size_ratio = block["size"] / font_stats["most_common_size"]
        if size_ratio > 2.0:
            score += 8  # Very large text
        elif size_ratio > 1.5:
            score += 5  # Large text
        elif size_ratio > 1.2:
            score += 3  # Moderately large text
        elif size_ratio < 0.9:
            score -= 2  # Smaller than body text
        
        # Position score - top of page is more likely
        y_position = block["bbox"][1]
        if y_position < 150:  # Very top
            score += 4
        elif y_position < 300:  # Upper portion
            score += 2
        elif y_position > 600:  # Lower portion
            score -= 2
        
        # Bold/italic formatting
        if block["flags"] & 2**4:  # Bold
            score += 2
        
        # Length considerations - titles have optimal length
        text_length = len(text)
        if 20 <= text_length <= 80:
            score += 3  # Ideal title length
        elif 10 <= text_length <= 150:
            score += 1  # Acceptable length
        elif text_length > 200:
            score -= 3  # Too long for a title
        elif text_length < 5:
            score -= 2  # Too short
        
        # Semantic content analysis
        title_indicators = [
            r'\b(report|document|study|analysis|proposal|plan|guide|manual|overview)\b',
            r'\b(introduction|summary|abstract|conclusion)\b',
            r'\b(project|research|development|system|application)\b'
        ]
        
        for pattern in title_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        
        # Penalize metadata-like content
        metadata_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates
            r'©|copyright|all rights reserved',
            r'^chapter\s+\d+',
            r'^section\s+\d+',
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 4
        
        # Context score - consider surrounding blocks
        if block_index > 0:
            prev_block = page_blocks[block_index - 1]
            if self._is_likely_metadata(prev_block["text"]):
                score += 1  # Title often comes after metadata
        
        # Uniqueness score - titles are typically unique on the page
        similar_blocks = [b for b in page_blocks if abs(b["size"] - block["size"]) < 2 and b["text"] != text]
        if len(similar_blocks) == 0:
            score += 2  # Unique formatting suggests importance
        elif len(similar_blocks) > 3:
            score -= 1  # Too common formatting
        
        return score

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
        """Extract headings with enhanced semantic understanding."""
        headings = []
        
        # Group text blocks by page for better context analysis
        pages = {}
        for block in text_blocks:
            page_num = block["page"]
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
        
        # Define more nuanced size thresholds
        base_size = font_stats["most_common_size"]
        size_diff = font_stats["size_diff"]
        
        # Adaptive thresholds based on document characteristics
        if size_diff > 10:  # Large variation in font sizes
            h1_threshold = base_size + (size_diff * 0.4)
            h2_threshold = base_size + (size_diff * 0.2)
            h3_threshold = base_size + (size_diff * 0.05)
        else:  # Small variation - be more sensitive
            h1_threshold = base_size + max(2, size_diff * 0.6)
            h2_threshold = base_size + max(1.5, size_diff * 0.3)
            h3_threshold = base_size + max(1, size_diff * 0.1)
        
        # Process each page for context-aware heading detection
        for page_num, page_blocks in pages.items():
            page_blocks.sort(key=lambda x: x["bbox"][1])  # Sort by Y position
            
            for i, block in enumerate(page_blocks):
                text = block["text"].strip()
                
                # Skip very short text, metadata, or obvious body text
                if len(text) < 3 or self._is_likely_metadata(text):
                    continue
                
                # Enhanced heading detection with context
                if self._is_semantic_heading(block, font_stats, page_blocks, i):
                    level = self._determine_semantic_heading_level(
                        block, h1_threshold, h2_threshold, h3_threshold, font_stats
                    )
                    
                    if level:
                        # Clean and potentially merge multi-line headings
                        clean_text = self._extract_complete_heading(block, page_blocks, i)
                        
                        if clean_text and len(clean_text.strip()) > 2:
                            # Avoid duplicate headings from title extraction
                            if not self._is_duplicate_heading(clean_text, headings):
                                headings.append({
                                    "level": level,
                                    "text": clean_text,
                                    "page": block["page"]
                                })
        
        # Enhanced post-processing for semantic consistency
        headings = self._post_process_semantic_headings(headings, font_stats)
        
        return headings

    def _is_semantic_heading(self, block: Dict, font_stats: Dict, page_blocks: List[Dict], block_index: int) -> bool:
        """Enhanced heading detection with semantic understanding."""
        text = block["text"].strip()
        size_ratio = block["size"] / font_stats["most_common_size"]
        
        # Must be larger than body text
        if size_ratio <= 1.05:
            return False
        
        # Length constraints - headings shouldn't be too long
        if len(text) > 300 or len(text) < 3:
            return False
        
        # Calculate semantic score
        score = 0
        
        # Size scoring
        if size_ratio > 1.8:
            score += 4
        elif size_ratio > 1.4:
            score += 3  
        elif size_ratio > 1.2:
            score += 2
        elif size_ratio > 1.1:
            score += 1
        
        # Bold formatting
        if block["flags"] & 2**4:
            score += 2
        
        # Position context - headings often have space around them
        if block_index > 0 and block_index < len(page_blocks) - 1:
            prev_block = page_blocks[block_index - 1]
            next_block = page_blocks[block_index + 1]
            
            # Check for whitespace/spacing around the heading
            y_gap_before = block["bbox"][1] - prev_block["bbox"][3] if prev_block else 50
            y_gap_after = next_block["bbox"][1] - block["bbox"][3] if next_block else 50
            
            if y_gap_before > 20 or y_gap_after > 15:
                score += 1
        
        # Heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][a-z]*\s+\d+',  # "Chapter 1", "Section 2"
            r'^[A-Z][A-Z\s]*[A-Z]$',  # ALL CAPS headings
            r'^[A-Z][^.!?]*$',  # Starts with capital, no sentence ending
            r'^\d+\.\d+\.?\s',  # "2.1 Subsection"
            r':\s*$',  # Ends with colon
        ]
        
        for pattern in heading_patterns:
            if re.search(pattern, text):
                score += 2
                break
        
        # Semantic content indicators
        heading_keywords = [
            r'\b(introduction|overview|summary|conclusion|background)\b',
            r'\b(methodology|approach|implementation|results)\b',
            r'\b(discussion|analysis|findings|recommendations)\b',
            r'\b(chapter|section|part|appendix|references)\b'
        ]
        
        for pattern in heading_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
                break
        
        # Penalize body-text-like content
        if re.search(r'\b(the|and|or|in|on|at|to|for|of|with|by|from)\s+\w+\s+\w+', text.lower()):
            if len(text) > 100:  # Long sentences with common words
                score -= 2
        
        # Font uniqueness on the page
        similar_blocks = [b for b in page_blocks 
                         if abs(b["size"] - block["size"]) < 1 
                         and b["font"] == block["font"] 
                         and b["text"] != text]
        
        if len(similar_blocks) <= 2:  # Relatively unique formatting
            score += 1
        
        return score >= 3

    def _determine_semantic_heading_level(self, block: Dict, h1_thresh: float, h2_thresh: float, h3_thresh: float, font_stats: Dict) -> str:
        """Determine heading level with semantic context."""
        font_size = block["size"]
        text = block["text"].strip()
        
        # Base level determination by size
        if font_size >= h1_thresh:
            base_level = "H1"
        elif font_size >= h2_thresh:
            base_level = "H2" 
        elif font_size >= h3_thresh:
            base_level = "H3"
        else:
            return None
        
        # Adjust level based on semantic content
        if re.search(r'^\d+\.\d+\.\d+', text):  # 1.2.3 format
            return "H3"
        elif re.search(r'^\d+\.\d+', text):  # 1.2 format
            return "H2" if base_level != "H3" else "H2"
        elif re.search(r'^\d+\.?\s', text):  # 1. or 1 format
            return "H1" if font_size >= h2_thresh else "H2"
        
        # Content-based adjustments
        if re.search(r'\b(chapter|part)\s+\d+\b', text, re.IGNORECASE):
            return "H1"
        elif re.search(r'\b(section|subsection)\b', text, re.IGNORECASE):
            return "H2"
        
        return base_level

    def _extract_complete_heading(self, main_block: Dict, page_blocks: List[Dict], block_index: int) -> str:
        """Extract potentially multi-line heading text."""
        heading_parts = [main_block["text"]]
        main_size = main_block["size"]
        main_font = main_block["font"]
        main_y = main_block["bbox"][1]
        
        # Look for continuation lines immediately before/after
        for offset in [-1, 1]:
            check_index = block_index + offset
            if 0 <= check_index < len(page_blocks):
                candidate = page_blocks[check_index]
                
                # Must be very close vertically and similar formatting
                y_distance = abs(candidate["bbox"][1] - main_y)
                size_similar = abs(candidate["size"] - main_size) < 1.5
                font_similar = candidate["font"] == main_font
                
                if (y_distance < 30 and size_similar and font_similar 
                    and len(candidate["text"].strip()) > 2
                    and not self._is_likely_metadata(candidate["text"])):
                    
                    if offset == -1:  # Before main block
                        heading_parts.insert(0, candidate["text"])
                    else:  # After main block  
                        heading_parts.append(candidate["text"])
        
        combined = " ".join(heading_parts).strip()
        return self._clean_heading_text(combined)

    def _is_duplicate_heading(self, text: str, existing_headings: List[Dict]) -> bool:
        """Check if heading is duplicate or very similar to existing ones."""
        text_lower = text.lower().strip()
        
        for heading in existing_headings:
            existing_lower = heading["text"].lower().strip()
            
            # Exact match
            if text_lower == existing_lower:
                return True
                
            # Very similar (one contains the other and they're close in length)
            if (text_lower in existing_lower or existing_lower in text_lower):
                ratio = min(len(text_lower), len(existing_lower)) / max(len(text_lower), len(existing_lower))
                if ratio > 0.7:
                    return True
        
        return False

    def _post_process_semantic_headings(self, headings: List[Dict], font_stats: Dict) -> List[Dict]:
        """Enhanced post-processing for semantic consistency."""
        if not headings:
            return headings
        
        # Remove duplicates and very similar headings
        cleaned = []
        for heading in headings:
            if not self._is_duplicate_heading(heading["text"], cleaned):
                cleaned.append(heading)
        
        # Sort by page and position
        cleaned.sort(key=lambda x: (x["page"], x["text"]))
        
        # Validate heading hierarchy and adjust if needed
        validated = []
        for i, heading in enumerate(cleaned):
            # Check for logical hierarchy issues
            if i > 0:
                prev_heading = validated[-1]
                
                # If we jump from H1 to H3, insert logical H2 level
                if (prev_heading["level"] == "H1" and heading["level"] == "H3" and 
                    heading["page"] == prev_heading["page"]):
                    # Adjust to H2 for better hierarchy
                    heading["level"] = "H2"
            
            validated.append(heading)
        
        return validated

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
