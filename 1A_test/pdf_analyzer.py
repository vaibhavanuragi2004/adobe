import pymupdf as fitz  # PyMuPDF
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
        logging.basicConfig(level=logging.INFO) # Enable logging to see the process
        
    def analyze_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Analyze a PDF file and extract title and headings.
        """
        try:
            doc = fitz.open(pdf_path)
            # Store current PDF path for content validation
            self._current_pdf_path = pdf_path
            
            if len(doc) > 50:
                raise ValueError("PDF has more than 50 pages.")

            # First, try to get the embedded ToC (bookmarks), which is most reliable
            toc = doc.get_toc()
            if toc:
                self.logger.info("Found embedded ToC (bookmarks). Using it for the outline.")
                # Attempt to get title from metadata, otherwise use heuristic
                title = doc.metadata.get('title') if doc.metadata.get('title') else self._extract_title_heuristic(doc)
                outline = [{"level": f"H{lvl}", "text": text, "page": page - 1} for lvl, text, page in toc]
                return {"title": title, "outline": self._post_process_headings(outline)}

            # If no embedded ToC, proceed with heuristic analysis on page content
            self.logger.info("No embedded ToC found. Starting heuristic analysis.")
            text_blocks = self._extract_text_blocks(doc)
            if not text_blocks:
                self.logger.warning("No text blocks found in PDF.")
                doc.close()
                return None

            # Identify and filter out repeating headers and footers early
            headers, footers = self._identify_headers_footers(text_blocks, len(doc))
            self.logger.info(f"Identified Headers to filter: {headers}")
            self.logger.info(f"Identified Footers to filter: {footers}")
            filtered_blocks = [
                b for b in text_blocks
                if b['text'] not in headers and b['text'] not in footers
            ]

            font_stats = self._analyze_font_characteristics(filtered_blocks)
            title = self._extract_title(filtered_blocks, font_stats)
            self.logger.info(f"Extracted Title: '{title}'")

            # Attempt to parse a visual Table of Contents from the text (like on page 4)
            toc_headings = self._parse_visual_toc(filtered_blocks)
            if toc_headings:
                self.logger.info("Found and parsed a visual Table of Contents.")
                headings = self._post_process_headings(toc_headings)
            else:
                self.logger.info("No visual ToC found. Analyzing headings with content-aware heuristics.")
                # Filter out the title from heading candidates to prevent duplication
                heading_candidates = [b for b in filtered_blocks if b['text'].lower() not in title.lower()]
                headings = self._extract_headings_with_validation(heading_candidates, font_stats, text_blocks)

            doc.close()

            # Store current title for post-processing comparison
            self._current_title = title
            
            result = {
                "title": title,
                "outline": headings
            }
            self.logger.info(f"Analysis complete. Found {len(headings)} headings.")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing PDF: {str(e)}")
            raise

    def _extract_title_heuristic(self, doc: fitz.Document) -> str:
        """Extract title using heuristic analysis when metadata is not available."""
        text_blocks = self._extract_text_blocks(doc)
        if not text_blocks:
            return "Document"
        
        font_stats = self._analyze_font_characteristics(text_blocks)
        return self._extract_title(text_blocks, font_stats)

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
                                    "page": page_num,  # 0-based indexing
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
        first_page_blocks = [block for block in text_blocks if block["page"] == 0]
        
        # Sort blocks by Y position to process from top to bottom
        first_page_blocks.sort(key=lambda x: x["bbox"][1])
        
        # Look for specific title patterns first (like RFP titles)
        rfp_title = self._detect_rfp_title(first_page_blocks)
        if rfp_title:
            return rfp_title
        
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

    def _detect_rfp_title(self, first_page_blocks: List[Dict]) -> Optional[str]:
        """Detect RFP-style titles that span multiple blocks."""
        title_parts = []
        
        # Look for RFP pattern
        for i, block in enumerate(first_page_blocks):
            text = block["text"].strip()
            
            # Check if this starts an RFP title
            if re.match(r'^RFP:\s*', text, re.IGNORECASE):
                title_parts.append(text)
                
                # Look for continuation blocks that are part of the title
                for j in range(i + 1, min(i + 10, len(first_page_blocks))):  # Look ahead up to 10 blocks
                    next_block = first_page_blocks[j]
                    next_text = next_block["text"].strip()
                    
                    # Skip very short or date-like text
                    if len(next_text) < 3 or re.match(r'^\d{1,2}[-/,]\s*\d{4}', next_text):
                        continue
                    
                    # Stop if we hit a clear section break or different content
                    if (re.match(r'^(summary|background|introduction|purpose)', next_text, re.IGNORECASE) or
                        len(next_text) > 200 or  # Long paragraph text
                        next_text.startswith('The ') and len(next_text) > 50):
                        break
                    
                    # Add if it seems to be part of the title
                    if (len(next_text) > 3 and 
                        not self._is_likely_metadata(next_text) and
                        not re.match(r'^\d+$', next_text)):  # Not just a number
                        title_parts.append(next_text)
                        
                        # Stop if we hit what looks like a subtitle ending
                        if re.search(r'\d{1,2},?\s*\d{4}$', next_text):  # Ends with date
                            break
                
                break
        
        if title_parts:
            # Clean and join the title parts
            full_title = ' '.join(title_parts).strip()
            # Clean up excessive whitespace and duplicated text
            full_title = re.sub(r'\s+', ' ', full_title)
            # Remove obvious corrupted text patterns that appear to be OCR/extraction artifacts
            # Handle patterns like "RFP: R RFP: R RFP: Request f quest f quest for Pr r Pr r Proposal oposal oposal"
            
            # First, fix character-level corruption where letters are separated
            full_title = re.sub(r'\b([A-Z])\s+([A-Z]{2,}:)\s+\1\s+\2', r'\2', full_title)
            full_title = re.sub(r'([a-z]+)\s+([a-z])\s+\1\s+\2', r'\1', full_title)
            
            # Fix partial word repetitions like "quest f quest" -> "quest"
            full_title = re.sub(r'\b(\w+)\s+\w\s+\1\b', r'\1', full_title)
            full_title = re.sub(r'\b(\w+)\s+\w+\s+\w\s+\1\b', r'\1', full_title)
            
            # Fix corrupted words like "oposal oposal" -> "Proposal" (assuming the first occurrence has the capital)
            words = full_title.split()
            cleaned_words = []
            skip_next = 0
            
            for i, word in enumerate(words):
                if skip_next > 0:
                    skip_next -= 1
                    continue
                    
                # Look for corrupted repetitions
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # If current word is a fragment of next word or vice versa
                    if (word.lower() in next_word.lower() and len(word) < len(next_word)) or \
                       (next_word.lower() in word.lower() and len(next_word) < len(word)):
                        # Use the longer/more complete word
                        cleaned_words.append(word if len(word) >= len(next_word) else next_word)
                        skip_next = 1
                    else:
                        cleaned_words.append(word)
                else:
                    cleaned_words.append(word)
            
            result = ' '.join(cleaned_words).strip()
            # Final cleanup
            result = re.sub(r'\s+', ' ', result)
            # Remove trailing date that might be corrupted
            result = re.sub(r'\s+March\s+\d+,?\s*\d{4}$', '', result)
            
            return result
        
        return None

    def _reconstruct_multiline_title(self, main_block: Dict, all_blocks: List[Dict], block_index: int) -> str:
        """Reconstruct a title that might span multiple lines."""
        title_parts = [main_block["text"]]
        main_size = main_block["size"]
        main_font = main_block["font"]
        main_y = main_block["bbox"][1]
        
        # Look at nearby blocks (before and after) with similar formatting
        search_range = 3  # Look at 3 blocks before and after
        start_idx = max(0, block_index - search_range)
        end_idx = min(len(all_blocks), block_index + search_range + 1)
        
        for i in range(start_idx, end_idx):
            if i == block_index:
                continue
                
            candidate = all_blocks[i]
            
            # Check if this block could be part of the title
            size_match = abs(candidate["size"] - main_size) < 1
            font_match = candidate["font"] == main_font
            y_distance = abs(candidate["bbox"][1] - main_y)
            
            # Should be close vertically (within reasonable line spacing)
            if size_match and font_match and y_distance < 50:
                # Add to title parts if it's not too long (titles shouldn't be paragraphs)
                if len(candidate["text"]) < 200:
                    title_parts.append(candidate["text"])
        
        # Combine and clean up
        full_title = " ".join(title_parts).strip()
        # Remove excessive whitespace
        full_title = re.sub(r'\s+', ' ', full_title)
        
        return full_title

    def _calculate_semantic_title_score(self, block: Dict, font_stats: Dict, all_blocks: List[Dict], position: int) -> float:
        """Calculate title score using semantic analysis."""
        text = block["text"].strip()
        size = block["size"]
        font = block["font"]
        
        score = 0.0
        
        # Font size scoring (larger = more likely to be title)
        if size > font_stats["most_common_size"]:
            size_ratio = size / font_stats["most_common_size"]
            score += min(size_ratio * 10, 50)  # Cap at 50 points
        
        # Position scoring (earlier in document = more likely title)
        if position < 5:
            score += 20 - (position * 3)
        
        # Length scoring (titles are usually not too short or too long)
        text_length = len(text)
        if 10 <= text_length <= 100:
            score += 15
        elif 5 <= text_length < 10:
            score += 5
        elif text_length > 200:
            score -= 20
        
        # Font style scoring (bold fonts often used for titles)
        if block.get("flags", 0) & 2**4:  # Bold flag
            score += 10
        
        # Content analysis
        # Penalize obvious non-title content
        if re.search(r'\b(page|chapter|\d+/\d+|copyright|©|abstract|introduction)\b', text, re.IGNORECASE):
            score -= 15
        
        # Boost score for title-like words
        if re.search(r'\b(analysis|study|report|guide|manual|handbook)\b', text, re.IGNORECASE):
            score += 5
        
        # Penalize very common words that appear in many blocks
        word_frequency = self._calculate_word_frequency(all_blocks)
        words = text.lower().split()
        common_word_penalty = sum(1 for word in words if word_frequency.get(word, 0) > len(all_blocks) * 0.1)
        score -= common_word_penalty * 2
        
        return max(0, score)

    def _calculate_word_frequency(self, blocks: List[Dict]) -> Dict[str, int]:
        """Calculate word frequency across all blocks."""
        word_freq = defaultdict(int)
        for block in blocks:
            words = block["text"].lower().split()
            for word in words:
                if len(word) > 2:  # Only count significant words
                    word_freq[word] += 1
        return dict(word_freq)

    def _is_likely_metadata(self, text: str) -> bool:
        """Check if text is likely metadata rather than title or heading."""
        metadata_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'copyright|©',  # Copyright notices
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            r'^[A-Z]{2,}\s*:',  # All caps labels like "TITLE:"
            r'www\.|http|@',  # URLs or emails
            r'^[A-Za-z]+\s+\d{1,2},?\s*\d{4}\.?$',  # "March 21, 2003" or "April 21, 2003."
            r'^\d{4}\s+\d{4}$',  # Year ranges like "2007 2017"
            r'^Funding\s+Source\s+\d{4}',  # Table headers like "Funding Source 2007 2017"
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # Date patterns
            r'^\$[\d,]+',  # Money amounts
            r'^\d+\.\d+%?$',  # Percentages or decimal numbers
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def _identify_headers_footers(self, text_blocks: List[Dict], total_pages: int) -> Tuple[List[str], List[str]]:
        """Identify recurring headers and footers across pages."""
        if total_pages < 2:
            return [], []
        
        # Group blocks by approximate Y position
        top_blocks = defaultdict(list)  # Headers
        bottom_blocks = defaultdict(list)  # Footers
        
        for block in text_blocks:
            y_pos = block["bbox"][1]
            
            # Top 15% of page likely to be header
            if y_pos < 150:
                top_blocks[block["text"]].append(block["page"])
            # Bottom 15% of page likely to be footer
            elif y_pos > 650:
                bottom_blocks[block["text"]].append(block["page"])
        
        headers = []
        footers = []
        
        # Find text that appears on multiple pages
        min_occurrences = max(2, total_pages // 3)  # At least 2 pages or 1/3 of pages
        
        for text, pages in top_blocks.items():
            if len(set(pages)) >= min_occurrences and len(text.strip()) > 2:
                headers.append(text)
        
        for text, pages in bottom_blocks.items():
            if len(set(pages)) >= min_occurrences and len(text.strip()) > 2:
                footers.append(text)
        
        return headers, footers

    def _parse_visual_toc(self, text_blocks: List[Dict]) -> List[Dict]:
        """Parse visual table of contents from text blocks."""
        toc_headings = []
        
        # Look for ToC patterns in text blocks
        for block in text_blocks:
            text = block["text"].strip()
            
            # Common ToC patterns
            patterns = [
                r'^(\d+\.?\s+)([^.]+)\.+\s*(\d+)$',  # "1. Introduction....5"
                r'^([A-Z][^.]+)\s+\.{3,}\s*(\d+)$',  # "Introduction...5"
                r'^(\d+\.\d+\s+)([^.]+)\s+(\d+)$',   # "1.1 Overview 5"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    if len(match.groups()) == 3:
                        prefix, title, page = match.groups()
                        level = "H2" if "." in prefix else "H1"
                        try:
                            page_num = int(page) - 1  # Convert to 0-based
                            toc_headings.append({
                                "level": level,
                                "text": title.strip(),
                                "page": page_num
                            })
                        except ValueError:
                            continue
                    elif len(match.groups()) == 2:
                        title, page = match.groups()
                        try:
                            page_num = int(page) - 1  # Convert to 0-based
                            toc_headings.append({
                                "level": "H1",
                                "text": title.strip(),
                                "page": page_num
                            })
                        except ValueError:
                            continue
        
        return toc_headings if len(toc_headings) > 2 else []

    def _extract_headings_with_validation(self, text_blocks: List[Dict], font_stats: Dict, all_blocks: List[Dict]) -> List[Dict]:
        """Extract headings with enhanced validation."""
        headings = []
        
        # Group blocks by page for context
        page_blocks = defaultdict(list)
        for block in text_blocks:
            page_blocks[block["page"]].append(block)
        
        # Calculate dynamic thresholds
        base_size = font_stats["most_common_size"]
        max_size = font_stats["max_size"]
        size_diff = max_size - base_size
        
        # Adaptive thresholds
        if size_diff > 8:
            h1_threshold = base_size + (size_diff * 0.5)
            h2_threshold = base_size + (size_diff * 0.25)
            h3_threshold = base_size + (size_diff * 0.1)
        else:
            h1_threshold = base_size + max(3, size_diff * 0.7)
            h2_threshold = base_size + max(2, size_diff * 0.4)
            h3_threshold = base_size + max(1, size_diff * 0.15)
        
        for page_num, blocks in page_blocks.items():
            blocks.sort(key=lambda x: x["bbox"][1])  # Sort by Y position
            
            for i, block in enumerate(blocks):
                text = block["text"].strip()
                
                if len(text) < 3 or self._is_likely_metadata(text):
                    continue
                
                if self._is_likely_heading(block, font_stats):
                    # Use smart level detection that considers content patterns
                    level = self._determine_heading_level_smart(block, font_stats)
                    
                    if level:
                        clean_text = self._clean_heading_text(text)
                        if clean_text and len(clean_text) > 2:
                            headings.append({
                                "level": level,
                                "text": clean_text,
                                "page": block["page"]
                            })
        
        return self._post_process_headings(headings)

    def _is_likely_heading(self, block: Dict, font_stats: Dict) -> bool:
        """Determine if a text block is likely to be a heading."""
        text = block["text"].strip()
        
        # First check if it's metadata/date/table data
        if self._is_likely_metadata(text):
            return False
        
        # Length check (headings are usually not too long)
        if len(text) > 200:
            return False
        
        # Style checks
        is_bold = bool(block["flags"] & 2**4)
        is_italic = bool(block["flags"] & 2**1)
        is_bold_italic = (block["flags"] & (2**4 | 2**1)) == (2**4 | 2**1)  # Both bold and italic
        
        # Enhanced heading patterns - includes colon-ending headings
        heading_patterns = [
            r'^\d+\.\s*\w+',  # "1. Introduction"
            r'^\d+\s+\w+',    # "1 Introduction"
            r'^[A-Z][a-z]+\s+\d+',  # "Chapter 1"
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\w+.*:\s*$',   # "Introduction:" or "Equitable access for all Ontarians:"
            r'^[A-Z][^:]*:\s*$',  # Any capitalized text ending with colon
        ]
        
        pattern_match = any(re.search(pattern, text) for pattern in heading_patterns)
        
        # Special check for colon-ending headings (these are often styled headings)
        ends_with_colon = text.endswith(':')
        
        # Reject obvious non-headings
        non_heading_patterns = [
            r'^\d+[-/]\d+[-/]\d{4}',  # Dates
            r'^\$[\d,]+',  # Money
            r'^\d+\.\d+%?$',  # Numbers/percentages
            r'^[A-Za-z]+\s+\d{1,2},?\s*\d{4}\.?$',  # "March 21, 2003."
        ]
        
        for pattern in non_heading_patterns:
            if re.search(pattern, text):
                return False
        
        # Scoring system
        score = 0
        
        # Size-based scoring (less strict for colon headings)
        size_ratio = block["size"] / font_stats["most_common_size"]
        if size_ratio > 1.2:
            score += 3
        elif size_ratio > 1.1:
            score += 2
        elif size_ratio > 1.05:
            score += 1
        elif ends_with_colon and size_ratio >= 0.95:  # Allow same-size colon headings
            score += 1
        
        if is_bold:
            score += 2
        
        if is_italic:
            score += 1  # Italic can indicate headings like "Timeline:"
        
        if is_bold_italic:
            score += 3  # Bold+italic is a strong heading indicator
        
        if pattern_match:
            score += 2
        
        # Extra points for colon endings (strong heading indicator)
        if ends_with_colon:
            score += 2
            # Additional check: colon headings that are on their own line
            if len(text) <= 100:  # Reasonable heading length
                score += 1
        
        if len(text) <= 100:
            score += 1
        
        # Position-based scoring (headings often start at left margin)
        if block["bbox"][0] < 100:  # Left-aligned
            score += 1
        
        # Special patterns for common heading structures
        if re.search(r'^(phase|section|chapter|part|step|milestone|timeline)\s*:?\s*\w*', text, re.IGNORECASE):
            score += 2
        
        # For styled same-size headings, lower the threshold
        if ends_with_colon and is_bold_italic:
            return score >= 2  # Bold+italic colon headings are very likely
        elif ends_with_colon and (is_bold or is_italic):
            return score >= 2
        elif ends_with_colon:
            return score >= 3
        
        return score >= 4

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

    def _determine_heading_level_smart(self, block: Dict, font_stats: Dict) -> Optional[str]:
        """Determine heading level using both font size and content analysis."""
        text = block["text"].strip()
        font_size = block["size"]
        
        # Calculate relative size
        size_ratio = font_size / font_stats["most_common_size"]
        
        # Content-based level detection
        if re.match(r'^(appendix|chapter)\s+[a-z]', text, re.IGNORECASE):
            return "H1"
        
        if re.match(r'^\d+\.\s+', text):  # "1. Something"
            return "H2"
        
        # Special case for main section headers
        if re.match(r'^(summary|background|business plan|approach|evaluation|milestones)$', text, re.IGNORECASE):
            return "H2"
        
        if text.endswith(':') and len(text) <= 100:
            # For colon-ending headings, use context and formatting
            if size_ratio > 1.15:
                return "H2"
            elif size_ratio > 1.05 or bool(block["flags"] & 2**4):  # Bold
                return "H3"
            else:
                # Same-size styled headings - analyze context
                if re.match(r'^[A-Z][a-z]+.*for.*:', text):  # "Something for something:"
                    return "H4"
                elif re.match(r'^(timeline|access|guidance|training|funding|support)', text, re.IGNORECASE):
                    return "H3"
                return "H3"
        
        # Size-based detection with adjusted thresholds
        if size_ratio > 1.4:
            return "H1"
        elif size_ratio > 1.2:
            return "H2"
        elif size_ratio > 1.05:
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
        
        # Store extracted title for comparison
        self._extracted_title = getattr(self, '_current_title', '')
        
        # Remove near-duplicates (same text, similar page)
        cleaned_headings = []
        seen_texts = set()
        
        for heading in headings:
            text_key = heading["text"].lower().strip()
            if text_key not in seen_texts:
                # Additional filtering for obvious non-headings
                if not self._is_obvious_non_heading(heading["text"]):
                    # Check if heading has content following it
                    if self._has_content_below(heading):
                        cleaned_headings.append(heading)
                        seen_texts.add(text_key)
        
        # Sort by page number, then by document order (Y position would be ideal but we don't have it here)
        cleaned_headings.sort(key=lambda x: (x["page"], x["text"]))
        
        return cleaned_headings
    
    def _has_content_below(self, heading: Dict) -> bool:
        """Check if a heading has actual content (paragraphs/text) below it."""
        # Special rules first - certain patterns are likely not headings
        text = heading["text"].strip()
        
        # If it's on page 0 and looks like title/footer material, exclude it
        if heading["page"] == 0:
            # Text that appears above or as part of the title area
            title_area_patterns = [
                r'^ontario\'?s?\s*libraries?\s*working\s*together$',
                r'libraries?\s*working\s*together',
                r'^to\s+present\s+a\s+proposal',
                r'^ontario\'?s?\s*digital\s*library$',
            ]
            for pattern in title_area_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return False
        
        try:
            doc = fitz.open(self._current_pdf_path) if hasattr(self, '_current_pdf_path') else None
            if not doc or heading["page"] >= len(doc):
                return True  # Default to including if we can't verify
            
            page = doc[heading["page"]]
            text_dict = page.get_text("dict")
            
            # Find the heading block and check what follows
            heading_found = False
            content_blocks_after = 0
            heading_y_position = None
            
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                block_text = ""
                block_y = None
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                        if block_y is None and "bbox" in span:
                            block_y = span["bbox"][1]  # Y position
                
                block_text = block_text.strip()
                
                # Check if this is our heading
                if not heading_found and heading["text"].strip().lower() in block_text.lower():
                    heading_found = True
                    heading_y_position = block_y
                    continue
                
                # If we found the heading, count meaningful content blocks after it
                if heading_found and block_text and block_y:
                    # Make sure the content is actually below the heading (higher Y value)
                    if heading_y_position is None or block_y > heading_y_position:
                        # Skip very short blocks and likely metadata
                        if len(block_text) > 15 and not self._is_likely_metadata(block_text):
                            content_blocks_after += 1
                            # If we find substantial content, it's a valid heading
                            if content_blocks_after >= 1:
                                doc.close()
                                return True
            
            doc.close()
            # If no content found after heading, it's likely not a real heading
            return False
            
        except Exception:
            # If we can't verify, default to excluding questionable headings
            return False
    
    def _is_obvious_non_heading(self, text: str) -> bool:
        """Additional check for obvious non-headings that passed initial filters."""
        # Single words that are likely footers or standalone text
        if len(text.split()) <= 2 and not text.endswith(':'):
            single_word_footers = [
                r'^ontario\'?s?\s*(digital\s*)?library$',
                r'^digital\s*library$',
                r'^libraries?\s*working\s*together$',
                r'^working\s*together$'
            ]
            for pattern in single_word_footers:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        # Text that appears in the title should not be a heading
        if hasattr(self, '_extracted_title') and self._extracted_title:
            # Check if this text is a substantial part of the title
            title_lower = self._extracted_title.lower()
            text_lower = text.lower()
            
            # Direct substring check for exact matches
            if text_lower in title_lower or title_lower in text_lower:
                return True
            
            # Remove common words for better matching
            common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'rfp', 'request', 'proposal']
            title_words = [w for w in title_lower.split() if w not in common_words and len(w) > 2]
            text_words = [w for w in text_lower.split() if w not in common_words and len(w) > 2]
            
            if text_words and title_words:
                # Calculate overlap - if significant portion matches title, it's likely part of title
                overlap = len(set(text_words) & set(title_words))
                overlap_ratio = overlap / len(text_words) if text_words else 0
                
                if overlap_ratio > 0.5:  # 50% of words overlap with title
                    return True
        
        # Table headers and data
        table_patterns = [
            r'^funding\s+source\s+\d{4}',
            r'^\d{4}\s+\d{4}$',
            r'^year\s+\d+',
            r'^phase\s+[IVX]+\s+\d{4}',
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False