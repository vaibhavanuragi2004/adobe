import fitz  # PyMuPDF
import re
from collections import defaultdict
from typing import List, Dict, Optional

class PDFAnalyzer:
    """
    PDF analyzer that extracts titles and hierarchical headings from PDF documents.
    Uses font size, style, and positioning analysis for intelligent heading detection.
    This is the self-contained module for Challenge 1A.
    """
    
    def __init__(self):
        # In a real app, you might add logging here
        pass
        
    def analyze_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Main method to analyze a PDF file and extract its structure.
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > 50:
                raise ValueError("PDF has more than 50 pages.")
            
            # This is the core pipeline
            text_blocks = self._extract_text_blocks(doc)
            if not text_blocks:
                doc.close()
                return {"title": "Empty Document", "outline": []}
            
            font_stats = self._analyze_font_characteristics(text_blocks)
            title = self._extract_title(text_blocks, font_stats)
            headings = self._extract_headings(text_blocks, font_stats, title)
            
            doc.close()
            
            return {"title": title, "outline": headings}
            
        except Exception as e:
            print(f"Error analyzing PDF {pdf_path}: {e}")
            raise

    # --- Step 1: Text Block Extraction and Grouping ---

    def _extract_text_blocks(self, doc: fitz.Document) -> List[Dict]:
        """
        Extracts and groups text into semantic blocks. This is a crucial pre-processing step.
        It merges spans into lines and consecutive lines into logical blocks.
        """
        text_blocks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    block_lines = []
                    for line in block["lines"]:
                        line_text_parts = []
                        line_bbox, line_font, line_size, line_flags = None, None, None, None
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text_parts.append(text)
                                if line_font is None or span["size"] > (line_size or 0):
                                    line_font, line_size, line_flags, line_bbox = span["font"], span["size"], span["flags"], span["bbox"]
                        
                        if line_text_parts:
                            full_line_text = " ".join(line_text_parts)
                            if len(full_line_text) > 1:
                                block_lines.append({
                                    "text": full_line_text, "page": page_num, "font": line_font,
                                    "size": line_size, "flags": line_flags, "bbox": line_bbox,
                                    "line_y": line_bbox[1] if line_bbox else 0
                                })
                    
                    if block_lines:
                        current_group = [block_lines[0]]
                        for i in range(1, len(block_lines)):
                            current, prev = block_lines[i], current_group[-1]
                            size_similar = abs(current["size"] - prev["size"]) < 1
                            font_similar = current["font"] == prev["font"]
                            y_close = abs(current["line_y"] - prev["line_y"]) < 50
                            
                            if size_similar and font_similar and y_close:
                                current_group.append(current)
                            else:
                                if current_group:
                                    self._add_grouped_text_block(current_group, text_blocks)
                                current_group = [current]
                        
                        if current_group:
                            self._add_grouped_text_block(current_group, text_blocks)
        return text_blocks

    def _add_grouped_text_block(self, line_group: List[Dict], text_blocks: List[Dict]):
        """Helper to combine a group of lines into a single logical block."""
        if not line_group: return
            
        combined_text = " ".join(line["text"] for line in line_group).strip()
        first_line = line_group[0]
        
        min_x = min(line["bbox"][0] for line in line_group)
        min_y = min(line["bbox"][1] for line in line_group)  
        max_x = max(line["bbox"][2] for line in line_group)
        max_y = max(line["bbox"][3] for line in line_group)
        
        is_bold = bool(first_line["flags"] & 2**4)
        is_italic = bool(first_line["flags"] & 2**1)
        
        text_blocks.append({
            "text": combined_text, "page": first_line["page"], "font": first_line["font"],
            "size": first_line["size"], "flags": first_line["flags"], "is_bold": is_bold,
            "is_italic": is_italic, "bbox": (min_x, min_y, max_x, max_y)
        })

    # --- Step 2: Font Analysis ---

    def _analyze_font_characteristics(self, text_blocks: List[Dict]) -> Dict:
        """Analyzes font sizes to find the most common (body text) size."""
        if not text_blocks: return {"most_common_size": 12.0} # Default
        
        size_frequency = defaultdict(int)
        for block in text_blocks:
            size_frequency[round(block["size"], 1)] += 1
        
        most_common_size = max(size_frequency, key=size_frequency.get)
        return {"most_common_size": most_common_size}

    # --- Step 3: Title Extraction ---

    def _extract_title(self, text_blocks: List[Dict], font_stats: Dict) -> str:
        """Extracts the document title, prioritizing the largest font on the first page."""
        candidates = []
        first_page_blocks = [block for block in text_blocks if block["page"] == 0]
        
        max_size_on_page = 0
        if first_page_blocks:
            max_size_on_page = max(b["size"] for b in first_page_blocks)
        
        first_page_blocks.sort(key=lambda x: x["bbox"][1])
        
        for i, block in enumerate(first_page_blocks):
            text = block["text"].strip()
            if len(text) < 3 or self._is_likely_metadata(text, is_title_candidate=True):
                continue
            
            score = self._calculate_semantic_title_score(block, font_stats)
            if max_size_on_page > 0 and abs(block["size"] - max_size_on_page) < 1:
                score += 10
                
            if score > 0:
                full_title = self._reconstruct_multiline_title(block, first_page_blocks, i)
                candidates.append({"text": full_title, "score": score})
        
        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return re.sub(r'\s+', ' ', candidates[0]["text"].strip())
        
        # Fallback
        for block in first_page_blocks:
            text = block["text"].strip()
            if len(text) > 5 and not self._is_likely_metadata(text):
                return text
        return "Document"

    def _calculate_semantic_title_score(self, block: Dict, font_stats: Dict) -> float:
        """Scores a block on its likelihood of being a title."""
        score = 0.0
        text, size = block["text"], block["size"]
        size_ratio = size / font_stats.get("most_common_size", 12)

        if size_ratio > 1.5: score += 5
        if block.get("is_bold", False): score += 2
        if block["bbox"][1] < 150: score += 4 # High on page
        
        text_length = len(text)
        if 10 <= text_length <= 150: score += 2
        elif text_length < 5 or text_length > 200: score -= 3

        return score

    def _reconstruct_multiline_title(self, main_block: Dict, page_blocks: List[Dict], block_index: int) -> str:
        """Combines vertically close lines with similar formatting into a single title."""
        title_parts = [main_block["text"]]
        for i in range(block_index + 1, min(block_index + 3, len(page_blocks))):
            candidate = page_blocks[i]
            y_distance = candidate["bbox"][1] - page_blocks[i-1]["bbox"][3]
            size_similar = abs(candidate["size"] - main_block["size"]) < 3
            if y_distance < 20 and size_similar and not self._is_likely_metadata(candidate['text']):
                title_parts.append(candidate["text"])
            else:
                break
        return " ".join(title_parts)

    # --- Step 4: Heading Extraction ---

    def _extract_headings(self, text_blocks: List[Dict], font_stats: Dict, title: str) -> List[Dict]:
        """Extracts headings by scoring blocks and applying post-processing."""
        headings = []
        pages = defaultdict(list)
        for block in text_blocks:
            # Exclude blocks that are part of the main title
            if block["page"] == 0 and block["text"] in title:
                continue
            pages[block["page"]].append(block)

        for page_num, page_blocks in pages.items():
            page_blocks.sort(key=lambda x: x["bbox"][1])
            for i, block in enumerate(page_blocks):
                if self._is_semantic_heading(block, font_stats):
                    size_ratio = block["size"] / font_stats["most_common_size"]
                    
                    if size_ratio > 1.4: level = "H1"
                    elif size_ratio > 1.2: level = "H2"
                    else: level = "H3"
                    
                    headings.append({"level": level, "text": block["text"], "page": page_num})
        
        return self._post_process_headings(headings)

    def _is_semantic_heading(self, block: Dict, font_stats: Dict) -> bool:
        """Scores a block on its likelihood of being a heading."""
        text = block["text"].strip()
        size_ratio = block["size"] / font_stats.get("most_common_size", 12)
        is_bold = block.get("is_bold", False)
        is_italic = block.get("is_italic", False)
        
        if size_ratio < 1.05 and not (is_bold or is_italic):
            return False
        if len(text) > 200 or len(text) < 3 or self._is_likely_metadata(text):
            return False
        
        score = 0
        if size_ratio > 1.1: score += (size_ratio - 1.0) * 5
        if is_bold: score += 2
        if is_italic: score += 2
        if text.endswith(':'): score += 2
        if len(text.split()) < 10: score += 1
            
        return score >= 3

    # --- Step 5: Post-Processing and Helpers ---

    def _is_likely_metadata(self, text: str, is_title_candidate: bool = False) -> bool:
        """Checks if text is likely non-content (e.g., page numbers, dates)."""
        text_lower = text.lower().strip()
        if not text_lower: return True

        # For titles, be more lenient with dates
        date_pattern = r'^[a-z]+\s+\d{1,2},?\s+\d{4}$'
        if is_title_candidate and re.match(date_pattern, text_lower):
            return False # A date can be a subtitle

        patterns = [
            r'^page\s+\d+', r'^\d+$', r'©|copyright',
            date_pattern, r'www\.', r'\.com'
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _post_process_headings(self, headings: List[Dict]) -> List[Dict]:
        """Cleans up the final list of headings, removing duplicates."""
        if not headings: return []
        
        unique_headings = []
        seen = set()
        for heading in headings:
            key = (heading["text"], heading["page"])
            if key not in seen:
                unique_headings.append(heading)
                seen.add(key)
        
        # Here you could add hierarchy validation if needed
        return unique_headings

        return cleaned_headings
