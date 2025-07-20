# PDF Analyzer - Replit Configuration

## Overview

This is a Flask-based PDF analysis web application that extracts titles and hierarchical headings from PDF documents. The application uses PyMuPDF for PDF processing and provides a web interface for users to upload PDFs and view structured analysis results.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Bootstrap 5 with dark theme
- **Styling**: Uses Bootstrap CDN for responsive design and Font Awesome for icons
- **JavaScript**: Vanilla JavaScript for client-side file validation and UI interactions
- **Templates**: Jinja2 templating with Flask for server-side rendering

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Structure**: Simple MVC pattern with route handlers in `app.py`
- **PDF Processing**: Custom `PDFAnalyzer` class using PyMuPDF (fitz) library
- **File Handling**: Werkzeug utilities for secure file uploads

### Key Design Decisions
- **Single-page application flow**: Upload → Process → Results display
- **Stateless processing**: No database persistence, files processed on-demand
- **Memory-efficient**: Files processed directly from uploads folder
- **Safety limits**: 50MB file size limit, 50-page document limit

## Key Components

### 1. Flask Application (`app.py`)
- Main application entry point
- Route handlers for file upload and PDF analysis
- File validation and security measures
- Flash messaging for user feedback

### 2. PDF Analyzer (`pdf_analyzer.py`)
- Core PDF processing logic using PyMuPDF
- Font analysis for intelligent heading detection
- Hierarchical structure extraction
- Text block analysis with positioning data

### 3. Frontend Templates
- **`index.html`**: Upload form with drag-and-drop interface
- **`result.html`**: Analysis results display with structured outline
- **`upload.js`**: Client-side validation and UI enhancements

### 4. File Upload System
- Secure filename handling with Werkzeug
- File type validation (PDF only)
- Size limitations for performance
- Temporary storage in uploads directory

## Data Flow

1. **Upload Phase**:
   - User selects PDF file through web interface
   - Client-side validation checks file type and size
   - File uploaded to server's uploads directory

2. **Processing Phase**:
   - PDFAnalyzer extracts text blocks with formatting data
   - Font characteristics analyzed across document
   - Title extraction using largest/prominent fonts
   - Heading hierarchy determined by font size patterns

3. **Results Phase**:
   - Structured data (title + outline) returned to template
   - Results displayed with heading levels and page numbers
   - JSON export functionality available

## External Dependencies

### Python Packages
- **Flask**: Web framework for HTTP handling and templating
- **PyMuPDF (fitz)**: PDF document processing and text extraction
- **Werkzeug**: Secure file handling utilities

### Frontend Dependencies
- **Bootstrap 5**: CSS framework via CDN
- **Font Awesome**: Icon library via CDN
- **Prism.js**: Code syntax highlighting for JSON display

### System Requirements
- Python 3.7+ for PDF processing capabilities
- File system access for temporary upload storage

## Deployment Strategy

### Development Setup
- Flask development server with debug mode enabled
- Hot reloading for code changes
- Local file storage in uploads directory

### Production Considerations
- Environment variable for session secret key
- File cleanup strategy for uploads directory
- Error handling and logging configuration
- WSGI server deployment (Gunicorn recommended)

### Security Measures
- File extension validation
- Secure filename generation
- File size limits to prevent DoS attacks
- Session security with configurable secret key

## Configuration

### Environment Variables
- `SESSION_SECRET`: Flask session encryption key (defaults to dev key)

### Upload Limits
- Maximum file size: 50MB
- Allowed extensions: PDF only
- Maximum pages per document: 50 pages

### Storage
- Upload directory: `./uploads/` (auto-created)
- No persistent data storage (stateless processing)