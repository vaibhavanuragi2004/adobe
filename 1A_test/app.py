import os
import json
import logging
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pdf_analyzer import PDFAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with file upload form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    """Handle PDF file upload and analysis."""
    try:
        # Check if file was uploaded
        if 'pdf_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['pdf_file']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check file type
        if not allowed_file(file.filename):
            flash('Please upload a PDF file', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize PDF analyzer
        analyzer = PDFAnalyzer()
        
        # Analyze the PDF
        try:
            app.logger.info(f"Starting analysis of {filename}")
            result = analyzer.analyze_pdf(filepath)
            app.logger.info(f"Analysis result: {result}")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result and result.get('title'):
                app.logger.info(f"Successfully analyzed {filename}, rendering result page")
                return render_template('result.html', 
                                     result=result, 
                                     filename=filename,
                                     json_output=json.dumps(result, indent=2),
                                     json_data=result)
            else:
                app.logger.warning(f"Analysis returned invalid result for {filename}: {result}")
                flash('Failed to analyze PDF. The file might be corrupted or contain no readable text.', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"Error analyzing PDF: {str(e)}")
            flash(f'Error analyzing PDF: {str(e)}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        flash('An error occurred during file upload', 'error')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    """Result page route - redirects to main page if accessed directly."""
    flash('Please upload a PDF file to see results', 'info')
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze_pdf():
    """API endpoint for PDF analysis (returns JSON)."""
    try:
        # Check if file was uploaded
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['pdf_file']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Please upload a PDF file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize PDF analyzer
        analyzer = PDFAnalyzer()
        
        # Analyze the PDF
        try:
            result = analyzer.analyze_pdf(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result:
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to analyze PDF. The file might be corrupted or contain no readable text.'}), 400
                
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"Error analyzing PDF: {str(e)}")
            return jsonify({'error': f'Error analyzing PDF: {str(e)}'}), 500
            
    except Exception as e:
        app.logger.error(f"API upload error: {str(e)}")
        return jsonify({'error': 'An error occurred during file upload'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Please upload a PDF file smaller than 50MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
