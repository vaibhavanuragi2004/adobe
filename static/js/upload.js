document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const fileInput = document.getElementById('pdf_file');

    // File validation
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            validateFile(file);
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        
        if (!file) {
            e.preventDefault();
            showAlert('Please select a PDF file', 'error');
            return;
        }

        if (!validateFile(file)) {
            e.preventDefault();
            return;
        }

        // Show loading modal
        showLoadingState();
        
        // Allow the form to submit normally - don't prevent default
        // The server will handle the response
    });

    function validateFile(file) {
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        // Check file type
        if (file.type !== 'application/pdf') {
            showAlert('Please select a valid PDF file', 'error');
            return false;
        }

        // Check file size
        if (file.size > maxSize) {
            showAlert('File size must be less than 50MB', 'error');
            return false;
        }

        return true;
    }

    function showLoadingState() {
        // Disable form elements
        analyzeBtn.disabled = true;
        fileInput.disabled = true;
        
        // Update button text
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        
        // Show loading modal
        loadingModal.show();
    }

    function resetForm() {
        // Re-enable form elements
        analyzeBtn.disabled = false;
        fileInput.disabled = false;
        
        // Reset button text
        analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze PDF';
    }

    function showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'info'} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert alert at the top of the container
        const container = document.querySelector('.container .col-lg-8');
        const firstCard = container.querySelector('.card');
        container.insertBefore(alertDiv, firstCard);

        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // File drag and drop functionality
    const formCard = document.querySelector('.card');
    const fileInputArea = document.querySelector('.form-control');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        formCard.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        formCard.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        formCard.addEventListener(eventName, unhighlight, false);
    });

    formCard.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        formCard.classList.add('border-primary');
        formCard.style.borderWidth = '2px';
    }

    function unhighlight() {
        formCard.classList.remove('border-primary');
        formCard.style.borderWidth = '';
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            validateFile(files[0]);
        }
    }

    // File size display
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);
            const fileName = file.name;
            
            // Update form text to show file info
            const formText = document.querySelector('.form-text');
            formText.innerHTML = `
                <i class="fas fa-file-pdf text-danger me-1"></i>
                <strong>${fileName}</strong> (${fileSize} MB)
                <br>
                <i class="fas fa-info-circle me-1"></i>
                Maximum file size: 50MB | Maximum pages: 50 | Only PDF files are allowed
            `;
        }
    });
});
