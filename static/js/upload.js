document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('pdf_file');

    // Simple form submission with basic loading state
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        
        // Basic file validation
        if (!file) {
            e.preventDefault();
            alert('Please select a PDF file');
            return;
        }

        if (file.type !== 'application/pdf') {
            e.preventDefault();
            alert('Please select a valid PDF file');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            e.preventDefault();
            alert('File size must be less than 50MB');
            return;
        }

        // Show simple loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        
        // Let form submit normally
    });

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
