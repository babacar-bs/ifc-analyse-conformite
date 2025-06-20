document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const fileInput = document.querySelector('input[type="file"]');
    
    form.addEventListener('submit', function(e) {
        // Validation basique du fichier
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Veuillez sélectionner un fichier');
        }
    });

    // Optionnel : prévisualisation du fichier
    fileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0].name;
        const fileNameDisplay = document.getElementById('file-name');
        if (fileNameDisplay) {
            fileNameDisplay.textContent = `Fichier sélectionné : ${fileName}`;
        }
    });
});