let newImage = false;

document.getElementById('image-upload').addEventListener('change', function() {
    const file = this.files[0];  // input element makes sure it's just one allowed
    if (file && file.type.match('image.*')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // display new image without actual upload
            const original_image = document.getElementById('original-image');
            original_image.src = e.target.result;
            original_image.alt = 'Uploaded image';
            hide('braces-preview');
            show('original-image');

            // update buttons
            disable('toggle-image-button');
            enable('braces-button');
            disable('download-button');
            newImage = true;

            // new input -> default config
            sessionStorage.clear()
            hide('controls-block');
        };
        reader.readAsDataURL(file);
    } else {
        alert("No image was found. Please make sure to select an image to display.")
    }
});

document.getElementById('braces-button').addEventListener('click', function () {
    const file = document.getElementById('image-upload').files[0];
    const formData = new FormData();

    if (newImage) {
        formData.append('image', file);
    } else {
        const imageURL = document.getElementById('original-image').src;
        const filename = imageURL.split('//').pop();
        formData.append('filename', filename);
    }
    sendImageData(formData)
});

function sendImageData(formData) {
    fetch('/api/process-image', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            newImage = false;

            // set preview image
            const preview = document.getElementById('braces-preview');
            preview.src = data.processedImageUrl + '?v=' + new Date().getTime();
            preview.alt = 'Processed image';
            hide('original-image');
            show('braces-preview');

            // update buttons
            enable('toggle-image-button')
            disable('braces-button')

            // update download link
            const download_link = document.getElementById('download-link');
            download_link.href = data.processedImageUrl;
            enable('download-button')

            // show used config values
            showUsedConfig(data.segmentationConfig.rows, 'row-num-control', 'row-num-value');
            showUsedConfig(data.segmentationConfig.bracket_size, 'bracket-size-control', 'bracket-size-value');
            showUsedConfig(data.segmentationConfig.mask_dilation, 'mask-dilation-control', 'mask-dilation-value');
            showUsedConfig(data.segmentationConfig.dist_transform_fraction, 'dist-transform-control', 'dist-transform-value');
            showUsedConfig(data.segmentationConfig.brighten_value, 'brighten-control', 'brighten-value');
            showUsedConfig(data.segmentationConfig.thresh_blocksize_fraction, 'thresh-blocksize-control', 'thresh-blocksize-value');
            showUsedConfig(data.segmentationConfig.closing_iterations, 'closing-iterations-control', 'closing-iterations-value');
            showUsedConfig(data.segmentationConfig.erosion_iterations, 'erosion-iterations-control', 'erosion-iterations-value');

            // and show the intermediate results / images
            document.getElementById('wire-plot-image').src = data.segmentationInterims.position_plot + '?v=' + new Date().getTime();
            document.getElementById('teeth-area-image').src = data.segmentationInterims.teeth_area + '?v=' + new Date().getTime();
            document.getElementById('brighten-image').src = data.segmentationInterims.brighten + '?v=' + new Date().getTime();
            document.getElementById('thresh-blocksize-image').src = data.segmentationInterims.adapt_thresh + '?v=' + new Date().getTime();
            document.getElementById('closing-iterations-image').src = data.segmentationInterims.closing + '?v=' + new Date().getTime();
            document.getElementById('dist-transform-image').src = data.segmentationInterims.dist_transform + '?v=' + new Date().getTime();
            document.getElementById('erosion-iterations-image').src = data.segmentationInterims.erosion + '?v=' + new Date().getTime();
            show('controls-block');

        } else {
            console.error('Failed to process image:', data.message);
        }
    })
    .catch(error => console.error('Error processing image:', error));
}

function showUsedConfig(value, inputElement, displayElement) {
    document.getElementById(inputElement).value = value;
    updateConfigValue(value, displayElement);
}

function updateConfigValue(value, element) {
    sessionStorage.setItem(element, value);
    if(element === 'dist-transform-value' || element === 'thresh-blocksize-value'){
        const percentage = (value * 100).toFixed(0);
        document.getElementById(element).textContent = percentage + '%';
    }
    else if(element === 'row-num-value' && value === 0) {
        document.getElementById(element).textContent = 'auto';
    }
    else {
        document.getElementById(element).textContent = value;
    }

}

// -----------------
// thanks to Anudeep Bulla from [this SO post](https://stackoverflow.com/a/19655662) for
// inspiration for the creation of event listeners on all config input elements:
let config_inputs = document.getElementsByClassName("config-input");

let timeoutID;
function debounceTrigger() {
    if(timeoutID) clearTimeout(timeoutID);
    // debounce the image processing computation since it's time-consuming!
    timeoutID = setTimeout(triggerImageProcessOnNewConfig, 500);
}

for (var i = 0; i < config_inputs.length; i++) {
    config_inputs[i].addEventListener('mouseup', debounceTrigger, false);
    config_inputs[i].addEventListener('input', function() {
        const displayElement = this.id.replace('control', 'value');
        updateConfigValue(this.value, displayElement);
    });
}

// ----------------------

function triggerImageProcessOnNewConfig() {
    const config = getConfiguration();
    const formData = new FormData;
    const imageURL = document.getElementById('braces-preview').src;
    const filename = imageURL.split('/').pop();
    formData.append('filename', filename);
    formData.append("config", JSON.stringify(config));
    sendImageData(formData);
}

function getConfiguration() {
    return {
        "rows": sessionStorage.getItem('row-num-value'),
        "bracket_size": sessionStorage.getItem('bracket-size-value'),
        "mask_dilation": sessionStorage.getItem('mask-dilation-value'),
        "dist_transform_fraction": sessionStorage.getItem('dist-transform-value'),
        "brighten_value": sessionStorage.getItem('brighten-value'),
        "thresh_blocksize_fraction": sessionStorage.getItem('thresh-blocksize-value'),
        "closing_iterations": sessionStorage.getItem('closing-iterations-value'),
        "erosion_iterations": sessionStorage.getItem('erosion-iterations-value')
    };
}

document.getElementById('toggle-image-button').addEventListener('mousedown', function () {
    hide('braces-preview');
    show('original-image');
});

document.getElementById('toggle-image-button').addEventListener('mouseup', function () {
    hide('original-image');
    show('braces-preview');
});

function show(element_id) {
    document.getElementById(element_id).style.display = 'initial';
}

function hide(element_id) {
    document.getElementById(element_id).style.display = 'none';
}

function disable(element_id) {
    const element = document.getElementById(element_id);
    element.disabled = true;
    element.style.cursor = 'auto';
}

function enable(element_id) {
    const element = document.getElementById(element_id);
    element.disabled = false;
    element.style.cursor = 'pointer';
}