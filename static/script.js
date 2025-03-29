$(document).ready(function() {
    const $btn = $('.btn');
    const $start = $('.btn-start');
    const $home = $('.btn-back');

    $btn.hover(
        function() {
            var svg = $(this).find('svg path');
            $(this).data('timeout', setTimeout(function() {
                svg.css('fill', '#ffffff'); 
            }, 100)); 
        }, 
        function() {
            clearTimeout($(this).data('timeout')); 
            $(this).find('svg path').css('fill', '#59a9d4');
        }
    );

    $start.on('click', function() {
        window.location.href = '/action'; 
    });

    $home.on('click', function() {
        window.location.href = '/opening'; 
    });
    
});

$(document).ready(function() {
    const $fileInput = $('#file-image-upload');
    const $imageUploadInput = $('#image-upload');
    const $predictButton = $('.btn-predict-image');
    const $modelSelect = $('#deep-learning-model');
    const $llmSelect = $('#llm-model');
    const $loading = $('#load');
    const $transparant = $('#transparant-bg');

    let uploadedImagePath = '';

    $fileInput.on('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            $.ajax({
                url: '/upload',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(data) {
                    if (data.error) {
                        swal({
                            title: "Error",
                            text: data.error,
                            icon: "error",
                            customClass: {
                                popup: 'custom-swal-popup',
                                title: 'custom-swal-title',
                                content: 'custom-swal-content',
                                confirmButton: 'custom-swal-button'
                            },
                        });
                    } else {
                        uploadedImagePath = data.img_path;
                        $imageUploadInput.val(file.name);
                    }
                },
                error: function(xhr, status, error) {
                    console.error('Error uploading image:', error);
                    swal({
                        title: "Error",
                        text: "Terjadi kesalahan saat mengunggah gambar.",
                        icon: "error",
                        customClass: {
                            popup: 'custom-swal-popup',
                            title: 'custom-swal-title',
                            content: 'custom-swal-content',
                            confirmButton: 'custom-swal-button'
                        },
                    });
                }
            });
        }
    });

    $predictButton.on('click', function() {
        if (!uploadedImagePath) {
            swal({
                text: "Silakan upload gambar terlebih dahulu.",
                icon: "error",
                customClass: {
                    popup: 'custom-swal-popup',
                    title: 'custom-swal-title',
                    content: 'custom-swal-content',
                    confirmButton: 'custom-swal-button'
                }
            });            
            return;
        }

        const modelOption = $modelSelect.val();
        const llmOption = $llmSelect.val();

        $loading.show();
        $transparant.show();

        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                img_path: uploadedImagePath,
                model_option: modelOption,
                llm_option: llmOption
            }),
            success: function(data) {
                if (data.error) {
                    swal({
                        title: "Error",
                        text: data.error,
                        icon: "error",
                        customClass: {
                            popup: 'custom-swal-popup',
                            title: 'custom-swal-title',
                            content: 'custom-swal-content',
                            confirmButton: 'custom-swal-button'
                        },
                    });
                } else {
                    $loading.hide();
                    $transparant.hide();
                    
                    window.location.href = `/result?img_path=${data.img_path}&actual_class=${data.actual_class}&probability_class=${data.probability_class}&response=${data.response}`;
                }
            },
            error: function(xhr, status, error) {
                $loading.hide();
                $transparantBg.hide();
                
                console.error('Error predicting image:', error);
                swal({
                    title: "Error",
                    text: "Terjadi kesalahan saat memprediksi gambar.",
                    icon: "error",
                    customClass: {
                        popup: 'custom-swal-popup',
                        title: 'custom-swal-title',
                        content: 'custom-swal-content',
                        confirmButton: 'custom-swal-button'
                    },
                });
            }
        });
    });
});