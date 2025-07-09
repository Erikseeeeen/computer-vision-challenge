function [upsampled_img, stats] = Upsample_Crop(img, crop_rect, scale_factor, options)
    % Upsample a single image using gradient-preserving enhancement
    % 
    % Inputs:
    %   img - Input image (RGB or grayscale)
    %   crop_rect - [x, y, width, height] crop rectangle
    %   scale_factor - Upsampling factor (default: 4)
    %   options - Optional parameters struct
    %
    % Outputs:
    %   upsampled_img - Enhanced upsampled image
    %   stats - Dictionary of quality metrics
    
    if nargin < 3, scale_factor = 4; end
    if nargin < 4, options = struct(); end
    
    % Default options
    options = set_default_options(options);
    
    % Extract crop
    x = max(1, round(crop_rect(1)));
    y = max(1, round(crop_rect(2)));
    w = round(crop_rect(3));
    h = round(crop_rect(4));
    
    crop = img(y:y+h-1, x:x+w-1, :);
    
    % Apply gradient-preserving upsampling
    upsampled_img = gradient_preserving_upsample(crop, scale_factor, options);
    
    % Calculate statistics
    simple_upsampled = simple_upsample(crop, scale_factor);
    stats = calculate_image_stats(crop, upsampled_img, simple_upsampled, []);
end

function [upsampled_img, stats] = upsample_multi_images_basic(images, crop_rect, scale_factor, options)
    % Combine multiple images and upsample using basic interpolation
    %
    % Inputs:
    %   images - Cell array of images or 4D array (h x w x c x n)
    %   crop_rect - [x, y, width, height] crop rectangle
    %   scale_factor - Upsampling factor (default: 4)
    %   options - Optional parameters struct
    %
    % Outputs:
    %   upsampled_img - Basic upsampled composite
    %   stats - Dictionary of quality metrics and alignment info
    
    if nargin < 3, scale_factor = 4; end
    if nargin < 4, options = struct(); end
    
    % Default options
    options = set_default_options(options);
    
    % Convert to 4D array if needed
    if iscell(images)
        images = cell_to_4d_array(images);
    end
    
    % Extract crops and align
    [aligned_crops, alignment_quality] = align_multi_images(images, crop_rect, options);
    
    % Create weighted composite
    composite = create_weighted_composite(aligned_crops, alignment_quality, options);
    
    % Basic upsampling
    upsampled_img = simple_upsample(composite, scale_factor);
    
    % Calculate statistics
    reference_crop = aligned_crops(:,:,:,1);
    stats = calculate_image_stats(reference_crop, upsampled_img, upsampled_img, alignment_quality);
end

function [upsampled_img, stats] = upsample_multi_images_gradient(images, crop_rect, scale_factor, options)
    % Combine multiple images and upsample using gradient-preserving enhancement
    %
    % Inputs:
    %   images - Cell array of images or 4D array (h x w x c x n)
    %   crop_rect - [x, y, width, height] crop rectangle  
    %   scale_factor - Upsampling factor (default: 4)
    %   options - Optional parameters struct
    %
    % Outputs:
    %   upsampled_img - Gradient-preserving enhanced upsampled composite
    %   stats - Dictionary of quality metrics and alignment info
    
    if nargin < 3, scale_factor = 4; end
    if nargin < 4, options = struct(); end
    
    % Default options
    options = set_default_options(options);
    
    % Convert to 4D array if needed
    if iscell(images)
        images = cell_to_4d_array(images);
    end
    
    % Extract crops and align
    [aligned_crops, alignment_quality] = align_multi_images(images, crop_rect, options);
    
    % Create weighted composite
    composite = create_weighted_composite(aligned_crops, alignment_quality, options);
    
    % Gradient-preserving upsampling
    upsampled_img = gradient_preserving_upsample(composite, scale_factor, options);
    
    % Calculate statistics
    reference_crop = aligned_crops(:,:,:,1);
    simple_upsampled = simple_upsample(composite, scale_factor);
    stats = calculate_image_stats(reference_crop, upsampled_img, simple_upsampled, alignment_quality);
end

function stats = calculate_image_stats(original, enhanced, simple, alignment_quality)
    % Calculate comprehensive quality metrics
    
    stats = containers.Map();
    
    % Basic metrics
    stats('sharpness_original') = calculate_sharpness(original);
    stats('sharpness_enhanced') = calculate_sharpness(enhanced);
    stats('gradient_mag_original') = calculate_gradient_magnitude(original);
    stats('gradient_mag_enhanced') = calculate_gradient_magnitude(enhanced);
    stats('edge_density_original') = calculate_edge_density(original);
    stats('edge_density_enhanced') = calculate_edge_density(enhanced);
    
    % Improvement ratios
    if ~isempty(simple)
        stats('sharpness_simple') = calculate_sharpness(simple);
        stats('gradient_mag_simple') = calculate_gradient_magnitude(simple);
        stats('edge_density_simple') = calculate_edge_density(simple);
        
        stats('sharpness_improvement') = stats('sharpness_enhanced') / stats('sharpness_simple');
        stats('gradient_improvement') = stats('gradient_mag_enhanced') / stats('gradient_mag_simple');
        stats('edge_improvement') = stats('edge_density_enhanced') / stats('edge_density_simple');
        
        stats('overall_improvement') = mean([stats('sharpness_improvement'), ...
                                           stats('gradient_improvement'), ...
                                           stats('edge_improvement')]);
    end
    
    % Alignment statistics
    if ~isempty(alignment_quality)
        stats('num_images') = length(alignment_quality);
        stats('good_alignments') = sum(alignment_quality >= 0.7);
        stats('mean_alignment_quality') = mean(alignment_quality);
        stats('alignment_std') = std(alignment_quality);
    end
end

%% Helper Functions

function options = set_default_options(options)
    % Set default parameters
    if ~isfield(options, 'min_correlation'), options.min_correlation = 0.7; end
    if ~isfield(options, 'search_margin'), options.search_margin = 50; end
    if ~isfield(options, 'gradient_weight'), options.gradient_weight = 0.4; end
    if ~isfield(options, 'adaptive_sigma'), options.adaptive_sigma = 1.2; end
    if ~isfield(options, 'structure_tensor_sigma'), options.structure_tensor_sigma = 1.0; end
    if ~isfield(options, 'frequency_boost'), options.frequency_boost = 0.3; end
    if ~isfield(options, 'multi_scale_levels'), options.multi_scale_levels = 5; end
    if ~isfield(options, 'use_gradient_preservation'), options.use_gradient_preservation = true; end
    if ~isfield(options, 'use_adaptive_unsharp'), options.use_adaptive_unsharp = true; end
    if ~isfield(options, 'use_structure_tensor'), options.use_structure_tensor = true; end
    if ~isfield(options, 'use_frequency_enhancement'), options.use_frequency_enhancement = true; end
    if ~isfield(options, 'use_multi_scale'), options.use_multi_scale = true; end
end

function images_4d = cell_to_4d_array(images_cell)
    % Convert cell array to 4D array
    num_images = length(images_cell);
    first_img = images_cell{1};
    [h, w, c] = size(first_img);
    
    images_4d = zeros(h, w, c, num_images);
    for i = 1:num_images
        images_4d(:,:,:,i) = im2double(images_cell{i});
    end
end

function [aligned_crops, alignment_quality] = align_multi_images(images, crop_rect, options)
    % Align multiple images to extract consistent crops
    
    [h, w, c, num_images] = size(images);
    
    x = max(1, round(crop_rect(1)));
    y = max(1, round(crop_rect(2)));
    crop_w = round(crop_rect(3));
    crop_h = round(crop_rect(4));
    
    % Extract template from first image
    template = images(y:y+crop_h-1, x:x+crop_w-1, :, 1);
    template_gray = rgb2gray(template);
    
    % Initialize outputs
    aligned_crops = zeros(crop_h, crop_w, c, num_images);
    alignment_quality = zeros(num_images, 1);
    
    % Reference image
    aligned_crops(:,:,:,1) = template;
    alignment_quality(1) = 1.0;
    
    % Align remaining images
    for i = 2:num_images
        current_img = images(:,:,:,i);
        [best_crop, best_corr] = align_single_image(template_gray, current_img, ...
                                                   crop_rect, options);
        
        if best_corr > 0.2
            aligned_crops(:,:,:,i) = best_crop;
            alignment_quality(i) = best_corr;
        else
            aligned_crops(:,:,:,i) = template;  % Fallback
            alignment_quality(i) = 0.2;
        end
    end
end

function [best_crop, best_corr] = align_single_image(template_gray, current_img, crop_rect, options)
    % Align single image using multiple methods
    
    x = crop_rect(1); y = crop_rect(2);
    crop_w = crop_rect(3); crop_h = crop_rect(4);
    [h, w, c] = size(current_img);
    
    current_gray = rgb2gray(current_img);
    
    % Search bounds
    margin = options.search_margin;
    search_x1 = max(1, x - margin);
    search_y1 = max(1, y - margin);
    search_x2 = min(w, x + crop_w + margin);
    search_y2 = min(h, y + crop_h + margin);
    
    search_region = current_gray(search_y1:search_y2, search_x1:search_x2);
    
    best_crop = [];
    best_corr = 0;
    
    % Method 1: Template matching
    try
        [crop1, corr1] = template_matching_align(template_gray, search_region, ...
                                               current_img, [search_x1, search_y1], ...
                                               crop_w, crop_h, c);
        if corr1 > best_corr
            best_crop = crop1;
            best_corr = corr1;
        end
    catch
    end
    
    % Method 2: Feature matching
    try
        [crop2, corr2] = feature_matching_align(template_gray, current_gray, ...
                                              current_img, crop_w, crop_h, c);
        if corr2 > best_corr
            best_crop = crop2;
            best_corr = corr2;
        end
    catch
    end
    
    % Fallback
    if isempty(best_crop)
        try
            best_crop = current_img(y:y+crop_h-1, x:x+crop_w-1, :);
            best_corr = 0.2;
        catch
            best_crop = zeros(crop_h, crop_w, c);
            best_corr = 0.1;
        end
    end
end

function [crop, correlation] = template_matching_align(template, search_region, full_img, search_offset, w, h, c)
    % Enhanced template matching with preprocessing
    
    correlation = 0;
    crop = zeros(h, w, c);
    
    % Try different preprocessing methods
    methods = {'none', 'histeq'};
    
    for m = 1:length(methods)
        switch methods{m}
            case 'histeq'
                proc_template = histeq(template);
                proc_search = histeq(search_region);
            otherwise
                proc_template = template;
                proc_search = search_region;
        end
        
        % Template matching
        corr_map = normxcorr2(proc_template, proc_search);
        [max_corr, max_idx] = max(corr_map(:));
        
        if max_corr > correlation
            [peak_y, peak_x] = ind2sub(size(corr_map), max_idx);
            match_x = search_offset(1) + peak_x - size(proc_template, 2);
            match_y = search_offset(2) + peak_y - size(proc_template, 1);
            
            % Extract crop
            x1 = max(1, min(size(full_img, 2) - w + 1, match_x));
            y1 = max(1, min(size(full_img, 1) - h + 1, match_y));
            
            if x1 + w - 1 <= size(full_img, 2) && y1 + h - 1 <= size(full_img, 1)
                test_crop = full_img(y1:y1+h-1, x1:x1+w-1, :);
                if isequal(size(test_crop), [h, w, c])
                    crop = test_crop;
                    correlation = max_corr;
                end
            end
        end
    end
end

function [crop, correlation] = feature_matching_align(template, image, full_img, w, h, c)
    % Feature-based alignment
    
    crop = zeros(h, w, c);
    correlation = 0;
    
    try
        % Detect features
        pts1 = detectHarrisFeatures(template, 'MinQuality', 0.01);
        pts2 = detectHarrisFeatures(image, 'MinQuality', 0.01);
        
        if pts1.Count < 10 || pts2.Count < 10, return; end
        
        % Extract and match features
        [feat1, valid1] = extractFeatures(template, pts1);
        [feat2, valid2] = extractFeatures(image, pts2);
        
        pairs = matchFeatures(feat1, feat2, 'MatchThreshold', 10.0);
        if size(pairs, 1) < 6, return; end
        
        matched1 = valid1(pairs(:, 1));
        matched2 = valid2(pairs(:, 2));
        
        % Estimate transform
        [tform, ~] = estimateGeometricTransform(matched2, matched1, 'affine', ...
                                               'MaxDistance', 4.0);
        
        % Apply transform
        output_view = imref2d([h, w]);
        for ch = 1:c
            crop(:, :, ch) = imwarp(full_img(:, :, ch), tform, ...
                                   'OutputView', output_view, 'FillValues', 0);
        end
        
        correlation = corr2(template, rgb2gray(crop));
        
    catch
        return;
    end
end

function composite = create_weighted_composite(aligned_crops, alignment_quality, options)
    % Create weighted composite from aligned crops
    
    good_matches = alignment_quality >= options.min_correlation;
    if sum(good_matches) < 2
        good_matches = alignment_quality >= 0.4;
    end
    
    weights = alignment_quality(good_matches);
    weights = weights / sum(weights);
    
    composite = zeros(size(aligned_crops(:,:,:,1)));
    good_indices = find(good_matches);
    
    for i = 1:length(good_indices)
        idx = good_indices(i);
        composite = composite + weights(i) * aligned_crops(:,:,:,idx);
    end
end

function upsampled = simple_upsample(img, scale_factor)
    % Simple cubic interpolation upsampling
    
    [h, w, c] = size(img);
    [X, Y] = meshgrid(1:w, 1:h);
    [X_hr, Y_hr] = meshgrid(linspace(1, w, w * scale_factor), ...
                            linspace(1, h, h * scale_factor));
    
    upsampled = zeros(h * scale_factor, w * scale_factor, c);
    for ch = 1:c
        upsampled(:, :, ch) = interp2(X, Y, img(:, :, ch), X_hr, Y_hr, 'cubic');
    end
end

function upsampled = gradient_preserving_upsample(img, scale_factor, options)
    % Advanced gradient-preserving upsampling
    
    [h, w, c] = size(img);
    hr_h = h * scale_factor;
    hr_w = w * scale_factor;
    
    [X, Y] = meshgrid(1:w, 1:h);
    [X_hr, Y_hr] = meshgrid(linspace(1, w, hr_w), linspace(1, h, hr_h));
    
    upsampled = zeros(hr_h, hr_w, c);
    
    for ch = 1:c
        % Basic interpolation
        basic = interp2(X, Y, img(:, :, ch), X_hr, Y_hr, 'cubic');
        result = basic;
        
        % Gradient preservation
        if options.use_gradient_preservation
            result = apply_gradient_preservation(img(:, :, ch), result, ...
                                               X, Y, X_hr, Y_hr, options.gradient_weight);
        end
        
        % Adaptive unsharp masking
        if options.use_adaptive_unsharp
            result = apply_adaptive_unsharp(result, options.adaptive_sigma);
        end
        
        % Structure tensor enhancement
        if options.use_structure_tensor
            result = apply_structure_tensor_enhancement(result, options.structure_tensor_sigma);
        end
        
        % Frequency enhancement
        if options.use_frequency_enhancement
            result = apply_frequency_enhancement(result, options.frequency_boost);
        end
        
        % Multi-scale enhancement
        if options.use_multi_scale
            result = apply_multiscale_enhancement(result, options.multi_scale_levels);
        end
        
        upsampled(:, :, ch) = max(0, min(1, result));
    end
end

function result = apply_gradient_preservation(lr_img, hr_img, X, Y, X_hr, Y_hr, weight)
% Computes the gradient corrections, and applies them to enhance the high-resolution image
    
    % Compute the gradients of the low-resolution image in the x and y directions
    [gx_lr, gy_lr] = imgradientxy(lr_img);

    % Interpolate the low-resolution gradients to the high-resolution grid using cubic interpolation
    gx_hr = interp2(X, Y, gx_lr, X_hr, Y_hr, 'cubic', 0);
    gy_hr = interp2(X, Y, gy_lr, X_hr, Y_hr, 'cubic', 0);

    % Compute the gradients of the current high-resolution image
    [gx_current, gy_current] = imgradientxy(hr_img);

     % Calculate the gradient corrections for both x and y directions    
    grad_correction_x = weight * (gx_hr - gx_current);
    grad_correction_y = weight * (gy_hr - gy_current);

    % Apply convolution to smooth the gradient corrections
    correction = conv2(grad_correction_x, [-1 1], 'same') + ...
                conv2(grad_correction_y, [-1; 1], 'same');
    
    % Enhance the high-resolution image by adding a fraction of the correction
    result = hr_img + 0.3 * correction;
end

function result = apply_adaptive_unsharp(img, sigma)
% Applies adaptive unsharp masking to enhance image sharpness.

    % Calculate the local variance of the image using a 5x5 neighborhood
    local_var = stdfilt(img, ones(5,5));

    % Normalize the local variance to the range [0, 1]
    local_var = local_var / max(local_var(:));

    % Apply Gaussian blur to the image using the specified sigma
    blurred = imgaussfilt(img, sigma);

    % Calculate the unsharp strength based on local variance
    unsharp_strength = 0.5 + 1.0 * local_var;

    % Apply the unsharp mask by adding the weighted difference between the original
    % and blurred images to the original image
    result = img + unsharp_strength .* (img - blurred);
end

function result = apply_structure_tensor_enhancement(img, sigma)
% Enhances an image using structure tensor analysis.

    % Compute the gradients of the image in the x and y directions
    [gx, gy] = imgradientxy(img);

    % Calculate the components of the structure tensor
    J11 = imgaussfilt(gx.^2, sigma);
    J12 = imgaussfilt(gx.*gy, sigma);
    J22 = imgaussfilt(gy.^2, sigma);

    % Compute the trace and determinant of the structure tensor 
    trace_j = J11 + J22;
    det_j = J11.*J22 - J12.^2;
    coherence = (trace_j - 2*sqrt(det_j + eps)) ./ (trace_j + eps);

    % Calculate the coherence measure
    enhancement_strength = 0.3 * coherence;
    
    % Attempt to enhance the image using imsharpen
    try
        % Fallback to convolution with a sharpening kernel if imsharpen fails
        enhanced = imsharpen(img, 'Amount', 1.0);
        result = img + enhancement_strength .* (enhanced - img);
    catch
        sharp_kernel = [0 -1 0; -1 5 -1; 0 -1 0];
        enhanced = conv2(img, sharp_kernel, 'same');
        result = img + enhancement_strength .* (enhanced - img);
    end
end

function result = apply_frequency_enhancement(img, boost_factor)
% Enhances an image in the frequency domain
    % Compute the 2D Fourier transform of the image
    F = fft2(img);

    % Shift the zero-frequency component to the center of the spectrum
    F_shifted = fftshift(F);

    % Get the dimensions of the image
    [h, w] = size(img);

    % Create a meshgrid for frequency coordinates
    [u, v] = meshgrid(-w/2:w/2-1, -h/2:h/2-1);

    % Calculate the distance from the origin in the frequency domain
    D = sqrt(u.^2 + v.^2);

    % Create a boost filter based on the distance
    H = 1 + boost_factor * (D / max(D(:)));

    % Apply the boost filter to the shifted Fourier transform
    F_enhanced = F_shifted .* H;

    % Shift back the enhanced Fourier transform
    F_enhanced = ifftshift(F_enhanced);

    % Compute the inverse Fourier transform to get the enhanced image
    result = real(ifft2(F_enhanced));
end

function result = apply_multiscale_enhancement(img, num_scales)
% Enhances an image using multiscale detail extraction
    result = img;  % Initialize the result with the original image
    for scale = 1:num_scales
        sigma = 2^(scale-1);  % Calculate the standard deviation for Gaussian blur
        blurred = imgaussfilt(result, sigma);  % Apply Gaussian blur
        detail = result - blurred;  % Extract details by subtracting blurred image from original
        detail_strength = 1.0 / scale;  % Determine the strength of the detail to add back
        result = result + detail_strength * detail;  % Combine details back into the result
    end
end

function sharpness = calculate_sharpness(img)
% Computes the sharpness of an image.
    gray = rgb2gray(img);
    sharpness = mean(imgradient(gray), 'all');
end

function grad_mag = calculate_gradient_magnitude(img)
% Computes the gradient magnitude of an image.
    gray = rgb2gray(img);
    [gx, gy] = imgradientxy(gray);
    grad_mag = mean(sqrt(gx.^2 + gy.^2), 'all');
end

function edge_density = calculate_edge_density(img)
% Computes the edge density of an image.
    gray = rgb2gray(img);
    edges = edge(gray, 'Canny');
    edge_density = sum(edges(:)) / numel(edges);
end