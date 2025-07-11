function vegetationSegmentationResults = VegetationSegment(imagePath, varargin)
% VegetationSegment - Vegetation segmentation on an image with configurable method and sensitivity.
%
% Syntax: 
%   results = VegetationSegment(imagePath)
%   results = VegetationSegment(imagePath, 'Method', method, 'Sensitivity', sensitivity)
%
% Inputs:
%   imagePath - Path to the input image
%   Method - (optional) String specifying segmentation method:
%            'NDVI', 'EVI', 'SAVI', 'MSAVI', 'GNDVI', 'VARI', 'ExG', 'GLI', 
%            'NGRDI', 'TGI', or 'Ensemble' (default)
%   Sensitivity - (optional) Numeric value controlling detection sensitivity
%                Default values vary by method

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'imagePath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Method', 'Ensemble', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Sensitivity', [], @(x) isnumeric(x) && isscalar(x));
    
    parse(p, imagePath, varargin{:});
    
    method = upper(string(p.Results.Method));
    sensitivity = p.Results.Sensitivity;
    
    % Validate method
    validMethods = ["NDVI", "EVI", "SAVI", "MSAVI", "GNDVI", "VARI", "EXG", ...
                    "GLI", "NGRDI", "TGI", "ENSEMBLE"];
    if ~ismember(method, validMethods)
        error('VegetationSegment:InvalidMethod', ...
              'Method must be one of: %s', strjoin(validMethods, ', '));
    end
    
    % Set default sensitivity values for each method
    defaultSensitivity = containers.Map();
    defaultSensitivity('NDVI') = 0.3;      % Standard NDVI threshold for vegetation
    defaultSensitivity('EVI') = 0.2;       % EVI threshold
    defaultSensitivity('SAVI') = 0.2;      % SAVI threshold
    defaultSensitivity('MSAVI') = 0.2;     % MSAVI threshold
    defaultSensitivity('GNDVI') = 0.2;     % GNDVI threshold
    defaultSensitivity('VARI') = 0.0;      % VARI threshold
    defaultSensitivity('EXG') = 0.1;       % Excess Green threshold
    defaultSensitivity('GLI') = 0.0;       % Green Leaf Index threshold
    defaultSensitivity('NGRDI') = 0.0;     % NGRDI threshold
    defaultSensitivity('TGI') = 0.0;       % Triangular Greenness Index threshold
    defaultSensitivity('ENSEMBLE') = 0.5;  % Ensemble threshold
    
    if isempty(sensitivity)
        sensitivity = defaultSensitivity(char(method));
    end

    % Input Validation and Image Loading
    if ~exist(imagePath, 'file')
        error('VegetationSegment:FileNotFound', 'Image file not found at path: %s', imagePath);
    end
    img = imread(imagePath);
    img_double = im2double(img);
    [rows, cols, ~] = size(img_double);
    total_pixels = rows * cols;

    R = img_double(:,:,1);
    G = img_double(:,:,2);
    B = img_double(:,:,3);

    % Execute selected method or ensemble
    switch method
        case 'NDVI'
            [final_vegetation_mask, method_index] = calculateNDVI(R, G, B, sensitivity);
            confidence_map = (method_index + 1) / 2; % Normalize to [0,1]
            
        case 'EVI'
            [final_vegetation_mask, method_index] = calculateEVI(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'SAVI'
            [final_vegetation_mask, method_index] = calculateSAVI(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'MSAVI'
            [final_vegetation_mask, method_index] = calculateMSAVI(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'GNDVI'
            [final_vegetation_mask, method_index] = calculateGNDVI(R, G, sensitivity);
            confidence_map = (method_index + 1) / 2; % Normalize to [0,1]
            
        case 'VARI'
            [final_vegetation_mask, method_index] = calculateVARI(R, G, B, sensitivity);
            confidence_map = (method_index + 1) / 2; % Normalize to [0,1]
            
        case 'EXG'
            [final_vegetation_mask, method_index] = calculateExG(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'GLI'
            [final_vegetation_mask, method_index] = calculateGLI(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'NGRDI'
            [final_vegetation_mask, method_index] = calculateNGRDI(R, G, sensitivity);
            confidence_map = (method_index + 1) / 2; % Normalize to [0,1]
            
        case 'TGI'
            [final_vegetation_mask, method_index] = calculateTGI(R, G, B, sensitivity);
            confidence_map = rescale(method_index, 0, 1);
            
        case 'ENSEMBLE'
            % Calculate all methods with their default sensitivities for ensemble
            [veg_ndvi, ndvi_index] = calculateNDVI(R, G, B, defaultSensitivity('NDVI'));
            [veg_evi, evi_index] = calculateEVI(R, G, B, defaultSensitivity('EVI'));
            [veg_savi, savi_index] = calculateSAVI(R, G, B, defaultSensitivity('SAVI'));
            [veg_gndvi, gndvi_index] = calculateGNDVI(R, G, defaultSensitivity('GNDVI'));
            [veg_vari, vari_index] = calculateVARI(R, G, B, defaultSensitivity('VARI'));
            [veg_exg, exg_index] = calculateExG(R, G, B, defaultSensitivity('EXG'));

            % ENSEMBLE METHOD: Combine Multiple Approaches
            weights.ndvi = 0.25;
            weights.evi = 0.20;
            weights.savi = 0.15;
            weights.gndvi = 0.15;
            weights.vari = 0.15;
            weights.exg = 0.10;

            % Normalize all indices to [0,1] for ensemble scoring
            ndvi_norm = (ndvi_index + 1) / 2;
            evi_norm = rescale(evi_index, 0, 1);
            savi_norm = rescale(savi_index, 0, 1);
            gndvi_norm = (gndvi_index + 1) / 2;
            vari_norm = (vari_index + 1) / 2;
            exg_norm = rescale(exg_index, 0, 1);

            ensemble_score = weights.ndvi * ndvi_norm + ...
                             weights.evi * evi_norm + ...
                             weights.savi * savi_norm + ...
                             weights.gndvi * gndvi_norm + ...
                             weights.vari * vari_norm + ...
                             weights.exg * exg_norm;

            final_vegetation_mask = ensemble_score > sensitivity;
            confidence_map = ensemble_score;
            method_index = ensemble_score;
            
        otherwise
            error('VegetationSegment:UnexpectedMethod', 'Unexpected method: %s', method);
    end

    % POST-PROCESSING: Morphological Operations (only for single methods, not ensemble)
    if ~strcmp(method, 'ENSEMBLE')
        final_vegetation_mask = postProcessVegetationMask(final_vegetation_mask);
    end

    pixel_area = sum(final_vegetation_mask(:));
    vegetation_percentage = (pixel_area / total_pixels) * 100;

    % Spatial calibration
    scale_bar_pixels = 50;
    real_world_length = 10;
    conversion_factor = (real_world_length / scale_bar_pixels)^2;
    real_world_area = pixel_area * conversion_factor;

    % Calculate vegetation health metrics
    gray_img = rgb2gray(img_double);
    [Gx, Gy] = gradient(gray_img);
    gradient_mag = sqrt(Gx.^2 + Gy.^2);
    boundary_pixels = bwperim(final_vegetation_mask);
    avg_boundary_strength = mean(gradient_mag(boundary_pixels));
    if isempty(boundary_pixels) || isnan(avg_boundary_strength)
        avg_boundary_strength = 0;
    end

    % Vegetation region statistics
    veg_stats_props = regionprops(final_vegetation_mask, 'Area', 'Perimeter', 'Solidity', 'BoundingBox');
    num_vegetation_regions = length(veg_stats_props);

    % Calculate average vegetation index value in detected regions
    if sum(final_vegetation_mask(:)) > 0
        avg_vegetation_index = mean(method_index(final_vegetation_mask));
    else
        avg_vegetation_index = 0;
    end

    % Build results structure
    vegetationSegmentationResults = struct();
    vegetationSegmentationResults.originalImage = img;
    vegetationSegmentationResults.finalVegetationMask = final_vegetation_mask;

    greenChannel = uint8(final_vegetation_mask) * 255;
    vegetationSegmentationResults.overlayAlpha = greenChannel;
    
    overlay = zeros(size(img), 'uint8');
    overlay(:,:,2) = greenChannel; % Green channel
    vegetationSegmentationResults.overlay = overlay;
    vegetationSegmentationResults.overlayImage = overlay;
    vegetationSegmentationResults.confidenceMap = confidence_map;

    % Store method-specific results
    vegetationSegmentationResults.selectedMethod = char(method);
    vegetationSegmentationResults.sensitivity = sensitivity;
    vegetationSegmentationResults.methodIndexMap = method_index;

    % Store statistics
    stats = struct();
    stats.pixelArea = pixel_area;
    stats.totalPixels = total_pixels;
    stats.vegetationPercentage = vegetation_percentage;
    stats.realWorldArea = real_world_area;
    stats.avgBoundaryStrength = avg_boundary_strength;
    stats.numVegetationRegions = num_vegetation_regions;
    stats.vegetationRegionProps = veg_stats_props;
    stats.avgVegetationIndex = avg_vegetation_index;
    stats.selectedMethod = char(method);
    stats.sensitivity = sensitivity;

    vegetationSegmentationResults.stats = stats;
    
    % If ensemble was used, also store individual method results
    if strcmp(method, 'ENSEMBLE')
        vegetationSegmentationResults.ndviMask = veg_ndvi;
        vegetationSegmentationResults.ndviIndexMap = ndvi_index;
        vegetationSegmentationResults.eviMask = veg_evi;
        vegetationSegmentationResults.eviIndexMap = evi_index;
        vegetationSegmentationResults.saviMask = veg_savi;
        vegetationSegmentationResults.saviIndexMap = savi_index;
        vegetationSegmentationResults.gndviMask = veg_gndvi;
        vegetationSegmentationResults.gndviIndexMap = gndvi_index;
        vegetationSegmentationResults.variMask = veg_vari;
        vegetationSegmentationResults.variIndexMap = vari_index;
        vegetationSegmentationResults.exgMask = veg_exg;
        vegetationSegmentationResults.exgIndexMap = exg_index;
        vegetationSegmentationResults.ensembleScoreMap = ensemble_score;
        
        % Add individual method statistics
        stats.ndviVegetationPercentage = (sum(veg_ndvi(:))/total_pixels)*100;
        stats.eviVegetationPercentage = (sum(veg_evi(:))/total_pixels)*100;
        stats.saviVegetationPercentage = (sum(veg_savi(:))/total_pixels)*100;
        stats.gndviVegetationPercentage = (sum(veg_gndvi(:))/total_pixels)*100;
        stats.variVegetationPercentage = (sum(veg_vari(:))/total_pixels)*100;
        stats.exgVegetationPercentage = (sum(veg_exg(:))/total_pixels)*100;
        vegetationSegmentationResults.stats = stats;
    end
end

% ========================================================================
% VEGETATION INDEX CALCULATION FUNCTIONS
% ========================================================================

function [vegetation_mask, ndvi] = calculateNDVI(red_channel, green_channel, blue_channel, sensitivity)
% calculateNDVI - Calculates the Normalized Difference Vegetation Index (NDVI)
%   NDVI = (NIR - Red) / (NIR + Red)
%   Since we don't have NIR, we approximate using Green channel
    if nargin < 4
        sensitivity = 0.3;  % Default threshold
    end
    
    denominator = green_channel + red_channel;
    valid_mask = denominator > 0.01;
    ndvi = zeros(size(green_channel), 'double');
    ndvi(valid_mask) = (green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask);
    vegetation_mask = ndvi > sensitivity;
end

function [vegetation_mask, evi] = calculateEVI(red_channel, green_channel, blue_channel, sensitivity)
% calculateEVI - Calculates the Enhanced Vegetation Index (EVI)
%   EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
%   Approximated using Green as NIR
    if nargin < 4
        sensitivity = 0.2;  % Default threshold
    end
    
    denominator = green_channel + 6*red_channel - 7.5*blue_channel + 1;
    valid_mask = abs(denominator) > 0.01;
    evi = zeros(size(green_channel), 'double');
    evi(valid_mask) = 2.5 * ((green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask));
    vegetation_mask = evi > sensitivity;
end

function [vegetation_mask, savi] = calculateSAVI(red_channel, green_channel, blue_channel, sensitivity)
% calculateSAVI - Calculates the Soil Adjusted Vegetation Index (SAVI)
%   SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
%   where L is soil brightness correction factor (typically 0.5)
    if nargin < 4
        sensitivity = 0.2;  % Default threshold
    end
    
    L = 0.5;  % Soil brightness correction factor
    denominator = green_channel + red_channel + L;
    valid_mask = denominator > 0.01;
    savi = zeros(size(green_channel), 'double');
    savi(valid_mask) = ((green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask)) * (1 + L);
    vegetation_mask = savi > sensitivity;
end

function [vegetation_mask, msavi] = calculateMSAVI(red_channel, green_channel, blue_channel, sensitivity)
% calculateMSAVI - Calculates the Modified Soil Adjusted Vegetation Index (MSAVI)
%   MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
    if nargin < 4
        sensitivity = 0.2;  % Default threshold
    end
    
    nir = green_channel;  % Approximation
    term1 = 2*nir + 1;
    term2 = sqrt(max(0, term1.^2 - 8*(nir - red_channel)));
    msavi = (term1 - term2) / 2;
    vegetation_mask = msavi > sensitivity;
end

function [vegetation_mask, gndvi] = calculateGNDVI(red_channel, green_channel, sensitivity)
% calculateGNDVI - Calculates the Green Normalized Difference Vegetation Index (GNDVI)
%   GNDVI = (NIR - Green) / (NIR + Green)
%   We use Red as approximate NIR for visible spectrum
    if nargin < 3
        sensitivity = 0.2;  % Default threshold
    end
    
    denominator = red_channel + green_channel;
    valid_mask = denominator > 0.01;
    gndvi = zeros(size(green_channel), 'double');
    gndvi(valid_mask) = (red_channel(valid_mask) - green_channel(valid_mask)) ./ denominator(valid_mask);
    vegetation_mask = gndvi > sensitivity;
end

function [vegetation_mask, vari] = calculateVARI(red_channel, green_channel, blue_channel, sensitivity)
% calculateVARI - Calculates the Visible Atmospherically Resistant Index (VARI)
%   VARI = (Green - Red) / (Green + Red - Blue)
    if nargin < 4
        sensitivity = 0.0;  % Default threshold
    end
    
    denominator = green_channel + red_channel - blue_channel;
    valid_mask = abs(denominator) > 0.01;
    vari = zeros(size(green_channel), 'double');
    vari(valid_mask) = (green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask);
    vegetation_mask = vari > sensitivity;
end

function [vegetation_mask, exg] = calculateExG(red_channel, green_channel, blue_channel, sensitivity)
% calculateExG - Calculates the Excess Green Index (ExG)
%   ExG = 2*Green - Red - Blue
    if nargin < 4
        sensitivity = 0.1;  % Default threshold
    end
    
    exg = 2*green_channel - red_channel - blue_channel;
    vegetation_mask = exg > sensitivity;
end

function [vegetation_mask, gli] = calculateGLI(red_channel, green_channel, blue_channel, sensitivity)
% calculateGLI - Calculates the Green Leaf Index (GLI)
%   GLI = (2*Green - Red - Blue) / (2*Green + Red + Blue)
    if nargin < 4
        sensitivity = 0.0;  % Default threshold
    end
    
    numerator = 2*green_channel - red_channel - blue_channel;
    denominator = 2*green_channel + red_channel + blue_channel;
    valid_mask = denominator > 0.01;
    gli = zeros(size(green_channel), 'double');
    gli(valid_mask) = numerator(valid_mask) ./ denominator(valid_mask);
    vegetation_mask = gli > sensitivity;
end

function [vegetation_mask, ngrdi] = calculateNGRDI(red_channel, green_channel, sensitivity)
% calculateNGRDI - Calculates the Normalized Green Red Difference Index (NGRDI)
%   NGRDI = (Green - Red) / (Green + Red)
    if nargin < 3
        sensitivity = 0.0;  % Default threshold
    end
    
    denominator = green_channel + red_channel;
    valid_mask = denominator > 0.01;
    ngrdi = zeros(size(green_channel), 'double');
    ngrdi(valid_mask) = (green_channel(valid_mask) - red_channel(valid_mask)) ./ denominator(valid_mask);
    vegetation_mask = ngrdi > sensitivity;
end

function [vegetation_mask, tgi] = calculateTGI(red_channel, green_channel, blue_channel, sensitivity)
% calculateTGI - Calculates the Triangular Greenness Index (TGI)
%   TGI = Green - 0.39*Red - 0.61*Blue
    if nargin < 4
        sensitivity = 0.0;  % Default threshold
    end
    
    tgi = green_channel - 0.39*red_channel - 0.61*blue_channel;
    vegetation_mask = tgi > sensitivity;
end

function cleaned_mask = postProcessVegetationMask(input_mask)
% postProcessVegetationMask - Refines a binary vegetation mask using
%   morphological operations to remove noise, fill holes, and smooth boundaries.
    cleaned_mask = bwareaopen(input_mask, 100);  % Remove small objects
    cleaned_mask = imfill(cleaned_mask, 'holes');
    se_smooth = strel('disk', 1);
    cleaned_mask = imopen(cleaned_mask, se_smooth);
    cleaned_mask = imclose(cleaned_mask, se_smooth);
    cleaned_mask = bwareaopen(cleaned_mask, 250);  % Final cleanup
end