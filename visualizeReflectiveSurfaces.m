function reflectiveResults = visualizeReflectiveSurfaces(imagePath, varargin)
% visualizeReflectiveSurfaces  Detect and visualize reflective regions
%
%   reflectiveResults = visualizeReflectiveSurfaces(imagePath)
%   processes the image at imagePath using the default Method = 'Ensemble'
%   (alias for 'Luminosity') and Sensitivity = 90 (top 10% brightest).
%
%   reflectiveResults = visualizeReflectiveSurfaces(..., 'Method', M)
%   specifies how to compute the reflectivity map:
%       'Ensemble'   – alias for 'Luminosity'
%       'Luminosity' – weighted sum 0.21 R + 0.72 G + 0.07 B
%       'TopPercent' – same luminosity map but returns only a binary
%                      mask of the top Sensitivity percentile
%
%   reflectiveResults = visualizeReflectiveSurfaces(..., 'Sensitivity', P)
%   sets the percentile threshold P (0–100) for 'TopPercent' mode.
%   Default is 90 (i.e. top 10% most reflective pixels).
%
% Outputs
%   reflectiveResults : struct containing
%       .originalImage      – H×W×3 double RGB input image
%       .reflectivityIndex  – H×W double map in [0..1]
%       .finalMask          – H×W logical mask of thresholded regions
%       .selectedMethod     – string, either 'LUMINOSITY' or 'TOPPERCENT'
%       .sensitivity        – numeric percentile threshold
%       .overlayAlpha       – H×W uint8 alpha map (0 or up to 255)
%       .overlay            – H×W×3 uint8 raw overlay image
%       .overlayImage       – H×W×3 uint8 final blended preview
%
% Example
%   R = visualizeReflectiveSurfaces('wall.jpg', ...
%       'Method','TopPercent','Sensitivity',95);
%   imshow(R.overlayImage);
%
% See also im2double, prctile, bwareaopen, imclose, strel, imoverlay

    %% Parse inputs
    p = inputParser;
    addRequired(   p, 'imagePath',   @(x) ischar(x) || isstring(x));
    addParameter(  p, 'Method',      'Ensemble', @(x) ischar(x) || isstring(x));
    addParameter(  p, 'Sensitivity', 90,         @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=100);
    parse(p, imagePath, varargin{:});

    rawMethod = validatestring(upper(string(p.Results.Method)), ...
                 ["ENSEMBLE","LUMINOSITY","TOPPERCENT"]);
    thrPerc   = p.Results.Sensitivity;

    % alias: Ensemble == Luminosity
    if rawMethod == "ENSEMBLE"
        method = "LUMINOSITY";
    else
        method = rawMethod;
    end

    %% Load and prep image
    I = im2double(imread(imagePath));
    [h,w,~] = size(I);

    %% Compute luminosity map
    coeff = [0.21, 0.72, 0.07];
    if size(I,3) < 3
        gray = mean(I,3);
    else
        gray = coeff(1)*I(:,:,1) + coeff(2)*I(:,:,2) + coeff(3)*I(:,:,3);
    end

    % normalize to [0,1]
    mn = min(gray(:)); mx = max(gray(:));
    if mx > mn
        riMap = (gray - mn) / (mx - mn);
    else
        riMap = zeros(h,w);
    end

    %% Build binary mask (always from riMap)
    tval = prctile(riMap(:), thrPerc);
    mask = riMap >= tval;
    mask = bwareaopen(mask,50);
    mask = imclose(mask, strel('disk',5));

    %% Build results struct
    reflectiveResults = struct();
    reflectiveResults.originalImage      = I;
    reflectiveResults.reflectivityIndex = riMap;
    reflectiveResults.finalMask          = mask;
    reflectiveResults.selectedMethod     = method;
    reflectiveResults.sensitivity        = thrPerc;

    switch method
      case "LUMINOSITY"
        % continuous gray overlay
        alphaMap     = uint8(riMap * 255);
        overlay      = repmat(alphaMap, [1,1,3]);
        overlayImage = overlay;

      case "TOPPERCENT"
        % binary mask overlay (cyan highlight)
        alphaMap     = uint8(mask * 255);
        overlay      = zeros(h,w,3,'uint8');
        overlay(:,:,3)= alphaMap;
        overlayImage = imoverlay(I, mask, 'cyan');
    end

    reflectiveResults.overlayAlpha = alphaMap;
    reflectiveResults.overlay      = overlay;
    reflectiveResults.overlayImage = overlayImage;
end
