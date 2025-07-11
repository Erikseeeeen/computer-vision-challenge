function reflectiveResults = visualizeReflectiveSurfaces(imagePath, varargin)
% visualizeReflectiveSurfaces  Detect, visualize and display reflective regions
%
%   reflectiveResults = visualizeReflectiveSurfaces(imagePath)
%   uses Method='Ensemble' (alias for 'Luminosity') and Sensitivity=90.
%
%   reflectiveResults = visualizeReflectiveSurfaces(...,'Method',M)
%     M ∈ {'Ensemble','Luminosity','TopPercent'}
%
%   reflectiveResults = visualizeReflectiveSurfaces(...,'Sensitivity',P)
%     P is percentile threshold for TopPercent mode.

    %% Parse inputs
    p = inputParser;
    addRequired(  p, 'imagePath',    @(x) ischar(x) || isstring(x));
    addParameter(p, 'Method',       'Ensemble',      @(x) ischar(x)||isstring(x));
    addParameter(p, 'Sensitivity',  90,              @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=100);
    parse(p,imagePath,varargin{:});

    rawMethod = validatestring(upper(string(p.Results.Method)), ...
                 ["ENSEMBLE","LUMINOSITY","TOPPERCENT"]);
    thrPerc   = p.Results.Sensitivity;

    % normalize alias
    if rawMethod=="ENSEMBLE"
        method = "LUMINOSITY";
    else
        method = rawMethod;
    end

    %% Load image & compute reflectivity index
    I = im2double(imread(imagePath));
    [h,w,~] = size(I);
    coeff = [0.21,0.72,0.07];
    if size(I,3)<3
        gray = mean(I,3);
    else
        gray = coeff(1)*I(:,:,1) + coeff(2)*I(:,:,2) + coeff(3)*I(:,:,3);
    end

    mn = min(gray(:)); mx = max(gray(:));
    if mx>mn
        riMap = (gray - mn)/(mx - mn);
    else
        riMap = zeros(h,w);
    end

    %% Binary mask (TopPercent cleanup)
    tval = prctile(riMap(:), thrPerc);
    mask = riMap >= tval;
    mask = bwareaopen(mask,50);
    mask = imclose(mask, strel('disk',5));

    %% Build both overlays

    % TopPercent overlay (pure blue mask + cyan blend)
    alphaTP     = uint8(mask * 255);
    overlayTP   = zeros(h,w,3,'uint8');
    overlayTP(:,:,3) = alphaTP;
    overlayImgTP = imoverlay(I, mask, 'cyan');

    %% Pack common results
    reflectiveResults = struct();
    reflectiveResults.originalImage     = I;
    reflectiveResults.reflectivityIndex = riMap;
    reflectiveResults.finalMask         = mask;
    reflectiveResults.selectedMethod    = method;
    reflectiveResults.sensitivity       = thrPerc;

    reflectiveResults.topPercent.alphaMap     = alphaTP;
    reflectiveResults.topPercent.overlay      = overlayTP;
    reflectiveResults.topPercent.overlayImage = overlayImgTP;

    %% Luminosity branch: use colormap instead of gray
    switch method
      case "LUMINOSITY"
        % 1) choose a colormap
        cmap = hot(256);

        % 2) map riMap into 1..256 indices
        idx = min(max(floor(riMap*255),0),255) + 1;

        % 3) build RGB heatmap [0..1]
        rgbMap = ind2rgb(idx, cmap);

        % 4) alpha-blend heatmap onto original
        ri3 = repmat(riMap, [1,1,3]);
        blended = (1 - ri3).*I + ri3.*rgbMap;

        % 5) quantize to uint8
        alphaMap      = uint8(riMap * 255);
        overlayMap    = im2uint8(rgbMap);
        overlayPreview= im2uint8(blended);

        % store under root for backward compatibility
        reflectiveResults.overlayAlpha    = alphaMap;
        reflectiveResults.overlay         = overlayMap;
        reflectiveResults.overlayImage    = overlayPreview;

      case "TOPPERCENT"
        reflectiveResults.overlayAlpha    = alphaTP;
        reflectiveResults.overlay         = overlayTP;
        reflectiveResults.overlayImage    = overlayImgTP;
    end
end
