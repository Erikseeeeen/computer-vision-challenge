function results = DifferenceHighlight(A, B) 
    differenceTolerance = 10;
    A = double(A);
    B = double(B);

    differenceMask = max(abs(A - B),[],3) > differenceTolerance;
    overlay = zeros(size(A), 'like', A);
    
    overlay(:,:,1) = differenceMask;

    results = struct();
    results.overlayImage = overlay;
    results.overlayAlpha = uint8(differenceMask) * 255;
end