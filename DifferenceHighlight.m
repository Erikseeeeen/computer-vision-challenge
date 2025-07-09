function results = DifferenceHighlight(A, B) 
    % DIFFERENCEHIGHLIGHT Highlights differences between two images and counts differing pixels.
    % 
    % RESULTS = DIFFERENCEHIGHLIGHT(A, B) compares two images A and B
    % and creates an overlay image that highlights the differences.
    % The function uses a tolerance level to determine significant differences.
    %
    % Inputs:
    %   A - First image (3D array).
    %   B - Second image (3D array).
    %
    % Outputs:
    %   results - A structure containing:
    %       overlayImage - Binary mask indicating differences.
    %       overlayAlpha - Alpha channel for overlay visualization.
    %       numDifferingPixels - Number of pixels with detected differences.

    differenceTolerance = 10; % Set the tolerance for difference detection
    A = double(A); % Convert image A to double precision
    B = double(B); % Convert image B to double precision

    % Create a mask where the maximum absolute difference exceeds the tolerance
    differenceMask = max(abs(A - B), [], 3) > differenceTolerance;

    % Count the number of differing pixels
    numDifferingPixels = sum(differenceMask(:)); % Sum all true values in the mask

    % Initialize the overlay image with the same size as A
    overlay = zeros(size(A), 'like', A);

    % Set the red channel of the overlay to indicate differences
    overlay(:,:,1) = differenceMask;

    % Prepare the results structure
    results = struct();
    results.overlayImage = overlay; % Store the overlay image
    results.overlayAlpha = uint8(differenceMask) * 255; % Create alpha channel for overlay
    results.numDifferingPixels = numDifferingPixels; % Store the count of differing pixels

end