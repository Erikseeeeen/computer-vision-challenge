function areaDifference = CalculateAreaDifference(currentLocation1, currentLocation2)
    % CalculateAreaDifference computes the area of the difference between two images.
    %
    % Inputs:
    %   currentLocation1 - The subfolder name where the first image is stored.
    %   currentLocation2 - The subfolder name where the second image is stored.
    %
    % Outputs:
    %   areaDifference - The area of the difference in square meters.

    % Extract scale from the first image
    m_per_pixel1 = ExtractScale(currentLocation1);

    % Extract scale from the second image
    m_per_pixel2 = ExtractScale(currentLocation2);

    % Scales should be equal, check if everything's correct
    if m_per_pixel1 ~= m_per_pixel2
        error("Rescaling must have caused and error")
    else
        m_per_pixel = m_per_pixel1;
    end

    % Get the number of differing pixels from the two images
    results = DifferenceHighlight(currentLocation1, currentLocation2);
    numDifferingPixels = results.numDifferingPixels;

    % Calculate the area difference in square meters
    areaDifference = numDifferingPixels * (m_per_pixel^2);

end