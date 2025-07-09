function analysis_output = vegetation_VARI(path)
    % vegetation_VARI computes the Vegetation Index (VARI) from an image.
    %
    % This function reads an image from the specified path, calculates the 
    % Vegetation Index (VARI) using the red, green, and blue channels, 
    % and generates a color overlay based on the VARI values.
    %
    % Inputs:
    %   path - A string specifying the file path of the image to be processed.
    %
    % Outputs:
    %   analysis_output - A structure containing:
    %       overlay - The VARI color overlay image.
    %       name - The name of the analysis performed.
    %
    % Example:
    %   output = vegetation_VARI('path/to/image.jpg');

    % Read the image from the specified path
    img = imread(path);

    % Convert the RGB channels to double for calculation
    red = double(img(:,:,1));
    green = double(img(:,:,2));
    blue = double(img(:,:,3));

    % Calculate the Vegetation Index (VARI)
    vari = (green - red) ./ (green + red - blue);

    % Clamp the VARI values to the range [-1, 1]
    variClamped = max(min(vari, 1), -1);

    % Rescale the clamped VARI values to the range [0, 1]
    variGray = rescale(variClamped, 0, 1);

    % Create a colormap for visualization
    cmap = jet(256);

    % Convert the grayscale VARI image to an indexed image
    index = gray2ind(variGray, 256);

    % Map the indexed image to RGB using the colormap
    variRGB = ind2rgb(index, cmap);

    % Store the results in the output structure
    analysis_output.overlay = variRGB;
    analysis_output.name = 'vegetation_VARI';
end