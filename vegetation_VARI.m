function analysis_output = vegetation_VARI(path)
    img = imread(path);
    red = double(img(:,:,1));
    green = double(img(:,:,2));
    blue = double(img(:,:,3));
    
    vari = (green - red) ./ (green + red - blue);
    variClamped    = max(min(vari, 1), -1);
    variGray       = rescale(variClamped, 0, 1);

    cmap           = jet(256);
    index          = gray2ind(variGray, 256);
    variRGB        = ind2rgb(index, cmap);

    analysis_output.overlay = variRGB;
    analysis_output.name = 'vegetation_VARI';
end