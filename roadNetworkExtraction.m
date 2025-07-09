function roadNetworkOverlay = roadNetworkExtraction(imagePath, numClusters, epsilon)
% ROADNETWORKEXTRACTION Extracts road networks from an image using k-means clustering and edge detection.
    %
    %   roadNetworkOverlay = ROADNETWORKEXTRACTION(imagePath, numClusters, epsilon) 
    %   reads an image from the specified path, applies k-means clustering to identify 
    %   road-like regions, and detects edges using the Canny method. The resulting 
    %   overlay of detected road networks is returned.
    %
    %   Inputs:
    %       imagePath   - String specifying the path to the input image.
    %       numClusters  - Integer specifying the number of clusters for k-means.
    %       epsilon      - Scalar specifying the threshold for Canny edge detection.
    %
    %   Outputs:
    %       roadNetworkOverlay - RGB image with detected road edges overlaid in red.
    %
    %   Example:
    %       overlay = roadNetworkExtraction('road_image.jpg', 5, 0.1);
    %       imshow(overlay); % Display the overlay image with detected roads.
    % Read input image and get dimensions
    img = imread(imagePath);
    [rows, cols, channels] = size(img);

    % Reshape for k-means: (rows*cols) x channels
    imgReshaped = reshape(double(img), rows*cols, channels);

    % Run k-means and get centroids
    [clusterIdx, centroids] = kmeans(imgReshaped, numClusters, ...
                                     'MaxIter',1000, 'Replicates',3);

    % Convert centroids to grayscale to identify "road" clusters
    grayC = 0.2989*centroids(:,1) + 0.5870*centroids(:,2) + 0.1140*centroids(:,3);
    grayC = grayC ./ 255;  

    % Define brightness range for road-like clusters
    t1 = 0.3;  
    t2 = 0.8;  
    roadCIDs = find(grayC >= t1 & grayC <= t2);

    % Build the road mask from cluster labels
    labelMap = reshape(clusterIdx, rows, cols);
    roadMask = ismember(labelMap, roadCIDs);

    % Clean mask: remove small specks, close gaps
    roadMask = bwareaopen(roadMask, 200);
    roadMask = imclose(roadMask, strel('disk',5));

    % Convert to grayscale for edge detection
    grayImg = rgb2gray(img);

    % Detect edges (Canny) and restrict to roadMask
    edgeImg = edge(grayImg, 'Canny', epsilon);
    edgeImg = edgeImg & roadMask;

    % Optional: clean tiny edge artifacts
    edgeImg = bwareaopen(edgeImg, 20);

    % Overlay edges in red on original image
    overlay = img;
    overlay(:,:,1) = max(overlay(:,:,1), uint8(edgeImg)*255);  
    overlay(:,:,2) = overlay(:,:,2) .* uint8(~edgeImg);        
    overlay(:,:,3) = overlay(:,:,3) .* uint8(~edgeImg);        

    % Display and return result
    figure; imshow(overlay);
    title('Road Network Overlay (K-Means Guided)');
    roadNetworkOverlay = overlay;
end
