% -------------------------------------------------------------------------
% Fast Mesh-Focused 3D Reconstruction 
% Shitty results tho
% -------------------------------------------------------------------------
clear; clc; close all;

fprintf('=== Fast Mesh Reconstruction ===\n');

imageDir = '/Users/tordnatlandsmyr/Desktop/computer-vision-challenge/Datasets/Frauenkirche';
try
    imds = imageDatastore(imageDir);
    fprintf('Found %d images\n', numel(imds.Files));
catch
    error('Cannot read images from: %s', imageDir);
end

% Load and resize images
images = cell(numel(imds.Files), 1);
grayImages = cell(numel(imds.Files), 1);

for i = 1:numel(imds.Files)
    I = readimage(imds, i);

    images{i} = I;
    grayImages{i} = rgb2gray(I);
    fprintf('   Loaded image %d\n', i);
end

fprintf('2. Computing features...\n');

allFeatures = cell(length(images), 1);
allPoints = cell(length(images), 1);

for i = 1:length(images)
    points = detectSURFFeatures(grayImages{i}, 'MetricThreshold', 20);
    [features, validPoints] = extractFeatures(grayImages{i}, points);
    
    allFeatures{i} = features;
    allPoints{i} = validPoints;
    
    fprintf('   Image %d: %d features\n', i, validPoints.Count);
end

fprintf('3. Finding best pairs...\n');

bestPairs = [];
pairsChecked = 0;
maxPairsToCheck = 1000;  % Stop after finding enough good pairs

for i = 1:length(images)
    for j = i+1:length(images)
        pairsChecked = pairsChecked + 1;
        
        % Quick feature matching
        indexPairs = matchFeatures(allFeatures{i}, allFeatures{j}, ...
            'MaxRatio', 0.75, 'Unique', true);
        
        fprintf('   Pair %d-%d: %d matches', i, j, size(indexPairs, 1));
        
        if size(indexPairs, 1) > 20
            matchedPoints1 = allPoints{i}(indexPairs(:, 1));
            matchedPoints2 = allPoints{j}(indexPairs(:, 2));
            
            % Quick disparity check
            disparities = sqrt(sum((matchedPoints1.Location - matchedPoints2.Location).^2, 2));
            avgDisparity = mean(disparities);
            
            if avgDisparity > 3 && avgDisparity < 150
                quality = size(indexPairs, 1) * min(avgDisparity, 50);  % Cap disparity weight
                bestPairs(end+1, :) = [i, j, quality, size(indexPairs, 1)];
                fprintf(' ✓ (disparity %.1f)\n', avgDisparity);
                
                % Early termination
                if size(bestPairs, 1) >= 5
                    fprintf('   Found enough pairs, stopping search\n');
                    break;
                end
            else
                fprintf(' ✗ (bad disparity %.1f)\n', avgDisparity);
            end
        else
            fprintf(' ✗ (too few matches)\n');
        end
        
        % Safety limit
        if pairsChecked >= maxPairsToCheck
            fprintf('   Checked %d pairs, stopping\n', maxPairsToCheck);
            break;
        end
    end
    
    if size(bestPairs, 1) >= 5 || pairsChecked >= maxPairsToCheck
        break;
    end
end

if isempty(bestPairs)
    error('No valid pairs found');
end

% Sort and take best pairs
[~, sortIdx] = sort(bestPairs(:, 3), 'descend');
bestPairs = bestPairs(sortIdx, :);

fprintf('Found %d valid pairs\n', size(bestPairs, 1));

fprintf('4. Stereo reconstruction...\n');

allPoints3D = [];
allColors = [];

% Process top 3 pairs max
numPairs = min(3, size(bestPairs, 1));

for pairIdx = 1:numPairs
    i = bestPairs(pairIdx, 1);
    j = bestPairs(pairIdx, 2);
    
    fprintf('   Processing pair %d-%d...\n', i, j);
    
    [points3D, colors] = fastStereoReconstruction(grayImages{i}, grayImages{j}, ...
        images{i}, allFeatures{i}, allFeatures{j}, allPoints{i}, allPoints{j});
    
    if ~isempty(points3D)
        allPoints3D = [allPoints3D; points3D];
        allColors = [allColors; colors];
        fprintf('     Added %d points\n', size(points3D, 1));
    else
        fprintf('     No points generated\n');
    end
end

fprintf('   Total points: %d\n', size(allPoints3D, 1));
fprintf('5. Generating mesh...\n');

if size(allPoints3D, 1) > 30
    % Quick cleanup
    [allPoints3D, uniqueIdx] = unique(round(allPoints3D * 50) / 50, 'rows');
    allColors = allColors(uniqueIdx, :);
    
    % Simple outlier removal
    [allPoints3D, allColors] = fastOutlierRemoval(allPoints3D, allColors);
    
    % Generate mesh
    [faces, vertices, vertexColors] = fastMeshGeneration(allPoints3D, allColors);
    
    fprintf('   Generated mesh: %d vertices, %d faces\n', size(vertices, 1), size(faces, 1));
else
    faces = [];
    vertices = allPoints3D;
    vertexColors = allColors;
    fprintf('   Insufficient points (%d) for mesh\n', size(allPoints3D, 1));
end

fprintf('6. Visualization...\n');

fastVisualization(images, allPoints3D, allColors, faces, vertices, vertexColors);

fprintf('=== Reconstruction Complete ===\n');
fprintf('Final: %d points, %d triangles\n', size(allPoints3D, 1), size(faces, 1));


function [points3D, colors] = fastStereoReconstruction(gray1, gray2, colorImg1, features1, features2, points1, points2)
    
    points3D = [];
    colors = [];
    
    % Quick matching
    indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.65, 'Unique', true);
    
    if size(indexPairs, 1) < 15
        return;
    end
    
    matchedPoints1 = points1(indexPairs(:, 1));
    matchedPoints2 = points2(indexPairs(:, 2));
    
    try
        % Fundamental matrix estimation
        [F, inlierIdx] = estimateFundamentalMatrix(...
            matchedPoints1.Location, matchedPoints2.Location, ...
            'Method', 'RANSAC', 'NumTrials', 1000, ...  % Reduced trials
            'DistanceThreshold', 2.0);
        
        if sum(inlierIdx) < 10
            return;
        end
        
        % Simple camera model
        imageSize = size(gray1);
        focalLength = max(imageSize) * 0.8; % Biggest assumption
        principalPoint = [imageSize(2)/2, imageSize(1)/2];
        K = [focalLength, 0, principalPoint(1); 
             0, focalLength, principalPoint(2); 
             0, 0, 1];
        
        % Essential matrix and pose
        E = K' * F * K;
        [R, t] = fastPoseEstimation(E, matchedPoints1(inlierIdx), matchedPoints2(inlierIdx), K);
        
        if isempty(R)
            return;
        end
        
        % Sparse 3D points from matches (faster than dense)
        [points3D, colors] = sparseTriangulation(matchedPoints1(inlierIdx), ...
            matchedPoints2(inlierIdx), colorImg1, R, t, K);
        
    catch ME
        fprintf('     Fast stereo failed: %s\n', ME.message);
    end
end

function [R, t] = fastPoseEstimation(E, points1, points2, K)
    
    R = [];
    t = [];
    
    [U, ~, V] = svd(E);
    if det(U) < 0, U = -U; end
    if det(V) < 0, V = -V; end
    
    W = [0 -1 0; 1 0 0; 0 0 1];
    R_candidates = cat(3, U * W * V', U * W' * V');
    t_candidates = [U(:, 3), -U(:, 3)];
    
    % Quick cheirality check with subset
    P1 = K * [eye(3), zeros(3,1)];
    maxValid = 0;
    subset = 1:min(10, points1.Count);  % Small subset for speed
    
    for i = 1:size(R_candidates, 3)
        for j = 1:size(t_candidates, 2)
            P2 = K * [R_candidates(:,:,i), t_candidates(:,j)];
            
            X = fastTriangulate(P1, P2, points1(subset).Location', points2(subset).Location');
            
            % Count valid points
            valid = sum(X(3,:) > 0 & X(3,:) < 50);
            
            if valid > maxValid
                maxValid = valid;
                R = R_candidates(:,:,i);
                t = t_candidates(:,j);
            end
        end
    end
end

function X = fastTriangulate(P1, P2, x1, x2)
    % Triangulation for small point sets
    
    numPoints = size(x1, 2);
    X = zeros(4, numPoints);
    
    for i = 1:numPoints
        A = [x1(1,i) * P1(3,:) - P1(1,:);
             x1(2,i) * P1(3,:) - P1(2,:);
             x2(1,i) * P2(3,:) - P2(1,:);
             x2(2,i) * P2(3,:) - P2(2,:)];
        
        [~, ~, V] = svd(A, 'econ');
        X(:,i) = V(:,end);
        
        if X(4,i) ~= 0
            X(:,i) = X(:,i) / X(4,i);
        end
    end
end

function [points3D, colors] = sparseTriangulation(points1, points2, colorImg, R, t, K)
    % Triangulate matched feature points (sparse but fast)
    
    P1 = K * [eye(3), zeros(3,1)];
    P2 = K * [R, t];
    
    pts1 = points1.Location';
    pts2 = points2.Location';
    
    X = fastTriangulate(P1, P2, pts1, pts2);
    
    % Filter valid points
    validIdx = X(3,:) > 0.1 & X(3,:) < 30 & ...
               abs(X(1,:)) < 15 & abs(X(2,:)) < 15;
    
    points3D = X(1:3, validIdx)';
    
    % Get colors ad pix location
    colors = zeros(sum(validIdx), 3);
    validPoints = pts1(:, validIdx);
    
    for i = 1:size(validPoints, 2)
        x = round(validPoints(1, i));
        y = round(validPoints(2, i));
        
        % Bounds check
        if x >= 1 && x <= size(colorImg, 2) && y >= 1 && y <= size(colorImg, 1)
            if size(colorImg, 3) == 3
                colors(i, :) = double(squeeze(colorImg(y, x, :))') / 255;
            else
                gray = double(colorImg(y, x)) / 255;
                colors(i, :) = [gray, gray, gray];
            end
        else
            colors(i, :) = [0.5, 0.5, 0.5];  % Default gray
        end
    end
end

function [cleanPoints, cleanColors] = fastOutlierRemoval(points3D, colors)
    % Fast statistical outlier removal
    
    if size(points3D, 1) < 10
        cleanPoints = points3D;
        cleanColors = colors;
        return;
    end
    
    % Simple distance-based filtering
    center = median(points3D, 1);  % Use median for robustness
    distances = sqrt(sum((points3D - center).^2, 2));
    threshold = prctile(distances, 95);  % Keep 95% of points
    
    inlierIdx = distances <= threshold;
    cleanPoints = points3D(inlierIdx, :);
    cleanColors = colors(inlierIdx, :);
    
    fprintf('     Removed %d outliers\n', sum(~inlierIdx));
end

function [faces, vertices, vertexColors] = fastMeshGeneration(points3D, colors)
    % Fast mesh generation without normal computation
    
    faces = [];
    vertices = points3D;
    vertexColors = colors;
    
    if size(points3D, 1) < 10
        return;
    end
    
    try
        % Simple 2D triangulation on dominant plane, this is not robust to
        % noise
        [~, ~, V] = svd(points3D - mean(points3D, 1), 'econ');
        
        % Project to 2D using first two principal components
        points2D = points3D * V(:, 1:2);
        
        % 2D Delaunay triangulation (much faster than 3D)
        DT = delaunayTriangulation(points2D);
        faces = DT.ConnectivityList;
        
        % Filter by edge length in 3D
        maxEdgeLength = 2.0;
        validFaces = [];
        
        for i = 1:size(faces, 1)
            triangle = faces(i, :);
            p1 = points3D(triangle(1), :);
            p2 = points3D(triangle(2), :);
            p3 = points3D(triangle(3), :);
            
            edges = [norm(p2-p1), norm(p3-p2), norm(p1-p3)];
            
            if max(edges) <= maxEdgeLength
                validFaces(end+1, :) = triangle;
            end
        end
        
        faces = validFaces;
        
    catch ME
        fprintf('     Fast mesh failed: %s\n', ME.message);
        % Fallback: convex hull
        try
            [faces, vertices] = convhull(points3D);
            
            % Map colors
            [~, colorIdx] = ismember(vertices, points3D, 'rows');
            validIdx = colorIdx > 0;
            vertexColors = zeros(size(vertices, 1), 3);
            vertexColors(validIdx, :) = colors(colorIdx(validIdx), :);
        catch
            faces = [];
        end
    end
end

function fastVisualization(images, points3D, colors, faces, vertices, vertexColors)
    % Fast visualization
    
    % Input images
    figure('Name', 'Input Images', 'Position', [100, 100, 800, 200]);
    numShow = min(3, length(images));
    for i = 1:numShow
        subplot(1, numShow, i);
        imshow(images{i});
        title(sprintf('Image %d', i), 'FontSize', 10);
    end
    
    % Results
    figure('Name', 'Fast 3D Reconstruction', 'Position', [200, 200, 1000, 400]);
    
    if ~isempty(points3D)
        subplot(1, 2, 1);
        scatter3(points3D(:,1), points3D(:,2), points3D(:,3), 25, colors, 'filled');
        title(sprintf('Point Cloud (%d points)', size(points3D, 1)));
        xlabel('X'); ylabel('Y'); zlabel('Z');
        axis equal; grid on; view(3); rotate3d on;
        
        subplot(1, 2, 2);
        if ~isempty(faces) && size(faces, 1) > 0
            trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), ...
                'FaceVertexCData', vertexColors, 'FaceColor', 'interp', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.9);
            title(sprintf('Mesh (%d triangles)', size(faces, 1)));
            material dull; lighting gouraud; camlight('headlight');
            xlabel('X'); ylabel('Y'); zlabel('Z');
            axis equal; grid on; view(3); rotate3d on;
        else
            scatter3(points3D(:,1), points3D(:,2), points3D(:,3), 35, colors, 'filled');
            title('Point Cloud Only');
            xlabel('X'); ylabel('Y'); zlabel('Z');
            axis equal; grid on; view(3); rotate3d on;
        end
    else
        text(0.5, 0.5, 'No reconstruction generated', 'HorizontalAlignment', 'center');
        title('Reconstruction Failed');
    end
end