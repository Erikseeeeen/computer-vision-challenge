%% Main Script: Register and crop batch images

scriptDir = fileparts(mfilename('fullpath'));
baseDir   = fullfile(scriptDir, 'Datasets');

dirs = dir(baseDir);
dirs = dirs([dirs.isdir]);
dirs = dirs(~ismember({dirs.name}, {'.','..'}));

for d = dirs'
    subdir = fullfile(baseDir, d.name);
    subDirs = dir(subdir);
    subDirs = subDirs([subDirs.isdir]);
    subDirs = subDirs(~ismember({subDirs.name}, {'.','..'}));
    if ~isempty(subDirs), continue; end
    
    folder = [subdir filesep];
    files1 = dir(fullfile(folder, '*.jpg'));
    files2 = dir(fullfile(folder, '*.JPG'));
    imageFiles = [files1; files2];
    if isempty(imageFiles), warning('No images in %s', folder); continue; end
    [~, idx] = sort({imageFiles.name});
    imageFiles = imageFiles(idx);
    n = numel(imageFiles);

    IrefRGB  = imread(fullfile(folder, imageFiles(1).name));
    IrefGray = im2double(rgb2gray(IrefRGB));
    [h, w]   = size(IrefGray);
    roi      = [1,1,w,floor(h*0.9)];

    outputView = imref2d(size(IrefGray));
    tforms     = repmat(affine2d(eye(3)),1,n);

    for k = 2:n
        I2 = im2double(rgb2gray(histMatchToRef(imread(fullfile(folder, imageFiles(k).name)), IrefRGB)));
        try
            p1 = detectSURFFeatures(IrefGray,'ROI',roi,'MetricThreshold',200);
            p2 = detectSURFFeatures(I2,'ROI',roi,'MetricThreshold',200);
            [f1,v1] = extractFeatures(IrefGray,p1);
            [f2,v2] = extractFeatures(I2,p2);
            pairs = matchFeatures(f1,f2,'Unique',true);
            m1 = v1(pairs(:,1)); m2 = v2(pairs(:,2));
            tforms(k) = estimateGeometricTransform(m2,m1,'similarity', ...
                             'MaxDistance',35,'Confidence',99.9,'MaxNumTrials',5000);
        catch
            tforms(k) = imregtform(I2, IrefGray,'similarity',optimizerConfig(),metricConfig());
        end
    end

    BB = zeros(n,4);
    for k = 1:n
        I = imread(fullfile(folder, imageFiles(k).name));
        if k>1
            I = imwarp(histMatchToRef(I,IrefRGB),tforms(k),'OutputView',outputView);
        end
        m = any(I~=0,3);
        BB(k,:) = regionprops(m,'BoundingBox').BoundingBox;
    end

    x0 = max(BB(:,1)); y0 = max(BB(:,2));
    x1 = min(BB(:,1)+BB(:,3)); y1 = min(BB(:,2)+BB(:,4));
    rect = [floor(x0)+1,floor(y0)+1,floor(x1-x0),floor(y1-y0)];

    outDir = fullfile(folder,'common_crop');
    if ~exist(outDir,'dir'), mkdir(outDir); end

    for k = 1:n
        I = imread(fullfile(folder, imageFiles(k).name));
        if k>1
            I = imwarp(histMatchToRef(I,IrefRGB),tforms(k),'OutputView',outputView);
        end
        sz = size(I);
        xE = min(rect(1)+rect(3)-1,sz(2));
        yE = min(rect(2)+rect(4)-1,sz(1));
        Icrop = imcrop(I,[rect(1),rect(2),xE-rect(1),yE-rect(2)]);
        if ~isempty(Icrop)
            imwrite(Icrop, fullfile(outDir, imageFiles(k).name));
        end
    end
end

%% Helper functions
function Iout = histMatchToRef(Iin,Iref)
    Iout = zeros(size(Iin),'like',Iin);
    for c=1:3
        Iout(:,:,c) = imhistmatch(Iin(:,:,c),Iref(:,:,c));
    end
end

function opt = optimizerConfig()
    opt = registration.optimizer.OnePlusOneEvolutionary;
    opt.GrowthFactor = 1.05;
    opt.InitialRadius= 0.006;
    opt.Epsilon      = 1.5e-4;
    opt.MaximumIterations = 200;
end

function m = metricConfig()
    m = registration.metric.MeanSquares;
end
