function results = roadNetworkExtraction(imagePath, varargin)
% roadNetworkExtraction  Extract and overlay road networks via multiple methods
%
%   results = roadNetworkExtraction(imagePath)
%       uses default Method = 'Ensemble' (fuses all methods):
%         NumClusters   = 5
%         Sensitivity = 0.10
%         ChanIter      = 200
%
%   results = roadNetworkExtraction(..., 'Method', M, ...)
%       M ∈ {'Ensemble','KMeansCanny','Geometry','ActiveContour'}
%
%   results = roadNetworkExtraction(..., 'NumClusters', K)
%   results = roadNetworkExtraction(..., 'Sensitivity', E)
%   results = roadNetworkExtraction(..., 'ChanIter', N)
%
%   In results struct, each method now gets its own
%   colored overlay:
%     • KMeansCanny  – red edges
%     • Geometry     – green edges
%     • ActiveContour– magenta contours
%     • Ensemble     – union of all three (red, green, magenta)
%
% Example
%   R = roadNetworkExtraction('img.jpg','Method','ActiveContour');
%   imshow(R.overlay);

  %% Parse inputs
  p = inputParser;
  addRequired(  p, 'imagePath',    @(x)ischar(x)||isstring(x));
  addParameter(p, 'Method',        'Ensemble',     @(x)ischar(x)||isstring(x));
  addParameter(p, 'NumClusters',   5,               @(x)isnumeric(x)&&isscalar(x)&&x>0);
  addParameter(p, 'Sensitivity', 0.10,            @(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<=1);
  addParameter(p, 'ChanIter',      200,             @(x)isnumeric(x)&&isscalar(x)&&x>0);
  parse(p, imagePath, varargin{:});

  method      = validatestring(string(p.Results.Method), ...
                               ["Ensemble","KMeansCanny","Geometry","ActiveContour"]);
  K           = p.Results.NumClusters;
  epsilon     = p.Results.Sensitivity;
  numChanIter = p.Results.ChanIter;

  %% Read input and init
  img = imread(imagePath);
  [M,N,~] = size(img);

  results = struct();
  results.originalImage  = img;
  results.selectedMethod = method;

  %% Dispatch by method
  switch method

    case "Ensemble"
      % run each sub-method
      kmRes   = roadNetworkExtraction(imagePath, 'Method','KMeansCanny', ...
                                    'NumClusters',K,'Sensitivity',epsilon);
      geomRes = roadNetworkExtraction(imagePath, 'Method','Geometry', ...
                                    'NumClusters',K,'Sensitivity',epsilon);
      acRes   = roadNetworkExtraction(imagePath, 'Method','ActiveContour', ...
                                    'ChanIter',numChanIter);

      % combine masks
      m1 = kmRes.edgeMask;      % red
      m2 = geomRes.finalMask;   % green
      m3 = acRes.finalMask;      % magenta
      ensembleMask = m1 | m2 | m3;

      % build overlay: red + green + magenta
      alphaMap = uint8(ensembleMask)*255;
      overlay  = img;
      % red channel: union of all
      overlay(:,:,1) = max(overlay(:,:,1), alphaMap);
      % green channel: only geometry
      overlay(:,:,2) = max(overlay(:,:,2), uint8(m2)*255);
      % blue channel: magenta & geometry share blue
      overlay(:,:,3) = max(overlay(:,:,3), uint8(m3)*255);

      results.kmeansCanny   = kmRes;
      results.geometry      = geomRes;
      results.activeContour = acRes;
      results.ensembleMask  = ensembleMask;
      results.overlayAlpha  = alphaMap;
      results.overlay       = overlay;

    case "KMeansCanny"
      % 1) K-means
      X = reshape(double(img), M*N, 3);
      [L, C] = kmeans(X, K, 'MaxIter',1000,'Replicates',3);
      labelMap = reshape(L, M, N);

      % 2) road-like clusters
      grayC = (0.2989*C(:,1)+0.5870*C(:,2)+0.1140*C(:,3))/255;
      rIDs  = find(grayC>=0.3 & grayC<=0.8);
      roadMask = ismember(labelMap, rIDs);
      roadMask = bwareaopen(roadMask,200);
      roadMask = imclose(roadMask, strel('disk',5));

      % 3) Canny edges in mask
      g = rgb2gray(img);
      edgeMask = edge(g,'Canny',epsilon) & roadMask;
      edgeMask = bwareaopen(edgeMask,20);

      % 4) red overlay
      alphaMap = uint8(edgeMask)*255;
      overlay  = img;
      overlay(:,:,1) = max(overlay(:,:,1), alphaMap);  % red
      overlay(:,:,2) = overlay(:,:,2).*uint8(~edgeMask);
      overlay(:,:,3) = overlay(:,:,3).*uint8(~edgeMask);

      % pack
      results.numClusters      = K;
      results.edgeThreshold    = epsilon;
      results.clusterIndexMap  = labelMap;
      results.clusterCentroids = C;
      results.roadMask         = roadMask;
      results.edgeMask         = edgeMask;
      results.overlayAlpha     = alphaMap;
      results.overlay          = overlay;

    case "Geometry"
      % 1) initial K-means mask
      X = reshape(double(img), M*N, 3);
      [L, C] = kmeans(X, K, 'MaxIter',1000,'Replicates',3);
      labelMap = reshape(L, M, N);
      grayC = (0.2989*C(:,1)+0.5870*C(:,2)+0.1140*C(:,3))/255;
      rIDs = find(grayC>=0.3 & grayC<=0.8);
      initMask = ismember(labelMap, rIDs);
      initMask = bwareaopen(initMask,200);
      initMask = imclose(initMask, strel('disk',5));

      % 2) Canny edges
      g = rgb2gray(img);
      edgeMask = edge(g,'Canny',epsilon) & initMask;
      edgeMask = bwareaopen(edgeMask,20);

      % 3) PCA straight lines
      cc = bwconncomp(edgeMask);
      strM = false(M,N); minLen=50; thr=0.92;
      for i=1:cc.NumObjects
        pix = cc.PixelIdxList{i};
        if numel(pix)<minLen, continue; end
        [r,c] = ind2sub([M,N],pix);
        P = [c r] - mean([c r],1);
        ev = eig(cov(P));
        if max(ev)/sum(ev) >= thr, strM(pix)=true; end
      end
      strM = bwmorph(strM,'bridge');
      strM = bwmorph(strM,'thicken',1);

      % 4) curved segments
      skel = bwmorph(initMask,'skel',Inf);
      cc2  = bwconncomp(skel);
      rp   = regionprops(cc2,'PixelList');
      curvM = false(M,N);
      for i=1:cc2.NumObjects
        pts = rp(i).PixelList;
        if size(pts,1)<20, continue; end
        d = diff(pts);
        a = atan2(d(:,2),d(:,1));
        v = var(unwrap(a));
        if v>0.001 && v<0.05
          curvM(sub2ind([M,N],pts(:,2),pts(:,1))) = true;
        end
      end

      % fuse & green overlay
      finalMask = (strM|curvM) & initMask;
      finalMask = imdilate(finalMask, strel('line',3,90));
      alphaMap = uint8(finalMask)*255;
      overlay  = img;
      overlay(:,:,1) = overlay(:,:,1).*uint8(~finalMask);
      overlay(:,:,2) = max(overlay(:,:,2), alphaMap);  % green
      overlay(:,:,3) = overlay(:,:,3).*uint8(~finalMask);

      % pack
      results.numClusters      = K;
      results.edgeThreshold    = epsilon;
      results.clusterIndexMap  = labelMap;
      results.clusterCentroids = C;
      results.initialMask      = initMask;
      results.straightMask     = strM;
      results.curvedMask       = curvM;
      results.finalMask        = finalMask;
      results.overlayAlpha     = alphaMap;
      results.overlay          = overlay;

    case "ActiveContour"
      % 1) grayscale & smooth
      if size(img,3)==3
        gray = rgb2gray(img);
      else
        gray = img;
      end
      smoothI = imgaussfilt(gray,2);

      % 2) initial mask
      lvl   = graythresh(smoothI);
      initM = imbinarize(smoothI,lvl);
      initM = imfill(initM,'holes');

      % 3) evolve contour
      bw = activecontour(smoothI,initM,numChanIter,'Chan-Vese');

      % 4) cleanup
      bw = bwareaopen(bw,300);
      bw = imclose(bw, strel('line',15,0));
      bw = imclose(bw, strel('line',15,90));
      finalM = imfill(bw,'holes');

      % 5) magenta overlay
      alphaMap = uint8(finalM)*255;
      overlay  = img;
      overlay(:,:,1) = max(overlay(:,:,1), alphaMap);  % red
      overlay(:,:,2) = overlay(:,:,2).*uint8(~finalM);
      overlay(:,:,3) = max(overlay(:,:,3), alphaMap);  % blue

      % pack
      results.grayImage     = gray;
      results.smoothedImage = smoothI;
      results.initMask      = initM;
      results.finalMask     = finalM;
      results.overlayAlpha  = alphaMap;
      results.overlay       = overlay;

    otherwise
      error('Unsupported method "%s".', method);
  end
end

