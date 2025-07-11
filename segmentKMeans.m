function segImg = segmentKMeans(imagePath, K)
    I = im2double(imread(imagePath));
    [h,w,~] = size(I);
    X = reshape(I, h*w, 3);               % each row = [R G B]

    % run k-means (random init, 5 replicates)
    [idx, C] = kmeans(X, K, ...
                     'Replicates',5, ...
                     'MaxIter',200);

    % reshape back and color each pixel by its cluster center
    segImg = reshape(C(idx,:), h, w, 3);

    figure;
    subplot(1,2,1), imshow(I),    title('Original');
    subplot(1,2,2), imshow(segImg), title([num2str(K) '-Means Segmentation']);
end
