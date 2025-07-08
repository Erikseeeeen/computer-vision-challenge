function [overlay, differenceMask] = difference(A, B, tol)
    A = im2double(imread(A));
    B = im2double(imread(B));

    differenceMask = max(abs(A - B),[],3) > tol;
    overlay  = im2uint8(A);

    overlayR = overlay(:,:,1);
    overlayG = overlay(:,:,2);
    overlayB = overlay(:,:,3);
    overlayR(differenceMask) = 255;
    overlayG(differenceMask) = 0;
    overlayB(differenceMask) = 0;
    overlay = cat(3, overlayR, overlayG, overlayB);
    
end