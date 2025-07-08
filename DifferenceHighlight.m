function overlay = DifferenceHighlight(A, B) 
    differenceTolerance = 10;
    A = double(A);
    B = double(B);

    differenceMask = max(abs(A - B),[],3) > differenceTolerance;
    overlay = zeros(size(A), 'like', A);
    
    overlay(:,:,1) = differenceMask;
end