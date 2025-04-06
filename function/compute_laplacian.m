function L = compute_laplacian(W)
    D = diag(sum(W, 2));
    L = D - W;
end

