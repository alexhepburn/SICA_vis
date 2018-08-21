function [W, multipliers] = findOptimumW(X, A, x0)
    %% compute parameters for Lagrangian
    n = size(X,1);
    d = size(X,2);
    epsilon = sum(sum(A))/2;
    b = trace(X'*X)/n;
    D = diag(sum(A));
    L = D - A;
    c = trace(X'*L*X)/epsilon;
    %% find optimum
    options = optimoptions(@fminunc, 'Algorithm', 'trust-region',...
        'GradObj', 'on', 'Hessian', 'on', 'MaxFunEvals', 1e05,...
        'MaxIter', 5000, 'Display', 'off');
    x_start = [0.1 0.1];
    [multipliers,~,~,~] = fminunc(@brownfgh, x_start, options);
    %% compute projection matricx.
    if nargin == 3
        multipliers = x0;
    end
    K = X'*(multipliers(1)*eye(size(X,1)) + multipliers(2)*L)*X;
    [W, eigValues] = eig(K);
    [~, sortedEigValueIdx] = sort(diag(eigValues), 'descend');
    W = W(:, sortedEigValueIdx);
    function [f,g,H] = brownfgh(w)
        invF = inv(2*(w(1)*eye(n) + w(2)*L));
        f = lagrange_dual(w(1), w(2), n, d, b, c, epsilon, L);
        g= [-d*trace(invF*eye(n))+n*b,...
         -d*trace(invF*L) + epsilon*c];
        H = [-2*d*trace(-invF *(eye(n))* invF * eye(n)) ... 
         -2*d*trace(-invF * L * invF * eye(n)); ...
         -2*d*trace(-invF * eye(n)* invF *L),...
         -2*d*trace(-invF * L* invF * L)];
    end
    function value = lagrange_dual(lambda, mu, n, d, b, c, epsilon, L)
        invSigma = 2*(lambda*eye(n) + mu*L);
        logZ = d/2 * (n*log(2*pi) - sum(log(eig(invSigma))));
        value = logZ + lambda*n*b + mu*epsilon*c;
    end
end

