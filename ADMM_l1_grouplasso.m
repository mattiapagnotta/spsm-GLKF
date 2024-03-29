
function [z, history] = ADMM_l1_grouplasso(A, b, lambda, p, rho, alpha)

%==========================================================================
% M.F.Pagnotta (January 2019).
%==========================================================================
% Use the Alternating Direction Method of Multipliers (ADMM) to solve group
% lasso (sum-of-norms regularization). The group lasso involves solving:
%
%   minimize (1/2)*||Ax-b||_2^2 + lambda*sum(||x_i||_2)
% 
% where lambda > 0 is the regularization parameter (usually chosen by
% cross-validation).
% 
% The scaled form of ADMM is here used, which consists of combining linear
% and quadratic terms in the augmented Lagrangian and scale the dual
% variable (u = (1/rho)*y).
% 
%--------------------------------------------------------------------------
% INPUT
% - A:          
% - b:          
% - lambda:     regularization parameter (scalar, > 0)
% - p:          K-element vector giving the block sizes n_i, so that x_i is
%               in R^{n_i}
% - rho:        augmented Lagrangian parameter (scalar, > 0)
% - alpha:      over-relaxation parameter (scalar with typical values in
%               [1,1.8])
%--------------------------------------------------------------------------
% OUTPUT
% - z:          solution (vector)
% - history:    structures containing objective values, primal and dual
%               residual norms, and their tolerances, at each iteration
%==========================================================================
% This function is inspired by the code of Stephen P. Boyd.
% More information can be found at the following link:
% https://web.stanford.edu/~boyd/papers/admm_distr_stats.html
% 
% References:
% [1] Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
%     Distributed optimization and statistical learning via the alternating
%     direction method of multipliers. Foundations and Trends� in Machine
%     Learning, 3(1), 1-122.
% [2] Yuan, M., & Lin, Y. (2006). Model selection and estimation in
%     regression with grouped variables. Journal of the Royal Statistical
%     Society: Series B (Statistical Methodology), 68(1), 49-67.
%==========================================================================


%% Global constants and defaults
maxiter     = 1000;                     % maximum number of iterations
eps_abs     = 1e-4;                     % absolute tolerance
eps_rel     = 1e-2;                     % relative tolerance

adj_penalty = 1;                        % If true: adjust penalty parameter at each iteration
%____________________
if adj_penalty==1
    tau_incr = 2;                       % parameter for inflating
    tau_decr = 2;                       % parameter for deflating
    par_mu   = 10;                     	% try to keep the two norms within a factor of par_mu
end
%____________________



%% Initialization
[m, n] = size(A);

ATb = A'*b;

% Check that sum(p) = total number of elements in x:
if (sum(p) ~= n),  error('invalid partition');
end
% Cumulative partition:
cum_part = cumsum(p);



%% ADMM solver
% Preallocation:
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if adj_penalty==0
    % Step 0. Factorization caching:
    [L, U] = factor(A, rho);
end


% Iterate ...
for k = 1:maxiter
    
    if adj_penalty==1
        % Step 0. Factorization caching:
        [L, U] = factor(A, rho);                                            % rho may change at each iteration
    end

    % Step 1. x-update ( x = (A'A+rhoI)^(-1)(A'b+rho(z-u)) ):
    q = ATb + rho*(z - u);
    if m >= n
       x = U \ (L \ q);
    else
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    % Step 2. z-update with relaxation:
    zold  = z;
    x_hat = alpha*x + (1-alpha)*zold;
    start_ind = 1;
    for i = 1:length(p)
        sel = start_ind:cum_part(i);
        z(sel) = shrinkage(x_hat(sel)+u(sel), lambda/rho);
        start_ind = cum_part(i) + 1;
    end
    
    % Step 3. u-update:
    u = u + x_hat - z;

    % Step 4. Convergence verification (stopping criteria):
    history.objval(k)   = objective(A, b, lambda, cum_part, x, z);          % objective value (k-th iteration)
    history.r_norm(k)   = norm(x - z);                                      % primal residual norm
    history.s_norm(k)   = norm(-rho*(z - zold));                            % dual residual norm
    history.eps_pri(k)  = sqrt(n)*eps_abs + eps_rel*max(norm(x), norm(-z)); % tolerance for primal feasibility condition
    history.eps_dual(k) = sqrt(n)*eps_abs + eps_rel*norm(rho*u);            % tolerance for dual feasibility condition
    % termination criteria:
    primal_crit = history.r_norm(k) <= history.eps_pri(k);                  % criterion on primal residuals
    dual_crit   = history.s_norm(k) <= history.eps_dual(k);                 % criterion on dual residuals
    if and(primal_crit, dual_crit)
        break;
    end
    
    
    % Step 5. Adjust penalty parameter [3].
    % Because ADMM is in the scaled form , the scaled dual variable u must
    % also be rescaled after updating rho:
    if adj_penalty==1
        if history.r_norm(k) > par_mu*history.s_norm(k)
            rho = tau_incr*rho;
            u   = u/tau_incr;
        elseif history.s_norm(k) > par_mu*history.r_norm(k)
            rho = rho/tau_decr;
            u   = u*tau_decr;
        end
        history.rho(k) = rho;
    end

end








%==========================================================================
%% FUNCTIONS
%==========================================================================
%--------------------------------------------------------------------------
function p = objective(A, b, lambda, cum_part, x, z)
obj = 0;
start_ind = 1;
for i = 1:length(cum_part)
    sel = start_ind:cum_part(i);
    obj = obj + norm(z(sel));
    start_ind = cum_part(i) + 1;
end
p = 1/2*sum((A*x - b).^2) + lambda*obj;


%--------------------------------------------------------------------------
function z = shrinkage(x, kappa)
% z = pos(1 - kappa/norm(x))*x;
if norm(x)==0
    z=0;
    return;
end
t = (1-kappa/norm(x));
t = max(t,0);
z = t*x;


%--------------------------------------------------------------------------
function [L, U] = factor(A, rho)
[m, n] = size(A);
if m >= n
    L = chol( A'*A + rho*speye(n), 'lower' );
else
    L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
end
% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');



