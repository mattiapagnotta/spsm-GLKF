
function [AR, obsN_cov] = aMVAR_spsmGLKF_ADMM(Y, p, c1, c2, Approach, SPARSE, smoother)

%==========================================================================
% [M.F.Pagnotta and D.Pascucci - January 2019]
% Implementation of the General Linear Kalman Filter (GLKF) using lasso
% (l1-norm) or group lasso (sum-of norms regularization in the statistical
% linear regression framework) penalty to enhance sparsity, and
% Rauch–Tung-Striebel (RTS) smoother to estimates variability.
% -------
% Last modified: 02.04.2019
%--------------------------------------------------------------------------
% INPUT
% - Y:          data [nTrial, nNodes, nSamp]
% - p:          model order (scalar)
% - c1,c2:      adaptation constants (scalar)
% - Approach:   way of dealing with multiple trials (options: 'AA' for
%               single-trial modeling followed by averaging; 'MT' for
%               multi-trial modeling, which means that one model is
%               simultaneously fitted to all trials)
% - SPARSE:     structure with info for sparsity
%                   .method (lasso: 'lasso_ADMM'; group-lasso: 'grplasso_ADMM')
%                   .level  ('all' or 'receiver')
%                   .lambda (regularization parameter)
%                   .filter (use zero-constraints only, other estimates same)
% - smoother:   type of smoother (for now only 'RTS')
%--------------------------------------------------------------------------
% OUTPUT
% - AR:         MVAR model coefficients [nNodes, nNodes, p, nSamp]
% - obsN_cov:   observation noise covariance matrix [nNodes, nNodes, nSamp,
%               numel(reiter)], where reiter is 1:nTrial for 'AA' and it is
%               1 for 'MT' (time-varying)
%==========================================================================
% References:
%
% -- spsm-GLKF ---
% [1] Pagnotta, M.F., Plomp, G., & Pascucci, D. (2019). A regularized and 
%     smoothed General Linear Kalman Filter for more accurate estimation of
%     time-varying directed connectivity*. 2019 41st Annual International 
%     Conference of the IEEE Engineering in Medicine and Biology Society 
%     (EMBC), 611-615.
% 
% -- GLKF ---
% [2] Milde, T., Leistritz, L., Astolfi, L., Miltner, W. H., Weiss, T.,
%     Babiloni, F., & Witte, H. (2010). A new Kalman filter approach for 
%     the estimation of high-dimensional time-variant multivariate AR
%     models and its application in analysis of laser-evoked brain
%     potentials. Neuroimage, 50(3), 960-969.
% 
% -- LASSO (ADMM) ---
% [3] Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
%     Distributed optimization and statistical learning via the alternating
%     direction method of multipliers. Foundations and Trends® in Machine
%     Learning, 3(1), 1-122.
% [4] Tibshirani, R. (1996). Regression shrinkage and selection via the
%     lasso. Journal of the Royal Statistical Society. Series B
%     (Methodological), 267-288.
% 
% -- RTS-smoother ---
% [5] Rauch, H. E., Striebel, C. T., & Tung, F. (1965). Maximum likelihood
%     estimates of linear dynamic systems. AIAA journal, 3(8), 1445-1450.
% [6] Simon, D. (2006). Optimal state estimation: Kalman, H infinity, and
%     nonlinear approaches. John Wiley & Sons.
% 
%==========================================================================
% Comments:
% + SPARSE.method    	% Multiple options available
% + SPARSE.lambda      	% scalar selected a priori (chosen by k-fold CV)
% + smoother = 'RTS'; 	% Rauch-Tung-Striebel (the only available for now)
%--------------------------------------------------------------------------

if nargin < 2,  error('Not enough input arguments');    end
if nargin < 3,  c1 = 0.02;                              end
if nargin < 4,  c2 = c1;                                end
if nargin < 5,  Approach = 'MT';                        end
if nargin < 6,  SPARSE.method = 'none';                 end
if nargin < 7,  smoother = 'none';                      end


% ------------------------------------------------------
[nTrial, nNodes, nSamp] = size(Y);

switch Approach
    % Single-trial estimation, followed by averaging:
    case 'AA'
        reiter = 1:nTrial;
        vec    = 1;
     % Multi-trial estimation (fitting to all trials together)
    case 'MT'
        reiter = 1;
        vec    = 1:nTrial;
end

A        = zeros(nNodes, nNodes*p, nSamp, numel(reiter));
obsN_cov = zeros(nNodes, nNodes, nSamp, numel(reiter));
xf_post  = zeros(nNodes*p, nNodes, nSamp);



% -- SPARSE (sparsity constraint) -----
if strcmp(SPARSE.method, 'none')
    flgSparse = 0;
elseif ~strcmp(SPARSE.method, 'none')
    flgSparse = 1;
    % Option to use l1 constraints only as a filter (force only
    % zero-values but keep unchanged the other estimates):
    if SPARSE.filter==1,        flg_L1asFILT = 1;
    elseif SPARSE.filter==0,    flg_L1asFILT = 0;
    end
end

% -- SMOOTHER -------------------------
if strcmp(smoother, 'none'),       flgSmooth = 0;
elseif ~strcmp(smoother, 'none'),  flgSmooth = 1;
end
if flgSmooth
    xf_pre  = zeros(nNodes*p, nNodes,   nSamp);                             % x: state estimates (MVAR coefficients)
    Pf_post = zeros(nNodes*p, nNodes*p, nSamp);                             % P: covariance of estimation error
    Pf_pre  = zeros(nNodes*p, nNodes*p, nSamp);
    %----
    Pk      = zeros(nNodes*p, nNodes*p, nSamp);
    xks     = zeros(nNodes*p, nNodes,   nSamp);
end




%==========================================================================
%=============================== GLKF =====================================
%==========================================================================
for t = reiter
    
    % Initialization:
    x  = zeros(nNodes*p, nNodes) + 1e-4;
    R  = eye(nNodes)*1e-2;
    P  = eye(nNodes*p)*1e-4;
    
    %----------------------------------------------------------------------
    % GLKF implementation
    %----------------------------------------------------------------------
    for k = p+1:nSamp
        
        % Observations:
        y   = Y(vec*t,:,k);
        
        % The transition matrix H at each time step is defined as a
        % function of the observations from the previous p time steps:
        H   = reshape(Y(vec*t,:,k-1:-1:k-p),[numel(vec) nNodes*p]);
        
        % Innovation (one-step prediction error - observations):
        E   = y - H*x;
        
        % Additive observation noise covariance matrix: R (W=E{ww'} in
        % [2]).
        % One way to compute R on the basis of the prediction error is
        % given by the following expression (Schack et al., 1995):
        R   = R*(1-c1) + c1*((E'*E)/(max((numel(vec)-1),1)));
        
        % The aim of minimizing the trace of the covariance matrix of the
        % estimation error leads to the following equations for the
        % computation of the Kalman gain matrix (K):
        S 	= H*P*H' + trace(R)*eye(numel(vec));
        K  	= P*H'/S;
        
        % The demand for a linear and recursive estimator for the state as
        % a function of previous state and actual observation matrix,
        % together with the demand for an unbiased estimation, results
        % into:
        xplus = x + K*E;
        xf_post(:,:,k) = xplus;
        if flgSmooth
            xf_pre(:,:,k) = x;
        end
        
        
        %------------------------------------------------------------------
        % Projection onto sparse-space
        %------------------------------------------------------------------
        if flgSparse
            rho   = 1;
            alpha = 1;
            
            if strcmp(SPARSE.level, 'all')
                tmp_xf_post = xf_post(:,:,k);
                tmp_xf_post = reshape(tmp_xf_post, [nNodes, p, nNodes]);
                tmp_xf_post = permute(tmp_xf_post, [2 1 3]);
                tmp_xf_post = reshape(tmp_xf_post, [p*nNodes*nNodes, 1]);
                % Solve lasso problem via ADMM:
                if strcmp(SPARSE.method, 'lasso_ADMM')
                    tmp_xf_post = ADMM_l1_lasso(eye(p*nNodes*nNodes), tmp_xf_post, SPARSE.lambda, rho, alpha);
                % Solve group-lasso problem via ADMM:
                elseif strcmp(SPARSE.method, 'grplasso_ADMM')
                    p_Kelement_vector = p .* ones(nNodes*nNodes,1);
                    tmp_xf_post = ADMM_l1_grouplasso(eye(p*nNodes*nNodes), tmp_xf_post, SPARSE.lambda, p_Kelement_vector, rho, alpha);
                % (If option is incorrect: error message)
                else
                    error('Method not available. Check option!')
                end
                tmp_xf_post = reshape(tmp_xf_post, [p, nNodes, nNodes]);
                tmp_xf_post = permute(tmp_xf_post, [2 1 3]);
                tmp_xf_post = reshape(tmp_xf_post, [nNodes*p, nNodes]);
                if flg_L1asFILT == 0
                    % Use l1-norm estimates
                    xf_post(:,:,k) = tmp_xf_post;
                elseif flg_L1asFILT == 1
                    % Alternative: set to zero only zero-elements from
                    % LASSO penalty
                    TMP = xf_post(:,:,k);
                    TMP([tmp_xf_post==0]) = 0;
                    xf_post(:,:,k) = TMP;    
                end
                
            elseif strcmp(SPARSE.level, 'receiver')
                for m = 1:nNodes
                    tmp_xf_post = xf_post(:,m,k);
                    tmp_xf_post = reshape(tmp_xf_post, [nNodes, p]);
                    tmp_xf_post = permute(tmp_xf_post, [2 1]);
                    tmp_xf_post = reshape(tmp_xf_post, [p*nNodes, 1]);
                    % Solve lasso problem via ADMM
                    if strcmp(SPARSE.method, 'lasso_ADMM')
                        tmp_xf_post = ADMM_l1_lasso(eye(p*nNodes), tmp_xf_post, SPARSE.lambda, rho, alpha);
                    % Solve group-lasso problem via ADMM
                    elseif strcmp(SPARSE.method, 'grplasso_ADMM')
                        p_Kelement_vector = p .* ones(nNodes,1);
                        tmp_xf_post = ADMM_l1_grouplasso(eye(p*nNodes), tmp_xf_post, SPARSE.lambda, p_Kelement_vector, rho, alpha);
                    % (If option is incorrect: error message)
                    else
                        error('Method not available. Check option!')
                    end
                    tmp_xf_post = reshape(tmp_xf_post, [p, nNodes]);
                    tmp_xf_post = permute(tmp_xf_post, [2 1]);
                    tmp_xf_post = reshape(tmp_xf_post, [nNodes*p, 1]);
                    if flg_L1asFILT == 0
                        % Use l1-norm estimates
                        xf_post(:,m,k) = tmp_xf_post;
                    elseif flg_L1asFILT == 1
                        % Alternative: set to zero only zero-elements from
                        % LASSO penalty
                        TMP = xf_post(:,m,k);
                        TMP([tmp_xf_post==0]) = 0;
                        xf_post(:,m,k) = TMP;
                    end
                end
                
            else
                error('Level not available or not specified!')
                
            end
            xplus = xf_post(:,:,k);
        end
        %------------------------------------------------------------------        
        
        
        % Since the transition matrix in the state equation is F=I:
        x   = xplus;
        
        % Covariance matrix of the state (P): Pplus = (I_dp - K*H)*P.
        % From considerations in Bar-Shalom and Fortmann (1988) that
        % expression can be transformed to:
        Pplus = P - K*S*K';
        if flgSmooth
            Pf_post(:,:,k) = Pplus;
            Pf_pre(:,:,k)  = P;
        end
        P     = Pplus;
        P(1:nNodes*p+1:end) = P(1:nNodes*p+1:end) + c2^2;                   % (Isaksson et al. 1981; Schloegl, 2000)
        
        obsN_cov(:,:,k,t) = R;
    end
    
    
    %----------------------------------------------------------------------
    % Rauch-Tung-Striebel (RTS) SMOOTHER
    %----------------------------------------------------------------------
    if flgSmooth
        % Initialize
        xks(:,:,end) = xf_post(:,:,end);
        Pk(:,:,end)  = Pf_post(:,:,end);
        % RTS smoother equations:
        for ii = (nSamp-1):-1:p+1
            Ck          = Pf_post(:,:,ii)/Pf_pre(:,:,ii+1);
            Pk(:,:,ii)  = Pf_post(:,:,ii) - Ck*(Pf_pre(:,:,ii+1)-Pk(:,:,ii+1))*Ck';
            xks(:,:,ii) = xf_post(:,:,ii) + Ck*(xks(:,:,ii+1)-xf_pre(:,:,ii+1));
        end
    end
    
    
    
    % Collect matrix of MVAR model coefficients: [nNodes, nNodes*p, nSamp, numel(reiter)]
    if flgSmooth==0,        A(:,:,:,t)  = permute(xf_post, [2 1 3]);
    elseif flgSmooth==1,    A(:,:,:,t)  = permute(xks,     [2 1 3]);
    end
    
end
%==========================================================================


% Collect output variables:
AR       = reshape(mean(A,4), [nNodes nNodes p nSamp]);                     % Matrix of MVAR model coefficients
obsN_cov = mean(obsN_cov, 4);                                               % Observation noise covariance matrix (time-varying)



