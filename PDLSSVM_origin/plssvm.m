function [w,z,v] = plssvm(X,Y,rho,c1,c2)
% linear_plssvm  Solve linear primal least square support vector machine (LSSVM) via ADMM
%
% Useage: [w,z,v] = plssvm(X,Y,rho,c1,c2)
%
% where rho,c1,c2 are non-negative parameters, and rho should be biger
% than 1; others are regular parameters.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_start = tic;
    % Global constants and defaults
    QUIET    = 0;
    MAX_ITER = 200;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    % Data preprocessing
    [m, n] = size(X);
    % alpha=rand(m,1);
    % v=rand(n,1);
    % u1=rand(n,1);
    % u2=rand(n,1);
    % u3=rand(m,1);
    u=zeros(n,1);
    v=zeros(n,1);
    w=zeros(n+1,1);
    z=rand(n,1);
    Iw=eye(n);
    ew=ones(n,1);
    e=ones(m,1);
    eps1=0.001;

    Y=diag(Y);
    H=[(1+rho)*Iw+c2*X'*X,c2*X'*e;e'*X,m];
    H=inv(H);
    t=0;
    while(t<=MAX_ITER) && norm(w(1:end-1)-z,2)>=eps1  
    %     temp=c2*X'*Y*e+rho*z-rho*u;
    %     temp1=e'*Y*e;
        d=[c2*X'*Y*e+rho*z-rho*u;e'*Y*e];
        w=H*d;
        z= shrinkage(w(1:n)+u,c1/rho*ew);
        v=v+(w(1:n)-z); 
        t=t+1;
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end


