function [alpha,z,v] = dlssvm(X,Y,rho,c1,c2)
% linear_dlssvm   Solve linear dual least square support vector machine (LSSVM) via ADMM
%
% [alpha,z,v] = dlssvm(X,Y,rho,c1,c2)
%
% where rho,c1,c2 are non-negative parameters, and rho should be biger
% than 1; others are regular parameters.
%
%
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
    YY=diag(Y);
    alpha=zeros(m,1);
    z=rand(m,1);
    v=zeros(m+1,1);
    eps1=0.001;

    e=ones(m,1);
    I=eye(m);
    A=[Y,I];
    B=[zeros(m,1),I];
%     H=YY*X*X'*YY'+1/c1*I;
    H=YY*X*X'*YY'+1/c1*I;
    H=inv(H+rho*A*A');
    t=0;

    while t<=MAX_ITER && norm(A'*alpha-B'*z,2)>=eps1  
%         alpha=H*(rho*A*B'*z-rho*A*v+rho*YY*e);
        alpha=H*(rho*A*B'*z-rho*A*v+rho*e);
        s=A'*alpha+v;
        s=s(2:end);
%         z=shrinkage(z,c2/rho*e);
        z=shrinkage(s,c2/rho*e);
        v=v+(A'*alpha-B'*z); 
        t=t+1;
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end


