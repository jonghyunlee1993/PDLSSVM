function [w,alpha,z,beta] = pdlssvm(X,Y,rho,c,c1,c2)
% linear_pdlssvm   Solve linear primal and dual least square support vector machine via ADMM
%
% Useage:[w,alpha,v,beta] = pdlssvm(X,Y,rho,c,c1,c2)
%
% Solves the following problem via ADMM:
%
%   minimize   sparse primal LSSVM + sparse dual LSSVM
%   s.t.       w=alpha*X*Y
%
% where rho,c,c1,c2 are non-negative parameters, and rho should be biger
% than 1; others are regular parameters.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_start = tic;
    % Global constants and defaults
    MAX_ITER = 500;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    % Data preprocessing
    [m, n] = size(X);
    e=ones(m,1);
        % alpha=rand(m,1);
    % v=rand(n,1);
    % u1=rand(n,1);
    % u2=rand(n,1);
    % u3=rand(m,1);
    alpha=rand(m,1);
    beta=rand(m,1);
    z=rand(n,1);
    w=zeros(n,1);
    u1=zeros(n,1);
    u2=zeros(m,1);
    u3=zeros(n,1);
    t=0;
    eps1=0.001;
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
 
    Y=diag(Y);
%     H=Y'*(X*X')*Y;
    H=X*X';
    B=X'*Y;
    I=eye(m);
    Iw=eye(n);
    Hw=B*B';
    ew=ones(n,1);
    temp=(1+2*rho)*Iw+c*Hw;
    temp=inv(temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%     temp1=inv((1+rho)*H+(1/c+rho)*I);
    temp1=inv(H+rho*(B'*B)+(1/c+rho)*I);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    while(t<=MAX_ITER) && max([norm(z-B*beta,2),norm(w-z,2),norm(beta-alpha)])>=eps1  
        theta=1/2*(z-u1+B*beta-u3);
        w = shrinkage(c1/(2*rho)*ew,theta);
        z=temp*(c*B*e+rho*w+rho*u1+rho*B*beta-rho*u3);
%         z=temp*(c*B*e+rho*w+rho*u2);
        alpha=shrinkage(c2/rho*e,beta-u2);
        beta=temp1*rho*(alpha+u2+B'*z+B'*u3+1/rho*e);
%         beta=shrinkage(c2/rho*e,alpha-u3+1/rho*Y*e);
%         beta=shrinkage(c2/rho*e,alpha-u3+1/rho*e);
         u1=u1+(w-z);
         u2=u2+(alpha-beta);  
         u3=u3+(z-B*beta);
             
%         w=temp*(c*B*e+rho*v+rho*u2-u1-u2);
%         v = shrinkage(c1/rho*ew,w-1/rho*u2);
%         alpha=temp1*(e+B'*u1-u3+rho*B'*w);
% %         beta=shrinkage(c2/rho*e,alpha-1/rho*u3);
%         beta=shrinkage(c2/rho*e,alpha-u3+1/rho*Y*e);
%         % z-update with relaxation
%         u1=u1+(w-B*alpha);
%         u2=u2+(w-v);
%         u3=u3+(alpha-beta);  
        t=t+1;
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end
