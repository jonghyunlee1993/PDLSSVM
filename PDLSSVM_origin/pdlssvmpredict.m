function [predictY,sparseprimal,sparsedual] = pdlssvmpredict(X,Y,testX,w,alpha,v,beta,sign1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predictY=[];
[m,n]=size(testX);
Y=diag(Y);

if (nargin <7 | nargin>8) % check correct number of arguments
    help pdlssvm
else
    if (nargin<8) sign1=1;, end
if sign1~=1
    if w==0
        predictY=sign(testX*v);  %testX������������ֱ�߾���
    else
        predictY=sign(testX*w);  %testX������������ֱ�߾���
    end
else
    if beta==0
        w=alpha'*Y*X;
        predictY=sign(testX*w');  %testX������������ֱ�߾���
    else
        w=beta'*Y*X;
        predictY=sign(testX*w');  %testX������������ֱ�߾���
    end
end
sparseprimal=sum(w==0);
sparsedual=sum(alpha==0);
end
end
