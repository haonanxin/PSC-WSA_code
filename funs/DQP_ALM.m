function [learn_x] = DQP_ALM(H,f,y)

num=length(f);
t=sum(y);

rho = 1.1;
mu = 0.01;
Lambda = zeros(num,1);

NITET = 100;

for iter = 1:NITET

    % Update x
    z1=H*y+f-0.5*mu*y+0.5*Lambda;
    m = EProjSimplex_new(-z1/(mu*t));
    x=m*t;

    % Update y
    z2=H*x-0.5*mu*x-0.5*Lambda;
    idx=find(-z2/mu>=0.5);
    y=zeros(num,1);
    y(idx)=1;
%     if isempty(idx)
%         [val,idx1]=sort(-z2,'descend');
%         y(idx1(1:t))=1;
%     else
%         y(idx)=1;
%     end

    % Update Lambda
    h = x-y;
    Lambda = Lambda+mu*h;

    % Update mu
    mu = rho*mu;

    %     obj(iter) = norm(h,'fro');
    obj(iter) = x'*H*x+x'*f;

    if iter>=2 && abs((obj(iter)-obj(iter-1))/obj(iter))<10^(-5)  % mu = rho^iter*mu
        break;
    elseif mu > 100
        break;
    else
        continue;
    end
%     if  mu > 10
%         break;
%     else
%         continue;
%     end
end

learn_x=zeros(num,1);
[val,idx1]=sort(x,'descend' );
learn_x(idx1(1:t))=1;

% figure(1)
% plot(1:length(obj),obj);

end