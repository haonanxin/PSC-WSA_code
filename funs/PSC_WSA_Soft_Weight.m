function [F,obj] = PSC_WSA_Soft_Weight(X,beta,mu,k,M,c)
m=size(M,2);    % number of anchors
V=length(X);
num=size(X{1},2);

Z=cell(V,1);
Q=cell(V,1);
G=cell(V,1);
A=cell(V,1);
XXT=cell(V,1);
alpha=ones(V,1)./V;
F=zeros(m,c);
F(1:c,:) = eye(c);
temp_K=sum(M,1);
K=1./ceil(temp_K);
for i = 1 :V
    temp_Z=zeros(k,m); % m  * n
    temp_Z(:,1:k) = eye(k);
    Z{i,1}=temp_Z;
    G{i,1}=Z{i,1}*F;

    tempQ=sparse(num,m);
    X_i=X{i};
    temp_X_i=sparse(num,num);
    for j=1:m
        idx=find(M(:,j)==1);
        temp_X_i(idx,idx)=X_i(:,idx)'*X_i(:,idx);

        [~,idx1]=sort(diag(temp_X_i(idx,idx)));
        tempQ(idx(idx1(1:ceil(length(idx)))),j)=1;
    end
    XXT{i}=temp_X_i;
    Q{i,1}=tempQ;
end



for iter=1:30
    iter

    % update A^v
    XQK=cell(V,1);
    for v=1:V
        XQ=fast_cal(X{v},Q{v});
        XQK{v}=XQ.*K;
        XQKZv=XQK{v}*Z{v}';
        [Uu1,~,Vv1] = svd(XQKZv,'econ');
        A{v} = Uu1*Vv1';
    end

    % update Z^v
    for v=1:V
        temp1=2*A{v}'*XQK{v}+2*beta*G{v}*F'/alpha(v);

        Zv=zeros(k,m);
        for i=1:m
            tempZ = EProjSimplex_new(0.5*temp1(:,i)/(mu+1+beta/alpha(v)));
            Zv(:,i)=tempZ;
        end
        Z{v}=Zv;
    end
    clear XQK temp1

    % update F
    P=[];
    for v=1:V
        P=[P,Z{v}'/sqrt(alpha(v))];
    end
    [nn, ~, ~] = svd(P, 'econ');
    F=nn(:,1:c);
    clear P

    % update G^v
    for v=1:V
        G{v}=Z{v}*F;
    end

    % update alpha
    h=zeros(V,1);
    for v=1:V
        temp2=Z{v}-G{v}*F';
        h(v)=sum(temp2.^2,'all');
    end
    alpha=sqrt(h)./sum(sqrt(h));

    % update Q^v
    for v=1:V
        Rv=XXT{v};
        e=1./(temp_K.^2);
        Dv=-2*(X{v}'*A{v}*Z{v}).*K;

        tempQ=sparse(num,m);
        for i=1:m
            idx=find(M(:,i)==1);
            H=e(i)*Rv(idx,idx);
            f=Dv(idx,i);
            len_idx=length(idx);
            tempQ(idx,i) = quadprog((H+H'), f,[],[],ones(1, len_idx),len_idx,zeros(len_idx,1),len_idx*ones(len_idx,1),[], optimset('Display', 'off'));
        end
        Q{v}=tempQ;
    end
    clear tempQ



    % calculate obj
    obj1=0;
    obj2=0;
    obj3=0;
    for v=1:V
        XQ=fast_cal(X{v},Q{v});
        temp1=XQ.*K-A{v}*Z{v};
        temp2=Z{v}-G{v}*F';

        obj1=obj1+sum(temp1.^2,'all');
        obj2=obj2+sum(temp2.^2,'all')/alpha(v);
        obj3=obj3+sum(Z{v}.^2,'all');
    end

    obj(iter)=obj1+beta*obj2+mu*obj3;
    if iter>3&&abs((obj(iter)-obj(iter-1))/obj(iter-1))<10^(-4)
        break
    end
end

end