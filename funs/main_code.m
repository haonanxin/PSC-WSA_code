function [F,loss,theta,time] = main_code(X,L,B,c,F,t)
num=size(X{1},1);
V=length(X);
U=cell(V,L-1);
R=cell(V,L-1);
BD=cell(V,L-1);  % B*D^{-0.5}
BDB=cell(V,L-1);  % B*D^{-0.5}
for v=1:V
    for l=1:L-1
        BD{v,l}=B{v,l}*diag(1./sqrt(sum(B{v,l},1)));
        BDB{v,l}=BD{v,l}*BD{v,l}';
        [uu, ~, ~] = svd(BD{v,l});
        U{v,l}=uu(:,1:c);
        R{v,l}=eye(c);
    end
end

tic;
for iter=1:20
    iter
    % update M
    f_vlk=zeros(V, L-1, c);
    for v=1:V
        for l=1:L-1
            f_vlk(v,l,:)=sum(F .* (BDB{v,l} * F),1)./sum(F,1);
        end
    end
    f_vlk_reshape=reshape(f_vlk,[],1);
    M_vec=zeros(size(f_vlk_reshape));
    for k=1:c
        [val1,~]=max(f_vlk(:,:,k),[],'all');
        idx1=find(f_vlk_reshape==val1);
        M_vec(idx1)=1;
        f_vlk_reshape(idx1)=10^5;
    end
    [val2,idx2]=sort(f_vlk_reshape,'descend');
    M_vec(idx2(c+1:t))=1;
    M=reshape(M_vec,size(f_vlk));


    % update theta
    Mfvlk=M.*f_vlk;
    theta=Mfvlk/sqrt(sum(Mfvlk.^2,"all"));

    % update R
    for v=1:V
        for l=1:L-1
            temp=diag(1./sqrt(sum(F,1)))*F'*U{v,l};
            [uuu, ~, vvv] = svd(temp);
            R{v,l}=uuu*vvv';
        end
    end

    % update Y
    Z=cell(1,c);
    for k=1:c
        Bgamma=cell(V,L-1);
        for v=1:V
            for l=1:L-1
                gamma=sqrt(M(v,l,k)*theta(v,l,k)*(1./sum(B{v,l})));
                Bgamma{v,l}=gamma.*B{v,l};
            end
        end
        Bgamma_reshaped = reshape(Bgamma', 1, []);
        Z{1,k} = cell2mat(Bgamma_reshaped);
        clear Bgamma
    end

    lambda1 = -1;
    lambda2 = 0;
    lambda=[];
    count=1;
    while abs(lambda2 - lambda1) > 1e-5
        obj1=0;
        obj2=0;
        YYY=F*diag(1./sqrt(sum(F,1)));
        G=0;
        alpha=zeros(V,L-1);
        for k=1:c
            yZ_k=F(:,k)'*Z{1,k};
            obj1=obj1+sum(yZ_k.^2)/sum(F(:,k));
        end
        for v=1:V
            for l=1:L-1
                temp3=norm(U{v,l} - YYY * R{v,l}, 'fro');
                alpha(v,l) = 0.5 / temp3;
                obj2 = obj2 +temp3;
            end
        end
        lambda1 = lambda2;
        lambda2 = obj1/obj2;
        lambda=[lambda,lambda2];
        for v=1:V
            for l=1:L-1
                G = G+2*lambda2*alpha(v,l)*U{v,l}*R{v,l}';
            end
        end
        F = Hierarchical_coordinate(F, Z, G, c);
        count = count+1;
        if count > 10
            break;
        end
    end
    loss(iter) = obtain_obj(F, Z, U, R, c, V, L);
     if iter > 2 && (loss(iter) - loss(iter - 1)) / loss(iter - 1) < 1e-5
        break
    end
end
time=toc;

end


function obj = obtain_obj(Y, Z, U, R, c, V, L)
obj1=0;
obj2=0;
YYY=Y*diag(1./sqrt(sum(Y,1)));
for k=1:c
    yZ_k=Y(:,k)'*Z{1,k};
    obj1=obj1+yZ_k*yZ_k'/sum(Y(:,k));
end
for v=1:V
    for l=1:L-1
        temp3=norm(U{v,l} - YYY * R{v,l}, 'fro');
        obj2 = obj2 +temp3;
    end
end
obj = obj1 / obj2;
end

