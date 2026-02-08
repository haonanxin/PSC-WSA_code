clear;clc;close all;
addpath('data')
addpath('funs');

dataset_name='prokaryotic';
load([dataset_name,'.mat'])
num=size(X{1},1);
V=length(X);
c=length(unique(Y));

% Normalization
for i = 1 :length(X)
    X{i} = full((X{i} - mean(X{i}, 2)) ./ repmat(std(X{i}, [], 2), 1, size(X{i}, 2)));
    X{i}=X{i}';
end

%% Parameter Setting of prokaryotic with normalization     ACC = 0.77
mu=1;	 beta=10;	 k=12;

%% Parameter Setting of Caltech101-7 with normalization       ACC = 0.82
% mu=100;	 beta=100;	 k=7;

%% Parameter Setting of NTU2012_mvcnn_gvcnn with normalization0        ACC = 0.76
% mu=100;	 beta=1;	 k=201;

%% Parameter Setting of Wiki_textimage with normalization     ACC = 0.61
% mu=0.01;	 beta=1;	 k=30;

%% Parameter Setting of CiteSeer with normalization               ACC = 0.57
% mu=100;	 beta=0.1;	 k=6;

%% Parameter Setting of STL10 with normalization               ACC = 0.94
% mu=10;	 beta=1;	 k=30;

%% Optimization of PSC-WSA Soft Weight
[M,idx] = PSIG(X);
[F,obj] = PSC_WSA_Soft_Weight(X,beta,mu,k,M,c);
Y_pre=kmeans(F(idx, :),c,'Replicates',100,'MaxIter',50);
my_result = ClusteringMeasure_new(Y, Y_pre);

disp(['********************************************']);
disp(['Running PSC-WSA on ',dataset_name,' to obtain ACC: ', num2str(my_result.ACC)]);
disp(['********************************************']);

