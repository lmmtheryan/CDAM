close all
clear all
clc

% Global parameter

param_g.P = 12; %number of patterns


% parameters for global self-resemblance

param_sal.P = 3; % LARK window size
param_sal.alpha = 0.42; % LARK sensitivity parameter
param_sal.h = 0.2; % smoothing parameter for LARK
param_sal.L = 7; % # of LARK in the feature matrix
param_sal.N = inf; % size of a center + surrounding region for computing self-resemblance
param_sal.sigma = 0.07; % fall-off parameter for self-resemblamnce. **For visual purpose, use 0.2 instead of 0.06**
param_sal.omega = 0;
param_sal.th=0.6;
paraM_sal.filter_th=0.1;

param_GC.K=12;
param_GC.G=50;
param_GC.beta=0.1;
param_GC.diffThreshold = 0.001;
param_GC.maxIterations = 100;

patterns=zeros(3*300*400,param_sal.P);
ObjLoc=cell(param_sal.P);
%% Compute Saliency Weight Matrix using the given patterns

tic
for k = 1:param_g.P
    param_GC.K=12;
    FN = ['./samples/t' num2str(k) '.jpg'];
    RGB = imread(FN);
    patterns(:,k)=RGB(:);
    %     pattern(:,k)=RGB(:);
    S1 = SalWeight(RGB,[64 64],param_sal); % Resize input images to [64 64]
    %% Plot saliency maps
    S2 = imresize(mat2gray(S1),[size(RGB,1), size(RGB,2)],'bilinear');
    Smax=max(max(S2));
    S3=(S2+param_sal.omega)/(Smax+param_sal.omega);
%     S3=S2;
%     figure(1)
%     subplot(3,4,k)
%     sc(cat(3,S3,double(RGB(:,:,1))),'prob_jet');
    S4=S3;
    for i=1:size(S4,1)
        for j=1:size(S4,2)
            if S4(i,j)<param_sal.th
                S4(i,j)=0;
            end
        end
    end
%     figure(2)
%     subplot(3,4,k)
%     sc(cat(3,S4,double(RGB(:,:,1))),'prob_jet');
    BW=im2bw(S3,param_sal.th);
    area_sum=sum(sum(BW));
    filter_th=area_sum*paraM_sal.filter_th;
    BW2=uint8(bwareafilt(BW,[filter_th area_sum]));
%     figure(3)
%     subplot(3,4,k)
%     sc(cat(3,BW2,double(RGB(:,:,1))),'prob_jet');
    fixedBG=1-BW2;
    %% Draw a convex hull that wraps the salient region
    [r c v]=find(fixedBG==0);
    ConH=convhull(r,c);
    fixedBG=ones(size(fixedBG));
    [xx,yy]=ndgrid(1:size(fixedBG,1),1:size(fixedBG,2));
    in=inpolygon(xx,yy,r(ConH),c(ConH));
    fixedBG(in)=0;
%     figure(4)
%     subplot(3,4,k)
%     subplot(3,4,k)
%     sc(cat(3,fixedBG,double(RGB(:,:,1))),'prob_jet');
    
    RGB=double(RGB);
    L = GCAlgo( RGB, fixedBG, param_GC);
    L = double(1 - L);
    SalObj = RGB.*repmat(L , [1 1 3]);
%     figure(5)
%     subplot(3,4,k)
%     imshow(uint8(SalObj))
    
    %% second round Grabcut
    for j=1:5
        BW=im2bw(SalObj,0);
%         area_sum=sum(sum(BW));
%         filter_th=area_sum*0.01;
%         BW=bwareafilt(BW,[filter_th area_sum]);
        figure(6)
        subplot(2,6,k)
        imshow(BW)
        CC = bwconncomp(BW);
        L=zeros(size(L));
        for i=1:CC.NumObjects
            param_GC.K=2;
            SubObj=ones(size(fixedBG));
            SubObj(CC.PixelIdxList{i})=0;
            SubL = GCAlgo( RGB, SubObj, param_GC);
            SubL = double(1 - SubL);
            L=L+SubL;
        end
        SalObj = RGB.*repmat(L , [1 1 3]);
    end
    ObjLoc{k}=L;
    mask=repmat(L , [1 1 3]);
    %     ObjMask(k,:)=mask(:);
%     figure(7)
%     subplot(2,6,k)
%     imshow(uint8(SalObj))
%     FN=['subjects',num2str(k),'.fig'];
%     figure
%     imshow(uint8(SalObj))
%     savefig(FN)
end
toc
% %% RNN
% T=0.002;
% Tmax=5;
% K1=10000;
% K2=2000;
% RGBsize=size(RGB);
% param_U.sigma=1000;
% param_U.sigma2=1000;
%
% w=1.5;
% fs=30;
%
% for i=1:1
% pattern_num=3;
% x=input(:,pattern_num);
% param_U.K=K1/ norm(dUa_func(x,pattern,S,param_g,param_U));
% h=zeros(size(x));
% err_norm1=zeros(Tmax/T+1,1);
% err_norm2=zeros(Tmax/T+1,1);
%
% for t=0:Tmax/T
%     t*T
%     err_norm1(t+1,1)=(norm(x-pattern(:,pattern_num))/norm(pattern(:,pattern_num)))^2;
% %     W=(x-h-param_U.K.*dUa_func( x,pattern,S,param_g,param_U))*pinv(threshold(x,0));
% %     x=x+T*dotx_func(x,W,h);
%     x=x-T*(-x+h+x-h-param_U.K*dUa_func(x,pattern,S,param_g,param_U));
%     param_U.K=K1/ norm(dUa_func(x,pattern,S,param_g,param_U));
% end
% err_norm1(Tmax/T+1,1)=(norm(x-pattern(:,pattern_num))/norm(pattern(:,pattern_num)))^2;
%
% figure
% imshow(reshape(uint8(x),size(RGB)));
%
% t=0:T:Tmax;
% figure
% plot(t,err_norm1','LineWidth',w)
% xlabel('time (s)','fontsize',fs)
% legend('NMSE','Location','East')
% set(gca,'LineWidth',w,'fontsize',fs)
% end
%









