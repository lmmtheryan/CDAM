close all
param_g.NI = 12; %number of noised inputs
param_g.P = 12;

testPattern=2;

target=cell(param_g.P);
ObjLocNum=zeros(param_g.P,1);
Locations=cell(param_g.P,1);

r=-10;
s=1.1;

for k = 1:param_g.P
    FN = ['./samples/t' num2str(k) '.jpg'];
    target{k} = double(imread(FN));
    BW=im2bw(ObjLoc{k},0);
    area_sum=sum(sum(BW));
    filter_th=area_sum*paraM_sal.filter_th;
    BW=bwareafilt(BW,[filter_th area_sum]);
    CC = bwconncomp(BW);
    ObjLocNum(k)=CC.NumObjects;
    Locations{k}=cell(ObjLocNum(k));
    for i=1:ObjLocNum(k)
        loc=zeros(size(BW));
        loc(CC.PixelIdxList{i})=1;
        %         figure
        %         imshow(im2bw(loc,0))
        Locations{k}{i}=loc;
    end
end


% for k=1:param_g.P
%     resp_index{k}=zeros(2,1);
% end


sz=size(target{1});
border=zeros(sz(1),sz(2));
border(1,:)=1;
border(:,1)=1;
border(sz(1),:)=1;
border(:,sz(2))=1;
NMSE_CDAM=zeros(9,1);
SR_CDAM=zeros(9,1);

for InputNum=testPattern:testPattern
    i=InputNum;
    %% RNN parameters
    Imax=3;
    Response=zeros(Imax+1,1);
    resp_index=cell(param_g.P,1);
    sigma=5;
    resp_val=zeros(param_g.P,1);
    
    for pr=0:0.1:0.8
        NMSE=0;
        %       i=9;
        FN = ['./samples/i' num2str(i) '.jpg'];
        
        im = imread(FN);
        %         im = imnoise(im,'salt & pepper',pr);
        im = imnoise(im,'gaussian',0,pr);
        X = double(im);
        sum_target=sum(sum(sum(target{i}.*target{i})));
        for t=1:Imax
            for k=1:param_g.P
                resp_val(k,1)=1;
                all_index=zeros(ObjLocNum(k),2);
                for O=1:ObjLocNum(k)
                    % y=kernel_filt(tar.*cos_window,double(input).*cos_window,Locations{k}{1});
                    y=real(kernel_filt(target{k},X,Locations{k}{O}));
                    
                    %% analyze the response
                    resp=y/sum(sum(Locations{k}{O}));
                    resp=exp(-sqrt(resp)/sigma);
                    %                 max(resp(:))
                    %                 figure(6)
                    %                 imagesc(resp)
                    %                 resp=exp(-resp/sigma);
                    %                 y_mask=Resp_Filt(resp,1);
                    %                 resp=resp.*y_mask/max(y_mask(:));
                    %                 max(resp(:))
                    %                 figure(7)
                    %                 imagesc(resp)
                    resp=resp.*(resp/sum(sum(resp))).^3;
                    %                 figure(8)
                    %                 imagesc(resp)
                    [maxval,ind] = max(resp(:));
                    [I J] = ind2sub([size(resp,1) size(resp,2)],ind);
                    all_index(O,:)=[I J];
                    %                 resp_index{k}=[I;J];
                    %                 [I J];
                    resp_val(k,1)= resp_val(k,1)*maxval;
                end
                %% Check if overlap between objects or border happens
                %                 if ObjLocNum(k)>1
                %                     check_obj_ovlp=circshift(Locations{k}{1},all_index(1,:)-[1,1]);
                %                     for O=2:ObjLocNum(k)
                %                         check_obj_ovlp=check_obj_ovlp.*circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                %                     end
                %                     if sum(sum(check_obj_ovlp))>10
                %                         resp_val(k,1)=0;
                %                     end
                %                 end
                for O=1:ObjLocNum(k)
                    shifted_sub=circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                    ovlp=shifted_sub(1,:)*shifted_sub(sz(1),:)'+shifted_sub(:,1)'*shifted_sub(:,sz(2));
                    if ovlp>10;
                        resp_val(k,1)=0;
                    end
                end
                resp_val(k,1)=resp_val(k,1)^(1/ObjLocNum(k));
                %     resp_val(k,1)=minval/maxval;
            end
            % resp_val
            K=1/sum(resp_val);
            dX=0;
            [maxval,ind]=max(K.*resp_val);
            %         for k=1:param_g.P
            %             %     gain=exp();
            % %             I=resp_index{k}(1)-1;
            % %             J=resp_index{k}(2)-1;
            %             %     resp=exp(-resp_val(k,1)/1000)*(double(circshift(target{k},[I,J,0]))-X);
            %             %             X=X+norm(resp(:,:,1)+resp(:,:,2)+resp(:,:,3))*resp;
            %             %             X=X+K*resp_val(k,1)*(double(circshift(target{k},[I,J,0]))-X);
            %             dX=dX+K*resp_val(k,1)*target{k};
            %             %     resp=exp(-resp_val(k,1)/1000);
            %             %     X=X+K/(resp+lambda)*resp*(double(circshift(target{k},[I,J,0]))-X);
            %         end
            %         X=dX;
            X=maxval*target{ind};
            %     for c=1:size(X_norm,3)
            %         X_norm(t,c)=norm(double(X(:,:,c)));
            %     end
            Response(t+1,1)=resp_val(i);
        end
        NMSE=sum(sum(sum((X-target{i}).*(X-target{i}))))/sum_target;
        NMSE_CDAM(uint8(pr/0.1+1))=NMSE_CDAM(uint8(pr/0.1+1))+NMSE;
        if NMSE<1e-4
            SR_CDAM(uint8(pr/0.1+1))=SR_CDAM(uint8(pr/0.1+1))+1;
        end
%         if NMSE>1e-4 && pr==0
%             InputNum
%         end
    end
    
    
end
NMSE_CDAM=NMSE_CDAM/param_g.NI;
SR_CDAM=SR_CDAM/param_g.NI;

figure(1)
Pr=0:0.1:0.8;
Pr=Pr';

w=3;
fs=60;

plot(Pr,NMSE_CDAM,'.-b','LineWidth',w)
hold on

figure(2)

plot(Pr,SR_CDAM,'.-b','LineWidth',w)
hold on


disp('r')

NMSE_CDAM_r=zeros(9,1);
SR_CDAM_r=zeros(9,1);

for InputNum=testPattern:testPattern
    i=InputNum;
    %% RNN parameters
    Imax=3;
    Response=zeros(Imax+1,1);
    resp_index=cell(param_g.P,1);
    sigma=5;
    resp_val=zeros(param_g.P,1);
    
    for pr=0:0.1:0.8
        %       i=9;
        FN = ['./samples/i' num2str(i) '.jpg'];
        im=imrotate(imread(FN),r,'bilinear','crop');
        %         im = imnoise(im,'salt & pepper',pr);
        im = imnoise(im,'gaussian',0,pr);
        X = double(im);
        sum_target=sum(sum(sum(target{i}.*target{i})));
        for t=1:Imax
            for k=1:param_g.P
                resp_val(k,1)=1;
                all_index=zeros(ObjLocNum(k),2);
                for O=1:ObjLocNum(k)
                    % y=kernel_filt(tar.*cos_window,double(input).*cos_window,Locations{k}{1});
                    y=real(kernel_filt(target{k},X,Locations{k}{O}));
                    
                    %% analyze the response
                    resp=y/sum(sum(Locations{k}{O}));
                    resp=exp(-sqrt(resp)/sigma);
                    %                 max(resp(:))
                    %                 figure(6)
                    %                 imagesc(resp)
                    %                 resp=exp(-resp/sigma);
                    %                 y_mask=Resp_Filt(resp,1);
                    %                 resp=resp.*y_mask/max(y_mask(:));
                    %                 max(resp(:))
                    %                 figure(7)
                    %                 imagesc(resp)
                    resp=resp.*(resp/sum(sum(resp))).^3;
                    %                 figure(8)
                    %                 imagesc(resp)
                    [maxval,ind] = max(resp(:));
                    [I J] = ind2sub([size(resp,1) size(resp,2)],ind);
                    all_index(O,:)=[I J];
                    %                 resp_index{k}=[I;J];
                    %                 [I J];
                    resp_val(k,1)= resp_val(k,1)*maxval;
                end
                %% Check if overlap between objects or border happens
                %                 if ObjLocNum(k)>1
                %                     check_obj_ovlp=circshift(Locations{k}{1},all_index(1,:)-[1,1]);
                %                     for O=2:ObjLocNum(k)
                %                         check_obj_ovlp=check_obj_ovlp.*circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                %                     end
                %                     if sum(sum(check_obj_ovlp))>10
                %                         resp_val(k,1)=0;
                %                     end
                %                 end
                for O=1:ObjLocNum(k)
                    shifted_sub=circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                    ovlp=shifted_sub(1,:)*shifted_sub(sz(1),:)'+shifted_sub(:,1)'*shifted_sub(:,sz(2));
                    if ovlp>10;
                        resp_val(k,1)=0;
                    end
                end
                resp_val(k,1)=resp_val(k,1)^(1/ObjLocNum(k));
                %     resp_val(k,1)=minval/maxval;
            end
            % resp_val
            K=1/sum(resp_val);
            dX=0;
            [maxval,ind]=max(K.*resp_val);
            %         for k=1:param_g.P
            %             %     gain=exp();
            % %             I=resp_index{k}(1)-1;
            % %             J=resp_index{k}(2)-1;
            %             %     resp=exp(-resp_val(k,1)/1000)*(double(circshift(target{k},[I,J,0]))-X);
            %             %             X=X+norm(resp(:,:,1)+resp(:,:,2)+resp(:,:,3))*resp;
            %             %             X=X+K*resp_val(k,1)*(double(circshift(target{k},[I,J,0]))-X);
            %             dX=dX+K*resp_val(k,1)*target{k};
            %             %     resp=exp(-resp_val(k,1)/1000);
            %             %     X=X+K/(resp+lambda)*resp*(double(circshift(target{k},[I,J,0]))-X);
            %         end
            %         X=dX;
            X=maxval*target{ind};
            %     for c=1:size(X_norm,3)
            %         X_norm(t,c)=norm(double(X(:,:,c)));
            %     end
            Response(t+1,1)=resp_val(i);
        end
        NMSE=sum(sum(sum((X-target{i}).*(X-target{i}))))/sum_target;
        NMSE_CDAM_r(uint8(pr/0.1+1))=NMSE_CDAM_r(uint8(pr/0.1+1))+NMSE;
        if NMSE<1e-4
            SR_CDAM_r(uint8(pr/0.1+1))= SR_CDAM_r(uint8(pr/0.1+1))+1;
        end
    end
    
    
end
NMSE_CDAM_r=NMSE_CDAM_r/param_g.NI;
SR_CDAM_r=SR_CDAM_r/param_g.NI;

figure(1)
Pr=0:0.1:0.8;
Pr=Pr';

plot(Pr,NMSE_CDAM_r,'.-r','LineWidth',w)
% LEG = findobj(AX,'type','text');
% set(LEG,'FontSize',60)
figure(2)

plot(Pr,SR_CDAM_r,'.-r','LineWidth',w)


disp('s')

NMSE_CDAM_s=zeros(9,1);
SR_CDAM_s=zeros(9,1);

for InputNum=testPattern:testPattern
    i=InputNum;
    %% RNN parameters
    Imax=3;
    Response=zeros(Imax+1,1);
    resp_index=cell(param_g.P,1);
    sigma=5;
    resp_val=zeros(param_g.P,1);
    
    for pr=0:0.1:0.8
        
        %       i=9;
        FN = ['./samples/i' num2str(i) '.jpg'];
        im=imread(FN);
        T = [s  0  0;
            0  s  0;
            0  0 1];
        t_proj = projective2d(T);
        im = imwarp(im,t_proj,'FillValues',255);
        %         im = imnoise(im,'salt & pepper',pr);
        imagesize = size(im);
        if s<1
            imagesize = size(im);
            im = padarray(im,[(300-imagesize(1))/2 (400-imagesize(2))/2],255,'both');
        else
            imagesize = size(im);
            im=im(uint32(imagesize(1)/2-149):uint32(imagesize(1)/2+150),uint32(imagesize(2)/2-199):uint32(imagesize(2)/2+200),:);
        end
        im = imnoise(im,'gaussian',0,pr);
        X = double(im);
        sum_target=sum(sum(sum(target{i}.*target{i})));
        for t=1:Imax
            for k=1:param_g.P
                resp_val(k,1)=1;
                all_index=zeros(ObjLocNum(k),2);
                for O=1:ObjLocNum(k)
                    % y=kernel_filt(tar.*cos_window,double(input).*cos_window,Locations{k}{1});
                    y=real(kernel_filt(target{k},X,Locations{k}{O}));
                    
                    %% analyze the response
                    resp=y/sum(sum(Locations{k}{O}));
                    resp=exp(-sqrt(resp)/sigma);
                    %                 max(resp(:))
                    %                 figure(6)
                    %                 imagesc(resp)
                    %                 resp=exp(-resp/sigma);
                    %                 y_mask=Resp_Filt(resp,1);
                    %                 resp=resp.*y_mask/max(y_mask(:));
                    %                 max(resp(:))
                    %                 figure(7)
                    %                 imagesc(resp)
                    resp=resp.*(resp/sum(sum(resp))).^3;
                    %                 figure(8)
                    %                 imagesc(resp)
                    [maxval,ind] = max(resp(:));
                    [I J] = ind2sub([size(resp,1) size(resp,2)],ind);
                    all_index(O,:)=[I J];
                    %                 resp_index{k}=[I;J];
                    %                 [I J];
                    resp_val(k,1)= resp_val(k,1)*maxval;
                end
                %% Check if overlap between objects or border happens
                %                 if ObjLocNum(k)>1
                %                     check_obj_ovlp=circshift(Locations{k}{1},all_index(1,:)-[1,1]);
                %                     for O=2:ObjLocNum(k)
                %                         check_obj_ovlp=check_obj_ovlp.*circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                %                     end
                %                     if sum(sum(check_obj_ovlp))>10
                %                         resp_val(k,1)=0;
                %                     end
                %                 end
                for O=1:ObjLocNum(k)
                    shifted_sub=circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                    ovlp=shifted_sub(1,:)*shifted_sub(sz(1),:)'+shifted_sub(:,1)'*shifted_sub(:,sz(2));
                    if ovlp>10;
                        resp_val(k,1)=0;
                    end
                end
                resp_val(k,1)=resp_val(k,1)^(1/ObjLocNum(k));
                %     resp_val(k,1)=minval/maxval;
            end
            % resp_val
            K=1/sum(resp_val);
            dX=0;
            [maxval,ind]=max(K.*resp_val);
            %         for k=1:param_g.P
            %             %     gain=exp();
            % %             I=resp_index{k}(1)-1;
            % %             J=resp_index{k}(2)-1;
            %             %     resp=exp(-resp_val(k,1)/1000)*(double(circshift(target{k},[I,J,0]))-X);
            %             %             X=X+norm(resp(:,:,1)+resp(:,:,2)+resp(:,:,3))*resp;
            %             %             X=X+K*resp_val(k,1)*(double(circshift(target{k},[I,J,0]))-X);
            %             dX=dX+K*resp_val(k,1)*target{k};
            %             %     resp=exp(-resp_val(k,1)/1000);
            %             %     X=X+K/(resp+lambda)*resp*(double(circshift(target{k},[I,J,0]))-X);
            %         end
            %         X=dX;
            X=maxval*target{ind};
            %     for c=1:size(X_norm,3)
            %         X_norm(t,c)=norm(double(X(:,:,c)));
            %     end
            Response(t+1,1)=resp_val(i);
        end
        NMSE=sum(sum(sum((X-target{i}).*(X-target{i}))))/sum_target;
        NMSE_CDAM_s(uint8(pr/0.1+1))=NMSE_CDAM_s(uint8(pr/0.1+1))+NMSE;
        if NMSE<1e-4
            SR_CDAM_s(uint8(pr/0.1+1))= SR_CDAM_s(uint8(pr/0.1+1))+1;
        end
    end
    
    
end
NMSE_CDAM_s=NMSE_CDAM_s/param_g.NI;
SR_CDAM_s=SR_CDAM_s/param_g.NI;

figure(1)
Pr=0:0.1:0.8;
Pr=Pr';

plot(Pr,NMSE_CDAM_s,'-g','LineWidth',w)
% LEG = findobj(AX,'type','text');
% set(LEG,'FontSize',60)
figure(2)

plot(Pr,SR_CDAM_s,'-g','LineWidth',w)

disp('p')
NMSE_CDAM_p=zeros(9,1);
SR_CDAM_p=zeros(9,1);

for InputNum=testPattern:testPattern
    
    i=InputNum;
    %% RNN parameters
    
    Imax=3;
    Response=zeros(Imax+1,1);
    NMSE=zeros(Imax+1,1);
    resp_index=cell(param_g.P,1);
    sigma=5;
    resp_val=zeros(param_g.P,1);
    
    for pr=0:0.1:0.8
        
        %       i=9;
        %         FN = ['./samples/i' num2str(i) '.jpg'];
        FN = ['./samples/i' num2str(i) '.jpg'];
        %         im=imrotate(imread(FN),r,'bilinear','crop');
        T = [1  0  -0.0003;
            0  1  -0.0003;
            -0.0003  0 1   ];
        %             T = [1  0  -0.0005;
        %                 0  1  -0.0005;
        %                 -0.0005  0 1   ];
        %   T = [1  0  0.001;
        %         0  1  0.001;
        %         0.001  0 1   ];
        t_proj = projective2d(T);
        im = imwarp(im,t_proj,'FillValues',255);
        im = imresize(im,[300,400]);
        %         im = imnoise(im,'salt & pepper',pr);
        im = imnoise(im,'gaussian',0,pr);
        X = double(im);
        sum_target=sum(sum(sum(target{i}.*target{i})));
        for t=1:Imax
            for k=1:param_g.P
                resp_val(k,1)=1;
                all_index=zeros(ObjLocNum(k),2);
                for O=1:ObjLocNum(k)
                    % y=kernel_filt(tar.*cos_window,double(input).*cos_window,Locations{k}{1});
                    y=real(kernel_filt(target{k},X,Locations{k}{O}));
                    
                    %% analyze the response
                    resp=y/sum(sum(Locations{k}{O}));
                    resp=exp(-sqrt(resp)/sigma);
                    %                 max(resp(:))
                    %                 figure(6)
                    %                 imagesc(resp)
                    %                 resp=exp(-resp/sigma);
                    %                 y_mask=Resp_Filt(resp,1);
                    %                 resp=resp.*y_mask/max(y_mask(:));
                    %                 max(resp(:))
                    %                 figure(7)
                    %                 imagesc(resp)
                    resp=resp.*(resp/sum(sum(resp))).^3;
                    %                 figure(8)
                    %                 imagesc(resp)
                    [maxval,ind] = max(resp(:));
                    [I J] = ind2sub([size(resp,1) size(resp,2)],ind);
                    all_index(O,:)=[I J];
                    %                 resp_index{k}=[I;J];
                    %                 [I J];
                    resp_val(k,1)= resp_val(k,1)*maxval;
                end
                %% Check if overlap between objects or border happens
                %                 if ObjLocNum(k)>1
                %                     check_obj_ovlp=circshift(Locations{k}{1},all_index(1,:)-[1,1]);
                %                     for O=2:ObjLocNum(k)
                %                         check_obj_ovlp=check_obj_ovlp.*circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                %                     end
                %                     if sum(sum(check_obj_ovlp))>10
                %                         resp_val(k,1)=0;
                %                     end
                %                 end
                for O=1:ObjLocNum(k)
                    shifted_sub=circshift(Locations{k}{O},all_index(O,:)-[1,1]);
                    ovlp=shifted_sub(1,:)*shifted_sub(sz(1),:)'+shifted_sub(:,1)'*shifted_sub(:,sz(2));
                    if ovlp>10;
                        resp_val(k,1)=0;
                    end
                end
                resp_val(k,1)=resp_val(k,1)^(1/ObjLocNum(k));
                %     resp_val(k,1)=minval/maxval;
            end
            % resp_val
            K=1/sum(resp_val);
            dX=0;
            [maxval,ind]=max(K.*resp_val);
            %         for k=1:param_g.P
            %             %     gain=exp();
            % %             I=resp_index{k}(1)-1;
            % %             J=resp_index{k}(2)-1;
            %             %     resp=exp(-resp_val(k,1)/1000)*(double(circshift(target{k},[I,J,0]))-X);
            %             %             X=X+norm(resp(:,:,1)+resp(:,:,2)+resp(:,:,3))*resp;
            %             %             X=X+K*resp_val(k,1)*(double(circshift(target{k},[I,J,0]))-X);
            %             dX=dX+K*resp_val(k,1)*target{k};
            %             %     resp=exp(-resp_val(k,1)/1000);
            %             %     X=X+K/(resp+lambda)*resp*(double(circshift(target{k},[I,J,0]))-X);
            %         end
            %         X=dX;
            X=maxval*target{ind};
            %     for c=1:size(X_norm,3)
            %         X_norm(t,c)=norm(double(X(:,:,c)));
            %     end
            Response(t+1,1)=resp_val(i);
        end
        NMSE=sum(sum(sum((X-target{i}).*(X-target{i}))))/sum_target;
        NMSE_CDAM_p(uint8(pr/0.1+1))=NMSE_CDAM_p(uint8(pr/0.1+1))+NMSE;
        if NMSE<1e-4
            SR_CDAM_p(uint8(pr/0.1+1))=SR_CDAM_p(uint8(pr/0.1+1))+1;
        end
    end
    
    
end
NMSE_CDAM_p=NMSE_CDAM_p/param_g.NI;
SR_CDAM_p=SR_CDAM_p/param_g.NI;

figure(1)
Pr=0:0.1:0.8;
Pr=Pr';

plot(Pr,NMSE_CDAM_p,'-k','LineWidth',w)
AX=legend('CDAM',['r=',num2str(r)],['s=',num2str(s)],'T_1');
% LEG = findobj(AX,'type','text');
% set(LEG,'FontSize',60)
xlabel({'$\sigma_G$'},'Interpreter','latex','fontsize',fs)
ylabel({'$NMSE$'},'Interpreter','latex','fontsize',fs)
set(gca,'LineWidth',w,'fontsize',fs)
hold off
% FN=['all_avg_rnsn_gn_r=' num2str(r) '.fig'];
FN=['all_avg_rnsn_gn_dis.fig'];
savefig(FN)

figure(2)
Pr=0:0.1:0.8;
Pr=Pr';

plot(Pr,SR_CDAM_p,'-k','LineWidth',w)
AX=legend('CDAM',['r=',num2str(r)],['s=',num2str(s)],'T_1');
% LEG = findobj(AX,'type','text');
% set(LEG,'FontSize',60)
xlabel({'$\sigma_G$'},'Interpreter','latex','fontsize',fs)
ylabel({'$NMSE$'},'Interpreter','latex','fontsize',fs)
set(gca,'LineWidth',w,'fontsize',fs)
hold off
% FN=['all_avg_rnsn_gn_r=' num2str(r) '.fig'];
FN=['all_avg_rnsn_gn_dis_SR.fig'];
savefig(FN)

% FN=['all_rn_spn' num2str(InputNum) '.eps'];
% savefig(FN)