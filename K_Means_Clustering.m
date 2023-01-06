%% Dataset Preparation

clc;
clear all;
close all;

image_numbers=[3,7,12,12,17,24,34,87,140,140,145,171,180,193;100,92,95,100,97,92,105,89,70,84,92,88,94,88];
I=[];
Label=[];

for idx=1:14
    image_text=['CS554Dataset\plane_',num2str(image_numbers(1,idx)),'th_Subject',num2str(image_numbers(2,idx)),'.png'];
    label_text=['CS554Dataset\Labels_plane_',num2str(image_numbers(1,idx)),'th_Subject',num2str(image_numbers(2,idx)),'.png'];
    I(:,:,idx)=imread(image_text);
    Label(:,:,idx)=imread(label_text);
end
clear image_text label_text;
IMG=double(I);

%% Choosing An Image from Dataset

idx=input('Enter index of image from dataset [1-14]: ');
assert(any(idx==1:14),'Wrong index, it should be in range [1-14]!');

I=uint8(I(:,:,idx));
IMG=IMG(:,:,idx);
Label=Label(:,:,idx);

%% K - Means Clustring

k=input('Enter cluster number (k) as input: ');
init={input('Enter initialization type of clustring [uniform, or random]: ','s')};
M=10;

disp(' ');disp(' ');
disp(['Chosen Image : plane_',num2str(image_numbers(1,idx)),'th_Subject',num2str(image_numbers(2,idx))]);
disp(['K = ',num2str(k),'    Initialization : ',init{1}]);

tic;
[Seg,r,c,loop]=k_mean_seg(IMG,k,init,M);
toc;
pause(0.3);
%% Plotting the Results

FONT=17;

% FIG 1 :

figure;
subplot(1,3,1);
imshow(I,[]);
title('Original Image','fontsize',FONT);
subplot(1,3,2);
imshow(Seg,[]);
title({['K = ',num2str(k) ,' Mean Clustring'],['Initialization : ',init{1}],['Iteration : ',num2str(loop)]},'fontsize',FONT);
hold on;
X=c;Y=r;
plot(X,Y,'ro','MarkerFaceColor','r');
subplot(1,3,3);
imshow(Label,[]);
title('Ground Truth','fontsize',FONT);
sgtitle(['Chosen Image : plane ',num2str(image_numbers(1,idx)),'th Subject ',num2str(image_numbers(2,idx))],'fontsize',FONT+1);

dim=[0.1,0.1,0.1,0.1];
htext=annotation('textbox',dim,'String','Choose the regions or region containing TUMOR by clicking once on them and then click ENTER to terminate and see the region/s with tumor!','FitBoxToText','on');

% FIG 2 :
pause(0.3);
loc=[];
try
    [c_input, r_input] = ginput;
    loc = [ loc ; int32([c_input r_input])];
catch end
delete(htext);

figure;
subplot(1,2,1);
A=logical(zeros(size(Seg)));
for KK=1:size(loc,1)
    A=A|(Seg==Seg(loc(KK,2),loc(KK,1)));
end 
imshow(A,[]);
title('Segmented Tumor','fontsize',FONT);
subplot(1,2,2);
imshow(Label,[]);
title('Ground Truth','fontsize',FONT);
sgtitle(['Chosen Image : plane ',num2str(image_numbers(1,idx)),'th Subject ',num2str(image_numbers(2,idx))],'fontsize',FONT+1);

% FIG 3 :

figure;
subplot(1,2,1);
imshow(Seg,[]);
title({'k mean seq() function results','Not Built-in function'},'fontsize',FONT)
[L,Centers] = imsegkmeans(I,k);
B = labeloverlay(I,L);
subplot(1,2,2);
imshow(rgb2gray(B),[])
title({'imsegkmeans() function results','Built-in function'},'fontsize',FONT);
sgtitle('Comparison of Built-in and Non Built-in Clustering Functions','fontsize',FONT+1);


% FIG 4 :

figure;
subplot(1,2,1);
rgbImage=gray2rgb(Seg); % The function that I wrote
imshow(rgbImage,[]);
title({'Labeling Fucntion Results','Not Built-in Function'},'fontsize',FONT);
subplot(1,2,2);
rgbImage1=label2rgb(Seg); % Built - in Function
imshow(rgbImage1,[]);
title({'label2rgb() function','Built-in Function'},'fontsize',FONT);
sgtitle('Comparison of Built-in and Non Built-in Labeling Functions','fontsize',FONT+1);

%% Accuracy, Precission, Recall, and F1 Score Analysis for K - Means Clustring

M=confusionmat(logical(Label(:)),A(:));

Diagonal=diag(M);
sum_rows=sum(M,2);

Overall_Accuracy = sum(Diagonal)/sum(M(:));

Precision=Diagonal./sum_rows;
Overall_Precision=mean(Precision);

sum_col=sum(M,1);

recall=Diagonal./sum_col';
Overall_Recall=mean(recall);

F1_Score=2*((Overall_Precision*Overall_Recall)/(Overall_Precision+Overall_Recall));

disp(' ');disp(' ');disp('Accuracy, Precission, Recall, and F1 Score Analysis for K - Means Clustring :');
disp(['K = ',num2str(k),'    Initialization : ',init{1}]);
disp(' ');disp(['Accuracy :  ',num2str(Overall_Accuracy*100),' %']);
disp(' ');disp(['Precission :  ',num2str(Overall_Precision*100),' %']);
disp(' ');disp(['Recall :  ',num2str(Overall_Recall*100),' %']);
disp(' ');disp(['F1 Score :  ',num2str(F1_Score*100),' %']);

%% Functions

function [Seg,r,c,loop]=k_mean_seg(IMG,k,init,M)
    Bi_Seg=binary_segmentation(IMG);
    s=sqrt(size(IMG,1)*size(IMG,2)/k);
    epsilon=2*s;
    if nargin < 4
        M=10;
    end
    if k < 160
        epsilon=8;
    end
    F=feature_space(IMG);
    
    if strcmp(init{1},'random')
        [Init,r,c]=random_init(IMG,k);
    elseif strcmp(init{1},'uniform')
        [Init,r,c]=uniform_init(IMG,k);
    end
    for i=1:k
        Add=0;
        for kp=3:size(F,3)
            Add=Add+L2(Init(i,1,kp).*ones(size(F(:,:,kp))),F(:,:,kp).*Bi_Seg);
        end
        S(:,:,i)=M/s*L2(Init(i,1,[1,2]).*ones(size(F(:,:,[1,2]))),F(:,:,[1,2]).*Bi_Seg)+Add;
    end
    [~,Seg]=min(S,[],3);
    
    loop=1;
    while 1==1
        R=r;C=c;
        r=[];c=[];
        for i=1:k
            Log=(Seg==i);
            Fi=F.*Log;
            meanii=[];
            for ii=1:size(Fi,3)
                meanii=[meanii,sum(Fi(:,:,ii),'all')/sum(Log(:))];
            end
            r=[r,round(meanii(1))];
            c=[c,round(meanii(2))];
            Init(i,1,:)=meanii;
            Add=0;
            for kp=3:size(F,3)
                Add=Add+L2(Init(i,1,kp).*ones(size(F(:,:,kp))),F(:,:,kp).*Bi_Seg);
            end
            S(:,:,i)=M/s*L2(Init(i,1,[1,2]).*ones(size(F(:,:,[1,2]))),F(:,:,[1,2]).*Bi_Seg)+Add;
        end

        if L2(R,r)<epsilon & L2(C,c)<epsilon
            break
        end
        
        loop=loop+1;
        [~,Seg]=min(S,[],3);
    end
end

function [Init,m,n]=uniform_init(IMG,k)
    Bi_Seg=binary_segmentation(IMG);
    
    cc=find(sum(Bi_Seg,1)~=0);
    rr=find(sum(Bi_Seg,2)~=0);
    
    c1=cc(1);c2=cc(end);
    r1=rr(1);r2=rr(end);
    
    clear cc rr;

    F=feature_space(IMG);
    Init=zeros(k,1,size(F,3));
    [M,N]=mean_multiply(k);
    i=1;
    
    m=round(linspace(r1,r2,M+2));
    m=m(2:end-1);
    n=round(linspace(c1,c2,N+2));
    n=n(2:end-1);

    Mm=[];
    Nn=[];
    for r=m
        for c=n
            Init(i,1,:)=F(r,c,:);
            i=i+1;
            Mm=[Mm,r];
            Nn=[Nn,c];
        end
    end
    m=Mm;n=Nn;
end

function [Init,r,c]=random_init(IMG,k)
    Bi_Seg=binary_segmentation(IMG);
    F=feature_space(IMG);
    Init=zeros(k,1,size(F,3));
    r=randperm(size(F,1));
    c=randperm(size(F,2));
    count=1;i=1;
    R=[];C=[];
    while count <= k
        while Bi_Seg(r(i),c(i))==0
            i=i+1;
        end
        Init(count,1,:)=F(r(i),c(i),:);
        R=[R,r(i)];C=[C,c(i)];
        count=count+1;i=i+1;
    end
    r=R;c=C;
    clear R C
end

function F=feature_space(IMG)

    r_IMG=size(IMG,1);
    c_IMG=size(IMG,2);
    R=[]; C=[];
    for r=1:size(IMG,1)
        R=[R;ones(1,c_IMG)*r];
    end
    for c=1:size(IMG,2)
        C=[C,ones(r_IMG,1)*c];
    end
    
    BW = edge(IMG,'sobel');
    
    [Gmag,~] = imgradient(IMG,'prewitt');
    
    
    F(:,:,1)=R;
    F(:,:,2)=C;
    F(:,:,3)=IMG;
    F(:,:,4)=double(BW)*200;
    F(:,:,5)=double(Gmag)/max(Gmag(:))*10;
    
end

function S=L2(fi,fj)
    difference = ( fi - fj ).^2;
    Sum = sum(difference,3); 
    S = sqrt(Sum);
end

function [m,n]=mean_multiply(k)
    for i=ceil(sqrt(k))+1:-1:1
        if mod(k,i)==0
            m=i;
            n=k/m;
            break
        end
    end
end

function rgbImage=gray2rgb(Seg)
    assert(max(Seg(:))-min(Seg(:))+1<256,'Gray scale is exceeding 255!');
    
    rgbImage=zeros(size(Seg,1),size(Seg,2),3);
    k_max=max(Seg(:));
    k_min=min(Seg(:));
    k=k_max-k_min+1;
    
    r=randperm(k)/k;
    g=randperm(k)/k;
    b=randperm(k)/k;
    
    I=Seg-k_min+1;
    for i=1:k
        mask=double(I==i);
        rgbImage(:,:,1)=rgbImage(:,:,1)+r(i).*mask;
        rgbImage(:,:,2)=rgbImage(:,:,2)+g(i).*mask;
        rgbImage(:,:,3)=rgbImage(:,:,3)+b(i).*mask;
    end
end

function Y=binary_segmentation(X)
    Y=double(X>0);
end




