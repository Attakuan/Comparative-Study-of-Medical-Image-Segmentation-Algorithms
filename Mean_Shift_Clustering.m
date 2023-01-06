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

%% Mean Shift PreProcessing #1

Bi_Seg=binary_segmentation(IMG);
cc=find(sum(Bi_Seg,1)~=0);
rr=find(sum(Bi_Seg,2)~=0);
c1=cc(1);c2=cc(end);
r1=rr(1);r2=rr(end);
clear cc rr;

template=zeros(size(IMG));
IMG=IMG(r1:r2,c1:c2);

%% Mean Shift Preprocessing #2

[data_without_background,R_n,C_n,Assign]=AssignNonZero(IMG);

%% Mean Shift Clustring

data=data_without_background;

bw=input('Enter Bandwidth : ');

tic
[n,m] = MeanShiftCluster(transpose(data),bw);
toc

disp(' ');disp(' ');
disp(['Chosen Image : plane_',num2str(image_numbers(1,idx)),'th_Subject',num2str(image_numbers(2,idx))]);
disp(['Bandwidth = ',num2str(bw),'    Kernel : Flat Kernel']);

m=ReAssign(transpose(m),R_n,C_n,Assign);
clusteredImage=label2rgb(m);

template=ones(size(template)).*m(1,1);
template(r1:r2,c1:c2)=m;

%% Plotting the Results

FONT=17;

% FIG 1 :

figure;
subplot(1,3,1);
imshow(I,[]);
title('Original Image','fontsize',FONT);
subplot(1,3,2);
imshow(template,[]);
title({'Mean Shift Clustring',['Bandwidth = ',num2str(bw)],['Cluster Number : ',num2str(length(n))]},'fontsize',FONT);
subplot(1,3,3);
imshow(Label,[]);
title('Ground Truth','fontsize',FONT);
sgtitle(['Chosen Image : plane ',num2str(image_numbers(1,idx)),'th Subject ',num2str(image_numbers(2,idx))],'fontsize',FONT+1);

dim=[0.1,0.1,0.1,0.1];
htext=annotation('textbox',dim,'String','Choose the regions or region containing TUMOR by clicking once on them and then click ENTER to terminate and see the region/s with tumor!','FitBoxToText','on');

% FIG 2 :

loc=[];
try
    [c_input, r_input] = ginput;
    loc = [ loc ; int32([c_input r_input])];
catch end
delete(htext);

figure;
subplot(1,2,1);
A=logical(zeros(size(template)));
for k=1:size(loc,1)
    A=A|(template==template(loc(k,2),loc(k,1)));
end 
imshow(A,[]);
title('Segmented Tumor','fontsize',FONT);
subplot(1,2,2);
imshow(Label,[]);
title('Ground Truth','fontsize',FONT);
sgtitle(['Chosen Image : plane ',num2str(image_numbers(1,idx)),'th Subject ',num2str(image_numbers(2,idx))],'fontsize',FONT+1);

%% Numeric Analysis of Preprocessing

size1=length(I(:));
size2=length(IMG(:));
size3=length(data_without_background(:));

disp(' ');disp('Preprocessing 1 : ')
disp('Before binary segmentation was implemented in the preprocessing part');
disp(['the length of the Image vector is ',num2str(size1)]);
disp('After implementing binary segmentation we get lower dimensional image.');
disp(' ');disp('Preprocessing 2 : ');
disp('Before assigning only nonzero parts to a vector by using Assign Functions that we implemented');
disp(['the length of the Image vector is ',num2str(size2)]);
disp(' ');disp('Result of Preprocessing : ');
disp(['The final length of the Image vector is ',num2str(size3)]);
disp(' ');disp('The Summary of Preprocessing :');
disp(['The length of data that we used is decreasing : ',num2str(size1),' --> ',num2str(size2),' --> ',num2str(size3)]);

%% Accuracy, Precission, Recall, and F1 Score Analysis for Mean Shift Clustring

M=confusionmat(logical(Label(:)),A(:));

Diagonal=diag(M);
sum_rows=sum(M,2);

Overall_Accuracy = sum(Diagonal)/sum(M(:));

Precision=Diagonal./sum_rows;
Overall_Precision=mean(Precision);

sum_col=sum(M,1);

recall=Diagonal./sum_col';
overall_recall=mean(recall);

F1_Score=2*((Overall_Precision*overall_recall)/(Overall_Precision+overall_recall));

disp(' ');disp(' ');disp('Accuracy, Precission, Recall, and F1 Score Analysis for Mean Shift Clustring :');
disp(' ');disp(['Accuracy :  ',num2str(Overall_Accuracy*100),' %']);
disp(' ');disp(['Precission :  ',num2str(Overall_Precision*100),' %']);
disp(' ');disp(['Recall :  ',num2str(overall_recall*100),' %']);
disp(' ');disp(['F1 Score :  ',num2str(F1_Score*100),' %']);

%% Functions

function [clustCent,data2cluster] = MeanShiftCluster(data,bw)
    
    % ---INPUT---
    % data              - input data, (numDim x numPts)
    % bw                - is bandwidth parameter (scalar)
    % ---OUTPUT---
    % clustCent         - is locations of cluster centers (numDim x numClust)
    % data2cluster      - for every data point which cluster it belongs to (numPts)

    %**** Initialize stuff ***
    [~,numPts] = size(data);
    numClust        = 0;
    bandSq          = bw^2;
    initPtInds      = 1:numPts;
    stopThresh      = 1e-3*bw;                              %when mean has converged
    clustCent       = [];                                   %center of clust
    beenVisitedFlag = zeros(1,numPts,'uint8');              %track if a points been seen already
    numInitPts      = numPts;                               %number of points to posibaly use as initilization points
    clusterVotes    = zeros(1,numPts,'uint16');             %used to resolve conflicts on cluster membership
    
    while numInitPts
        tempInd         = ceil( (numInitPts-1e-6)*rand);        %pick a random seed point
        stInd           = initPtInds(tempInd);                  %use this point as start of mean
        myMean          = data(:,stInd);                           % intilize mean to this points location
        myMembers       = [];                                   % points that will get added to this cluster                          
        thisClusterVotes    = zeros(1,numPts,'uint16');         %used to resolve conflicts on cluster membership
        
        while 1     %loop untill convergence

            sqDistToAll = (repmat(myMean,1,numPts) - data).^2;    %dist squared from mean to all points still active
            inInds      = find(sqDistToAll < bandSq);               %points within bandWidth
            thisClusterVotes(inInds) = thisClusterVotes(inInds)+1;  %add a vote for all the in points belonging to this cluster

            myOldMean   = myMean;                                   %save the old mean
            myMean      = mean(data(:,inInds),2);                %compute the new mean
            myMembers   = [myMembers inInds];                       %add any point within bandWidth to the cluster
            beenVisitedFlag(myMembers) = 1;                         %mark that these points have been visited
            
            %**** if mean doesn't move much stop this cluster ***
            if norm(myMean-myOldMean) < stopThresh

                %check for merge posibilities
                mergeWith = 0;
                for cN = 1:numClust
                    distToOther = norm(myMean-clustCent(:,cN));     %distance from posible new clust max to old clust max
                    if distToOther < bw/2                    %if its within bandwidth/2 merge new and old
                        mergeWith = cN;
                        break;
                    end
                end


                if mergeWith > 0    % something to merge
                    clustCent(:,mergeWith)       = 0.5*(myMean+clustCent(:,mergeWith));             %record the max as the mean of the two merged (I know biased twoards new ones)
                    %clustMembsCell{mergeWith}    = unique([clustMembsCell{mergeWith} myMembers]);   %record which points inside 
                    clusterVotes(mergeWith,:)    = clusterVotes(mergeWith,:) + thisClusterVotes;    %add these votes to the merged cluster
                else    %its a new cluster
                    numClust                    = numClust+1;                   %increment clusters
                    clustCent(:,numClust)       = myMean;                       %record the mean  
                    %clustMembsCell{numClust}    = myMembers;                    %store my members
                    clusterVotes(numClust,:)    = thisClusterVotes;
                end
                break;
            end
        end


        initPtInds      = find(beenVisitedFlag == 0);           %we can initialize with any of the points not yet visited
        numInitPts      = length(initPtInds);                   %number of active points in set
    end
    [~,data2cluster] = max(clusterVotes,[],1);                %a point belongs to the cluster with the most votes
    
end

function A=ReAssign(out,R_n,C_n,Assign)
    A=zeros(size(Assign));
    for k=1:length(R_n)
        A(R_n(k),C_n(k))=out(k,1);
    end
end

function [out,R_n,C_n,Assign]=AssignNonZero(A)
    [C,R]=meshgrid([1:size(A,2)],[1:size(A,1)]);
    Assign=(A~=0);
    R_n=R(Assign);
    C_n=C(Assign);
    out=zeros(length(R_n),1);
    assert(length(R_n)==length(C_n),'There is a dimension problem!');
    for k=1:length(R_n)
        out(k,1)=A(R_n(k),C_n(k));
    end
end

function Y=binary_segmentation(X)
    Y=double(X>0);
end



