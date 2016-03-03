addpath('C:\Users\susmit.joshi\Desktop\RIT Sem 4');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @misc{vedaldi08vlfeat,
%  Author = {A. Vedaldi and B. Fulkerson},
%  Title = {{VLFeat}: An Open and Portable Library
%           of Computer Vision Algorithms},
%  Year  = {2008},
%  Howpublished = {\url{http://www.vlfeat.org/}}

%%Emotion Intensity Recognition
%% Author: Susmit Mohan Joshi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%###lib-svm###%
%Chih-Chung Chang and Chih-Jen Lin, LIBSVM: a library for support vector machines.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mstd=50;
nstd=60;
%%%%%%%%%%%Adding path for vl_feat library%%%%%%%%%%%%%%%%%%%%
addpath('C:\MATLAB#1\vlfeat-0.9.20\toolbox\misc');
addpath('C:\MATLAB#1\vlfeat-0.9.20\toolbox');
vl_setup;

% addpath('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\drtoolbox');
% mexall;
%%%%%Reading the test images and applying LBP
toread=dir('C:\Users\susmit.joshi\Desktop\Dataset_Intensity\');
tostore=dir('C:\Users\susmit.joshi\Desktop\Dataset_Intensity_Landmarks\');
%toread=dir('C:\Users\susmit.joshi\Desktop\Expressions6\');
count=0;
%%%%%TestingSamples%%%%%
ntest=0;
%i=6;
for i=3:15
   toreadsubfolder=dir(strcat('C:\Users\susmit.joshi\Desktop\Dataset_Intensity\',toread(i).name));
   tostoresubfolder=dir(strcat('C:\Users\susmit.joshi\Desktop\Dataset_Intensity_Landmarks\',tostore(i).name));
  % toreadsubfolder=dir(strcat('C:\Users\susmit.joshi\Desktop\Expressions6\',toread(i).name));
    length_count(i-2)=length(toreadsubfolder)-3-ntest;
  for j=3:length(toreadsubfolder)-1-ntest
    % for j=3+ntest:length(toreadsubfolder)-1
        path=strcat('C:\Users\susmit.joshi\Desktop\Dataset_Intensity\',toread(i).name,'\',toreadsubfolder(j).name);
        %path=strcat('C:\Users\susmit.joshi\Desktop\Expressions6\',toread(i).name,'\',toreadsubfolder(j).name);
        img=imread(strcat('C:\Users\susmit.joshi\Desktop\Dataset_Intensity\',toread(i).name,'\',toreadsubfolder(j).name));
        landmark_im=load(strcat('C:\Users\susmit.joshi\Desktop\Dataset_Intensity_Landmarks\',tostore(i).name,'\',tostoresubfolder(j).name));
        x=landmark_im.txt2save(:,1); y=landmark_im.txt2save(:,2);
        %img=imread(strcat('C:\Users\susmit.joshi\Desktop\Expressions6\',toread(i).name,'\',toreadsubfolder(j).name));
        count=count+1;
        [k l m]=size(img);
        if(m==3)
            img=rgb2gray(img);
        end
        %%%%%%%%%%%%Used for cropping the face%%%%%%%%%%%%%%%%
        x=ceil(x);
        y=ceil(y);
        countx=0;county=0;
        for  z1=1:1:length(x)
            img(y(z1),x(z1))=255;
        end
        
        icoor=min(y); jcoor=min(x);
        icoornext=max(y); jcoornext=max(x) ;
        
        imnew=img(icoor:icoornext,jcoor:jcoornext);
         
%         mapp=getmapping(8,'u1');
        imnew=imresize(imnew,[250 250]);
        lbp=vl_lbp(single(imnew),15);
        lbp = reshape(lbp,prod(size(lbp)),1);
         lbp1(:,count)=lbp;

     
    end
end

Feature_Vectors=lbp1';
save('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\features.mat','Feature_Vectors');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%Labelling for SVMs%%%%%%%%%%%%%%%%%%%
labels=-1*ones(13,sum(length_count));
buffer=length_count(1);
for i=1:13
    if(i==1)
    labels(1,1:length_count(i))=1;
    else
    labels(i,(buffer+1):(buffer+length_count(i)))=1;
    buffer= buffer+length_count(i); 
    end
end

save('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\svmlabels.mat','labels');
save('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\lengthcount.mat','length_count');