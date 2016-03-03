%%%%%%%
%% Author: Susmit Mohan Joshi
%%%%
%% Script for Cross- Validation over the CK dataset for Emotion Intensity Recognition
clear all;

fprintf('Enter the Dimensionality Reduction Method Required: \n');
fprintf(' 1. PCA \n 2. LDA \n 3. sLPP \n 4. Kernel PCA \n');

method=input('Enter the method number: ');

addpath('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog');
features=load('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\features.mat');
data=features.Feature_Vectors;
[s1 s2]=size(data);
label=load('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\svmlabels.mat');
label=label.labels;
counter=load('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\Codes\lengthcount.mat');
length_count=counter.length_count;
AvgC=zeros(13,13);
%%%%%%%%%%Labels for Testing%%%%%%%%%%%%
lab=ones(1,s1);
buffer=length_count(1);
for i=1:13
    if(i==1)
        lab(1,1:length_count(i))=1;
    else
        lab(1,(buffer+1):(buffer+length_count(i)))=i;
        buffer= buffer+length_count(i);
    end
end
AccuSave=0;

iter=0;
TRAIN = crossvalind('Kfold',s1,21);
count_train=0; count_test=0;
for j=1:21
    count_train=0; count_test=0;iter=iter+1;
    for i=1:length(TRAIN)
        if(TRAIN(i)~=(j))
            count_train=count_train+1;
            Train_vectors(count_train,:)=data(i,:);
            Train_labels(:,count_train)=label(:,i);
            slpp(count_train)=lab(i);
        end
        if(TRAIN(i)==(j))
            count_test=count_test+1;
            Test_vectors(count_test,:)=data(i,:);
            Test(count_test)=lab(i);
        end
    end
    
    [ l1 l2]=size(Test_vectors)
    

%%%%%%%%%%%%%%%%%%% LPP %%%%%%%%%%%%%%%%%%%%
if(method==3)
      gnd = slpp';
      options = [];
      options.Metric = 'Euclidean';
      options.NeighborMode = 'Supervised';
          options.gnd = gnd; 
      W = constructW(double(Train_vectors),options);      
      options.PCARatio = 1;
      [eigvector, eigvalue] = LPP(W, options, double(Train_vectors));
      mappedA= double(Train_vectors)*eigvector;
      mean1=mean(Train_vectors);
      for i=1:l1
        temp(i,:)=double(Test_vectors(i,:))-mean1;
      end
      
       test_vectors=temp*eigvector;
       
      svmstring='-c 100 -g 0.0219959 -b 0.50 -t 1 -d 1';
end


%%%%%%%LDA%%%%%%%%%%%%%5
if(method == 2) 
      gnd = slpp';
      options = [];
      options.Fisherface = 1;
      options.ReguType='Ridge';
      options.Regu =0;
      options.ReguAlpha=100;
     options.ReducedDim=1; 
      options.PCARatio=100;
      [eigvector, eigvalue] = LDA(gnd, options, double(Train_vectors));
      mappedA = double(Train_vectors)*eigvector;
      
      mean1=mean(Train_vectors);
      for i=1:l1
        temp(i,:)=double(Test_vectors(i,:))-mean1;
      end
      
       test_vectors=temp*eigvector;
       
        svmstring='-c 200 -g 0.000719959 -b 0.50 -t 2 -d 1';
end

%%%%%%%%%% KernelPCA %%%%%%%%%%%%%

if(method == 4)
    [mappedA mapping] = compute_mapping(Train_vectors,'KernelPCA',1000,'poly');
    test_vectors= out_of_sample(double(Test_vectors), mapping);
    svmstring='-c 0.00005 -g 0.7 -b 1 -t 1 -d 1';
end

%%%%%%%PCA %%%%%%%%%%%%%%%%

if(method == 1)
    [mappedA mapping] = compute_mapping(Train_vectors,'PCA',1000);
    test_vectors= out_of_sample(double(Test_vectors), mapping);
    svmstring='-c 100 -g 0.001 -b 1 -t 1  -d 1';
end

    %%%%Training the SVM's%%%%%
    for  i=1:1:13 
        %%%LDA Code%%%
        model{i} = svmtrain(Train_labels(i,:)',double(mappedA), svmstring); %% KPCA-c 70 -g 0.1 -b 1 -t 1 -d 1
        w(i,:) = (model{i}.sv_coef' * full(model{i}.SVs)); %%LPP -c 100 -g 0.0219959 -b 0.50 -t 1 -d 1 size [250 250] Window= 15
        b(i) = model{i}.rho;        %%LPP -c 0.700 -g 0.119959 -b 0.50 -t 3 -d 1 SUPERVISED
    end
    count_label=0;
    for i=1:1:l1
        count_label=count_label+1;
        ans=w*test_vectors(i,:)'+b';
        [asa I]= max(ans);
        resultGT(count_label)=I;
    end
    
    %% Confusion Matrix and Accuracy calculation
    [C,order] = confusionmat(Test,resultGT,'ORDER',[1 2 3 4 5 6 7 8 9 10 11 12 13]);
    AvgC=AvgC+C;
    accu=trace(C)/(sum(sum(C)))
    % str=['accuracy=',num2str(accu*100),'%'];
    % disp(str);
    list(iter)=accu;
  end
accu=trace(AvgC)/(sum(sum(AvgC)));

%%-c 100 -g 0.001 -b 1 -t 1  -d 1 FOR PCA
%str=strcat['accuracy after cross-validation=',accu,' %'];
%%% For LDA -c 200 -g 0.000719959 -b 0.50 -t 2 -d 1
%%% For NPE -c 101 -g 0.029959 -b 0.50 -t 1 -d 1
%%%% For Testing Images= 54 , Training Images =1071 accuracy after cross-validation=85.0617
%save('C:\Users\susmit.joshi\OneDrive\FinalCE_Project\Expression Recog\CrossValExp\list.mat');

