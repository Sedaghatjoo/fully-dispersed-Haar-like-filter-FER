clc;
clear;
close all

load Pics
load Labels
s_pic = 48;

%% divide data to train data and test data
classNum = 7;
div = 0.8;
for iter=1:20
    kt = 0;
    for it=1:classNum
        ind1 = (Labels==it);
        data = Pics(ind1,:);
        N = sum(ind1);
        in = randperm(N);
        data = data(in,:);
        
        Ntrain(it) = floor(div*N);
        traindata = data(1:Ntrain(it),:);
        
        ntr = sum(Ntrain(1:it));
        Xtrain(ntr-Ntrain(it)+1:ntr,:)= data(1:Ntrain(it),:);
        Ytrain(ntr-Ntrain(it)+1:ntr) = it;
        
        
        
        Ntest(it) = N - Ntrain(it);
        nte = sum(Ntest(1:it));
        Xtest(nte-Ntest(it)+1:nte,:)= data(Ntrain(it)+1:end,:);
        Ytest(nte-Ntest(it)+1:nte) = it;
        
        
        %%  divide Clutter to train Clutter and test Clutter
        for jt=1:classNum
            if jt~=it
                ind2 = Labels==jt;
                clutter = Pics(ind2,:);
                cluttertrain = clutter;
                NCtrain = size(cluttertrain,1);
                
                
                %% Create Filter
                % mean of train data
                Me_Face = sum(traindata)/Ntrain(it);
                Me_Face = reshape(Me_Face,s_pic,s_pic);
                
                Me_Clutter = sum(cluttertrain)/NCtrain;
                Me_Clutter = reshape(Me_Clutter,s_pic,s_pic);
                
                %Best Location for Filter
                Mf = (Me_Face) + (1/2-Me_Clutter);
                Mf = histeq(Mf);
                num1 = 64;%128; %pixels number of black
                num2 = 64;%128; %pixels number of white
                k=0;
                for j=1:s_pic
                    for i=1:s_pic
                        k=k+1;
                        Indi(k)=i;
                        Indj(k)=j;
                    end
                end
                Mf=Mf(1:end);
                [val,ind] = sort(Mf);
                
                ind1=ind(1:num1);
                I1=Indi(ind1);
                J1=Indj(ind1);
                %best Location for black pixels
                lor1=I1;
                loc1=J1;
                
                ind2=ind(end-num2:end);
                I2=Indi(ind2);
                J2=Indj(ind2);
                %best Location for white pixels
                lor2=I2;
                loc2=J2;
                
                
                W=zeros(s_pic,s_pic);
                B=zeros(s_pic,s_pic);
                for k=1:num1
                    B(I1(k),J1(k))=1;
                end
                for k=1:num2
                    W(I2(k),J2(k))=1;
                end
                
                filter=1/2*ones(s_pic,s_pic);
                filter(B>0)=0;
                filter(W>0)=1;
                %% extraction Features of train data
                B1=B(1:end);
                BB = repmat(B1,Ntrain(it),1);
                A = traindata.*BB;
                % A0=sum(A>0,2);
                % MuB1=sum(A,2)./A0;
                A0=sum(B1);
                MuB1=sum(A,2)./A0;
                
                W1=W(1:end);
                WW = repmat(W1,Ntrain(it),1);
                AA = traindata.*WW;
                % AA0=sum(AA>0,2);
                % MuW1=sum(AA,2)./AA0;
                AA0=sum(W1);
                MuW1=sum(AA,2)./AA0;
                
                m11=sum(MuB1)/length(MuB1);
                m12=sum(MuW1)/length(MuW1);
                M1 = [m11 m12];
                Mu1=[MuB1 , MuW1];
                Sigma1=[MuB1-m11 , MuW1-m12]'*[MuB1-m11 , MuW1-m12];
                
                %% extraction Features of Clutter data
                BBC = repmat(B1,NCtrain,1);
                A_cl = cluttertrain.*BBC;
                % A0_cl = sum(A_cl>0,2);
                % MuB2 = sum(A_cl,2)./A0_cl;
                MuB2 = sum(A_cl,2)./A0;
                
                WWC = repmat(W1,NCtrain,1);
                AA_cl = cluttertrain.*WWC;
                % AA0_cl=sum(AA_cl>0,2);
                % MuW2=sum(AA_cl,2)./AA0_cl;
                MuW2=sum(AA_cl,2)./AA0;
                
                m21=sum(MuB2)/length(MuB2);
                m22=sum(MuW2)/length(MuW2);
                M2=[m21 m22];
                Mu2=[MuB2 , MuW2];
                Sigma2=[MuB2-m21  MuW2-m22]'*[MuB2-m21  MuW2-m22];
                
                %% Optimal Weight by Fisher's Linear Discriminant
                Wh=[-1;1];
                tef = M1*Wh;
                tec = M2*Wh;
                
                Ff = Mu1*Wh;
                Fc = Mu2*Wh;
                for i=1:1000
                    te(i) = tec+i/1000*(tef-tec);
                    error_f = sum(Ff<te(i))/Ntrain(it);
                    error_c=sum(Fc>te(i))/NCtrain;
                    err(i)=norm([error_f error_c],'inf');
                end
                error = min(err);
                te_op = te(err==min(err));
                te_op = te_op(1);
                
                %%
                ii=1;
                weightf = ones(1,Ntrain(it));
                WR = zeros(1,Ntrain(it));
                Neg=1;
                weightc = ones(1,NCtrain);
                WRc = zeros(1,NCtrain);
                Pos=1;
                % clear Me_Face
                
                while( ii<200)
                    
                    weightf  = weightf  + WR;
                    Me_Face  = weightf*traindata/sum(weightf);
                    Me_Face = reshape(Me_Face,s_pic,s_pic);
                    
                    weightc  = weightc  + WRc;
                    Me_Clutter  = weightc*cluttertrain/sum(weightc);
                    Me_Clutter = reshape(Me_Clutter,s_pic,s_pic);
                    In_Me_Clutter = (1/2-Me_Clutter);
                    
                    %% Best Location for Filter
                    cf=1;%length(Neg)/(Ntrain(it));
                    cc=1;%length(Pos)/(Ntrain(it));
                    
                    Mf = cf*(Me_Face) + cc*(1/2-Me_Clutter);
                    Mf = histeq(Mf);
                    Mf=Mf(1:end);
                    [val,ind] = sort(Mf);
                    
                    ind1=ind(1:num1);
                    I1=Indi(ind1);
                    J1=Indj(ind1);
                    %best Location for black pixels
                    lor1=I1;
                    loc1=J1;
                    
                    ind2=ind(end-num2:end);
                    I2=Indi(ind2);
                    J2=Indj(ind2);
                    %best Location for white pixels
                    lor2=I2;
                    loc2=J2;
                    
                    W=zeros(s_pic,s_pic);
                    B=zeros(s_pic,s_pic);
                    for k=1:num1
                        B(I1(k),J1(k))=1*-(Wh(1));
                    end
                    for k=1:num2
                        W(I2(k),J2(k))=1*(Wh(2));
                    end
                    
                    filter=1/2*ones(s_pic,s_pic);
                    filter(B>0)=0;
                    filter(W>0)=1;
                    
                    %% extraction Features of train data
                    
                    B1=B(1:end);
                    BB = repmat(B1,Ntrain(it),1);
                    A = traindata.*BB;
                    %         A0=sum(A>0,2);
                    A0   = sum(B1);
                    MuB1=sum(A,2)./A0;
                    
                    W1=W(1:end);
                    WW = repmat(W1,Ntrain(it),1);
                    AA = traindata.*WW;
                    %         AA0=sum(AA>0,2);
                    AA0   = sum(W1);
                    MuW1=sum(AA,2)./AA0;
                    
                    
                    m11=sum(MuB1)/length(MuB1);
                    m12=sum(MuW1)/length(MuW1);
                    M1 = [m11 m12];
                    Mu1=[MuB1 , MuW1];
                    Sigma1=[MuB1-m11 , MuW1-m12]'*[MuB1-m11 , MuW1-m12];
                    
                    %% extraction Features of Clutter data
                    BBC = repmat(B1,NCtrain,1);
                    A_cl = cluttertrain.*BBC;
                    %         A0_cl = sum(A_cl>0,2);
                    MuB2 = sum(A_cl,2)./A0;
                    
                    WWC = repmat(W1,NCtrain,1);
                    AA_cl = cluttertrain.*WWC;
                    %         AA0_cl=sum(AA_cl>0,2);
                    MuW2=sum(AA_cl,2)./AA0;
                    
                    m21=sum(MuB2)/length(MuB2);
                    m22=sum(MuW2)/length(MuW2);
                    M2=[m21 m22];
                    Mu2=[MuB2 , MuW2];
                    Sigma2=[MuB2-m21  MuW2-m22]'*[MuB2-m21  MuW2-m22];
                    
                    %% Optimal Weight by Fisher's Linear Discriminant
                    
                    tef = M1*Wh;
                    tec = M2*Wh;
                    
                    Ff = Mu1*Wh;
                    Fc = Mu2*Wh;
                    for i=1:1000
                        te(i) = tec+i/1000*(tef-tec);
                        error_f = sum(Ff<te(i))/Ntrain(it);
                        error_c=sum(Fc>te(i))/NCtrain;
                        err(i)=norm([error_f error_c],'inf');
                    end
                    error(ii) = min(err);
                    te_op = te(err==min(err));
                    te_op = te_op(1);
                    
                    
                    [af bf] =find(Ff<te_op);
                    Neg=af;
                    
                    [ac bc] =find(Fc>te_op);
                    Pos = ac;
                    
                    
                    ep=20;
                    Rf = (Ff-(te_op))';
                    WR = 0.01./(1+exp(ep*Rf));
                    
                    Rc = (Fc-(te_op))';
                    WRc= 0.01./(1+exp(-ep*Rc));
                    
                    
                    %% Test of testdata
                    
                    %**********test data************
                    %             BBt = repmat(B1,Ntest,1);
                    %             At = testdata.*BBt;
                    % %             A0t=sum(At>0,2);
                    %             MuB1t=sum(At,2)./A0;
                    %
                    %             WWt = repmat(W1,Ntest,1);
                    %             AAt = testdata.*WWt;
                    % %             AA0t=sum(AAt>0,2);
                    %             MuW1t=sum(AAt,2)./AA0;
                    %
                    %             Mu1t=[MuB1t , MuW1t];
                    %             %***********test Clutter************
                    %             BBCt = repmat(B1,NCtest,1);
                    %             A_clt = cluttertest.*BBCt;
                    % %             A0_clt = sum(A_clt>0,2);
                    %             MuB2t = sum(A_clt,2)./A0;
                    %
                    %             WWCt = repmat(W1,NCtest,1);
                    %             AA_clt = cluttertest.*WWCt;
                    % %             AA0_clt=sum(AA_clt>0,2);
                    %             MuW2t=sum(AA_clt,2)./AA0;
                    %             Mu2t=[MuB2t , MuW2t];
                    %             %*****************************
                    %             Fft=Mu1t*Wh;
                    %             [aft bft] =find(Fft<te_op);
                    %             Negt=aft;
                    %
                    %
                    %             Fct=Mu2t*Wh;
                    %             [act bct] =find(Fct>te_op);
                    %             Post = act;
                    %
                    %             error_ft=sum(Fft<te_op)/Ntest;
                    %             error_ct=sum(Fct>te_op)/NCtest;
                    %             errt=norm([error_ft error_ct],'inf');
                    %             errort(ii)=min(errt);
                    %
                    %
                    Acc(ii) = (sum(Ff>te_op)+sum(Fc<te_op))/(Ntrain(it)+NCtrain);
                    %             Acct(ii) = (sum(Fft>te_op)+sum(Fct<te_op))/(Ntest+NCtest);
                    %
                    %                     figure(jt)
                    %                     plot(Acct,'-r')
                    
                    %                     hold on
                    %                     plot(Acc,'-g')
                    %                     hold on
                    
                    
                    ii=ii+1;
                end
                
                kt = kt+1;
                Bb{kt} = B;
                Ww{kt} = W;
                te(kt) = te_op;
                clear MuB1 MuB2 MuW1 MuW2 Mu1 Mu2
            end
        end
        it;
        
    end
    
    %% Features Extraction
    for i=1:classNum
        indF1  = find(Ytrain==i);
        data1 = Xtrain(indF1,:);
        N1(i)= size(data1,1);
        nn1 = sum(N1(1:i));
        
        
        for j=1:classNum*(classNum-1)
            B  = Bb{j};
            B1=B(1:end);
            BB = repmat(B1,N1(i),1);
            A = data1.*BB;
            A0  = sum(B1);
            MuB1=sum(A,2)./A0;
            
            W  = Ww{j};
            W1=W(1:end);
            WW = repmat(W1,N1(i),1);
            AA = data1.*WW;
            AA0   = sum(W1);
            MuW1=sum(AA,2)./AA0;
            Mu1=[MuB1 , MuW1]*[-1;1]-te(j);
            
            XTrain(nn1-N1(i)+1:nn1,j)  = Mu1;
        end
        %         YTrain(nn1-N1(i)+1:nn1) = i*ones(N1(i),1);
    end
    YTrain = Ytrain;
    %*********Features Extraction  of Test Data
    dataT = Xtest;
    NTest = size(dataT,1);
    YTest = Ytest;%zeros(Ntest,1);
    for i=1:classNum*(classNum-1)
        clear MuB1 MuB2 MuW1 MuW2 Mu1 Mu2
        B  = Bb{i};
        B1=B(1:end);
        BB2 = repmat(B1,NTest,1);
        A2 = dataT.*BB2;
        A0  = sum(B1);
        MuB2=sum(A2,2)./A0;
        
        W  = Ww{i};
        W1=W(1:end);
        WW2 = repmat(W1,NTest,1);
        AA2 = dataT.*WW2;
        AA0   = sum(W1);
        MuW2=sum(AA2,2)./AA0;
        Mu2=[MuB2 , MuW2]*[-1;1]-te(i);
        
        XTest(:,i)=Mu2;
        
        %     indF2  = find(NumbersTest(:,1)==i-1);
        %     YTest(indF2) = i*ones(length(indF2),1);
        
    end
    
    %% SVM Classification
    t = templateSVM(...
        'KernelFunction', 'polynomial','PolynomialOrder', 2,'KernelScale',...
        'auto','BoxConstraint', 1,'Standardize', true);
    % t = templateSVM('Standardize',true,'KernelFunction','gaussian');
    % Mdl = fitcecoc(X,Y,'Learners',t,'FitPosterior',true,...
    %     'ClassNames',{'setosa','versicolor','virginica'},...
    %     'Verbose',2);
    svm_model1 = fitcecoc(XTrain,YTrain);
    svm_model2 = fitcecoc(XTrain,YTrain,'Learners',t);
    
    output1 = predict(svm_model1, XTest);
    output2 = predict(svm_model2, XTest);
    
    
    Conf1 = confusionmat(YTest,output1);
    
    Conf2 = confusionmat(YTest,output2);
    
    
    AccTo1 =  sum(diag(Conf1)) / sum(Conf1(:)) *100;
    AccTo2 =  sum(diag(Conf2)) / sum(Conf2(:)) *100;
    for i=1:classNum
        accuracy1(iter,i)= Conf1(i,i) / (sum(Conf1(i,:))) *100;
        accuracy2(iter,i)= Conf2(i,i) / (sum(Conf2(i,:))) *100;
    end
    iter = iter+1
end

acc1 = mean(accuracy1)
acc2 = mean(accuracy2)




