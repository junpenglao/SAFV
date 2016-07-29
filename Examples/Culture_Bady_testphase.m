%% fixation preferences
PredM = dataset('File','TestPhase.csv','Delimiter',',');
PredM.Emo2=nominal(PredM.Emo2);
PredM.Famil=nominal(PredM.Famil);
PredM.weight=PredM.TotlDur./5;
PredM.fixDurPor=PredM.fixDur./PredM.TotlDur;
PredM.Famil=nominal(PredM.Emo==PredM.Emo2,{'novel','famil'});
% export(PredM,'File',['TestPhase.csv'],'Delimiter',',')
dstmp=PredM(PredM.Famil=='famil',:);
dstmp.FA=dstmp.fixDur;
dstmp.NW=PredM.fixDur(PredM.Famil=='novel');
dstmp.Pref=dstmp.FA-dstmp.NW;
lme4=fitlme(dstmp,'Pref ~ Fctype * Grp * Emo + (TotlDur|Sbj)','Dummyvarcoding','effect');
anova(lme4)
%% smooth 2D historgram (Plotly)
lim1=-.5;
lim2=5;
nbin=25;
intervel=linspace(lim1,lim2,nbin);
bsize=intervel(2)-intervel(1);
% create the "computational grid"
n1 = 8; n2 = 8;
x1 = linspace(-0.0001,lim2,n1+1); x2 = linspace(-0.0001,lim2,n2+1);
xg = {(x1(1:n1)+x1(2:n1+1))'/2, (x2(1:n2)+x2(2:n2+1))'/2};

for ij=1:2
    if ij==1
        xdata1=PredM.fixDur(PredM.Emo2=='FEAR');
        ydata1=PredM.fixDur(PredM.Emo2=='HAPPY');
        Grpvec=PredM.Grp(PredM.Emo2=='FEAR');
        Sbj=PredM.Sbj(PredM.Emo2=='FEAR');
        xlabelname='Fear';
        ylabelname='Happy';
    else
        xdata1=PredM.fixDur(PredM.Famil=='famil');
        ydata1=PredM.fixDur(PredM.Famil=='novel');
        Grpvec=PredM.Grp(PredM.Famil=='famil');
        Sbj=PredM.Sbj(PredM.Famil=='famil');
        xlabelname='Familiarized Face';
        ylabelname='Novel Face';
    end
    
    
    xy = [xdata1,ydata1];
    
    i = ceil((xy(:,1)-x1(1))/(x1(2)-x1(1)));
    j = ceil((xy(:,2)-x2(1))/(x2(2)-x2(1)));
    counts = full(sparse(i,j,1,n1,n2));
    
    % [a,b]=find(tocont==max(tocont(:)))
    % lx(a),ly(b)
    
    Ns=length(unique(Sbj));
    Nboot=10000;
    n1n=n1;n2n=n2;
    x1n = linspace(-0.0001,lim2,n1n+1); x2n = linspace(-0.0001,lim2,n2n+1);
    bootmat=zeros(n1n,n2n,Nboot);
    indxSbj=kron(eye(Ns),[1;1]);
    indxSbj(indxSbj==1)=1:length(Sbj);
    for ib=1:Nboot
        tmp = randi(Ns,1,Ns);
        bindx=indxSbj(:,tmp);
        bs=bindx(bindx~=0);
        % bs=randi(length(Sbj),1,length(Sbj));
        xy2 = [xdata1(bs),ydata1(bs)];
        i = ceil((xy2(:,1)-x1n(1))/(x1n(2)-x1n(1)));
        j = ceil((xy2(:,2)-x2n(1))/(x2n(2)-x2n(1)));
        counts2 = full(sparse(i,j,1,n1n,n2n));
        %     tocont2=interp2(counts2',5);
        [a,b]=find(counts2==max(counts2(:)));
        for ii=1:length(a)
            bootmat(a(ii),b(ii),ib)=1;
        end
    end
    sbootmat=(sum(bootmat,3));
    [s1,s2]=sort(sbootmat(:),'descend');
    ss1=cumsum(s1);
    kss1=find(ss1>9500);
    region95=zeros(size(sbootmat));
    region95(s2(1:(kss1(1)-1)))=s1(1:(kss1(1)-1));
    
    % tocont2=interp2(region95',5);
    tocont2=imresize(region95',[n1t,n2t],'nearest');
    tocont2(tocont2~=max(tocont2(:)))=0;
    tocont2(tocont2==max(tocont2(:)))=1;
    [C,htmp]=imcontour(lx,ly,tocont2,1,'w','LineWidth',1.5);
    close()
    % Plotly Histogram_2d_Contour
    a=histcontour2(xdata1,ydata1,xlabelname,ylabelname,['fig' num2str(ij)],0,bsize,C);
end

%% detail statistices
lme1=fitlme(dstmp,'FA ~ Fctype * Grp * Emo + (TotlDur|Sbj)','Dummyvarcoding','effect');
anova(lme1)
lme2=fitlme(dstmp,'NW ~ Fctype * Grp * Emo + (TotlDur|Sbj)','Dummyvarcoding','effect');
anova(lme2)

dY=[dstmp.FA,dstmp.NW];
DX=lme4.designMatrix;
[beta,Sigma,E,CovB,logL] = mvregress(DX,dY);
beta1=beta(:,1);
beta2=beta(:,2);
betall=[beta1;beta2];
Fvaltmp=zeros(length(beta),1);
Pvaltmp=zeros(length(beta),1);
df2=lme4.DFE*2;
for ib=1:length(beta)
    c=zeros(1,16);c(1,ib)=1;c(1,ib+8)=-1;
    % c=zeros(2,16);c(1,ib)=1;c(2,ib+8)=1;
    Fvaltmp(ib)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
    Pvaltmp(ib)=1-fcdf(Fvaltmp(ib),rank(c),df2);
end

disp(dataset(lme1.Coefficients.Estimate,lme2.Coefficients.Estimate,lme1.Coefficients.pValue,lme2.Coefficients.pValue,...
    beta1,beta2,Fvaltmp,Pvaltmp,...
    'Varnames',{'Beta_FA','Beta_NW','p_FA','p_NW','Beta_FA2','Beta_NW2','F','p'},...
    'ObsNames',lme1.Coefficients.Name))
model1=dataset(beta1,beta2,Fvaltmp,Pvaltmp,...
    'Varnames',{'Beta_FA','Beta_NW','F','p'},...
    'ObsNames',lme1.Coefficients.Name);
%% multivariante fixation preferences
clc
dY=[dstmp.FA,dstmp.NW];

uniGrp=unique(dstmp.Grp);
uniEmo=unique(dstmp.Emo);
uniFctype=unique(dstmp.Fctype);
ifig=0;
DXc=zeros(length(dY),8);
labelstr2=cell(8,1);
covect=[];
meanmat=zeros(8,2);
for ig=1:length(uniGrp) % WC or EA
    for ifami=1:length(uniEmo) % familiar with FEAR or HAPPY
        for ifctype=1:length(uniFctype)
            idx=dstmp.Grp==uniGrp(ig)&dstmp.Emo==uniEmo(ifami)&strcmp(dstmp.Fctype,uniFctype{ifctype});
            ifig=ifig+1;
            labelstr2(ifig)={[char(uniGrp(ig)),'-',char(uniEmo(ifami)),'-',uniFctype{ifctype}]};
            DXc(idx,ifig)=1;
            % tmp=cov(dY(idx,1),dY(idx,2));
            meanmat(ifig,1)=mean(dstmp.FA(idx));meanmat(ifig,2)=mean(dstmp.NW(idx));
            % covect(ifig,:)=[tmp(1,1),tmp(1,2),tmp(2,2)];
        end
    end
end

[beta,Sigma,E,CovB,logL] = mvregress(DXc,dY,'algorithm','ecm');
figure;hold on
for ii=1:8
    scatter(beta(ii,1),beta(ii,2),'fill');
    covect(ii,:)=[CovB(ii,ii),CovB(ii,ii+8),CovB(ii+8,ii+8)];
end
legend(labelstr2)
line([0 5],[0,5]);
axis([0 5 0 5],'square')
ylabel('Familiar');
xlabel('Novel');
dataset(beta(:,1),covect(:,2),beta(:,2),beta(:,1)-beta(:,2),covect(:,1),covect(:,3),...
    'ObsNames',labelstr2,...
    'VarNames',{'Familiar','COV','Novel','Diff','VarF','VarN'})
%% numerical reports
beta1=beta(:,1);
beta2=beta(:,2);
betall=[beta1;beta2];
Fvaltmp=zeros(length(beta),1);
Pvaltmp=zeros(length(beta),1);
df2=lme4.DFE*2;
contrastmp=limo_OrthogContrasts([2,2,2]);
for ib=2:length(beta)
    c=zeros(1,16);c(1,1:8)=contrastmp{ib-1};c(1,9:16)=-contrastmp{ib-1};
    % c=zeros(2,16);c(1,1:8)=contrastmp{ib-1};c(2,9:16)=contrastmp{ib-1};
    Fvaltmp(ib)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
    Pvaltmp(ib)=1-fcdf(Fvaltmp(ib),rank(c),df2);
end

c=[ones(1,8),-ones(1,8)];
Fvaltmp(9)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
Pvaltmp(9)=1-fcdf(Fvaltmp(9),rank(c),df2);

c=[1 1 -1 -1 1 1 -1 -1, -1 -1 1 1 -1 -1 1 1];
Fvaltmp(10)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
Pvaltmp(10)=1-fcdf(Fvaltmp(10),rank(c),df2);


model2=dataset(Fvaltmp,Pvaltmp,...
    'Varnames',{'F','p'},...
    'ObsNames',{'Intercept','Grp','Emo','Fctype', ...
    'Grp:Emo','Grp:Fctype','Emo:Fctype','Grp:Emo:Fctype', ...
    'Familiarity','Expression'});
disp('Overall Effect')
disp(model2)

Nboot=10000;
alpha=.05;
meanMatCI=zeros(6,3);
for itype=1:2
    if itype==1
        FEvec=PredM.fixDur(PredM.Emo2=='FEAR');
        HAvec=PredM.fixDur(PredM.Emo2=='HAPPY');
    else
        FEvec=PredM.fixDur(PredM.Famil=='novel');
        HAvec=PredM.fixDur(PredM.Famil=='famil');
    end
    meanMatCI((itype-1)*3+1,1)=mean(FEvec);
    meanMatCI((itype-1)*3+2,1)=mean(HAvec);
    meanMatCI((itype-1)*3+3,1)=mean(FEvec-HAvec);
    
    nstmp=length(FEvec)/2;
    tempvec=zeros(Nboot,3);
    for ib=1:Nboot
        % bootstrap the subject
        itemsel=randsample(1:2:nstmp*2,nstmp,'true');
        selvec=[itemsel,itemsel+1];
        tempvec(ib,1)=mean(FEvec(selvec));
        tempvec(ib,2)=mean(HAvec(selvec));
        tempvec(ib,3)=mean(FEvec(selvec)-HAvec(selvec));
    end
    sortbtvec=sort(tempvec,1);
    meanMatCI((itype-1)*3+[1:3],2)=sortbtvec(round((alpha/2)*Nboot),:);
    meanMatCI((itype-1)*3+[1:3],3)=sortbtvec(round((1-alpha/2)*Nboot),:);
end
disp(dataset(meanMatCI,...
    'ObsNames',{'Fear','Happy','Fear-Happy','Novel','Familiarized','Novel-Familiarized'}))
% diffvec=beta(:,2)-beta(:,1);diffvec([1,2,5,6])=diffvec([1,2,5,6])*-1;figure;bar(diffvec);
% set(gca,'XTick',1:8,...                         % Change the axes tick marks
%     'XTickLabel',labelstr2);
%% barplot of each catigorical condition (with bootstrap 95%CI)
uniGrp=unique(PredM.Grp);
uniEmo=unique(PredM.Emo);
uniFctype=unique(PredM.Fctype);
ifig=0;
cc=colormap('parula');close()
X=[1];swvec=[-.5 -.2 .2 .5];ccvec=[64 55 10 1];

dY=[dstmp.FA,dstmp.NW];
meanMatCI2=zeros(4,3);
labelstr2=cell(4,1);
Nboot=10000;
alpha=.05;
for ifami=1:length(uniEmo) % familiar with FEAR or HAPPY
    for ifctype=1:length(uniFctype)
        idx=PredM.Emo==uniEmo(ifami)&strcmp(PredM.Fctype,uniFctype{ifctype});
        ifig=ifig+1;
        labelstr2(ifig)={[char(uniEmo(ifami)),'-',uniFctype{ifctype}]};
        tempds=PredM(idx,:);
        FEvec=tempds.fixDur(tempds.Emo2=='FEAR');
        HAvec=tempds.fixDur(tempds.Emo2=='HAPPY');
        meanMatCI2(ifig,1)=mean(FEvec-HAvec);
        nstmp=length(FEvec)/2;
        tempvec=zeros(Nboot,1);
        for ib=1:Nboot
            % bootstrap the subject
            itemsel=randsample(1:2:nstmp*2,nstmp,'true');
            selvec=[itemsel,itemsel+1];
            
            tempvec(ib)=mean(FEvec(selvec)-HAvec(selvec));
        end
        sortbtvec=sort(tempvec);
        meanMatCI2(ifig,2)=sortbtvec(round((alpha/2)*Nboot));
        meanMatCI2(ifig,3)=sortbtvec(round((1-alpha/2)*Nboot));
    end
end
figure;
barp=.125;
selvec=[4;3;2;1];% for the order of the label, double check
legendname={'Happy:Own-Race','Happy:Other-Race','Fear:Own-Race','Fear:Other-Race'};

hold on;
for ii=1:4
    Y=meanMatCI2(selvec(ii,:),1);
    L=Y-meanMatCI2(selvec(ii,:),2);
    U=meanMatCI2(selvec(ii,:),3)-Y;
    bar1(ii)=bar(X+swvec(ii),Y,barp,'FaceColor',cc(ccvec(ii),:),'EdgeColor',[0 0 0],'LineWidth',1.5);hold on
    errorbar(X+swvec(ii),Y,L,U,'.k','linewidth',1.5)
end
ylim([-.6 1.6]);xlim([0 4])
title('Viewing Preferences (s)')
set(gca,'YTick',[-.5 0 .5 1 1.5],...
    'YTickLabel',{'-.5','0','.5','1','1.5'},...
    'TickLength',[0 0]);
legend(bar1,legendname,'Location','northeast')
disp(dataset(meanMatCI2,...
    'ObsNames',labelstr2))

%%
gFvaltmp=zeros(4,1);
gPvaltmp=zeros(4,1);
% Group comparison (differences of the Fear - Happy)
c=[1 1 -1 -1 -1 -1 1 1,-1 -1 1 1 1 1 -1 -1];
gFvaltmp(1)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(1)=1-fcdf(gFvaltmp(1),rank(c),df2);
% Group comparison (differences of the Familiar - Novel)
c=[1 1 1 1 -1 -1 -1 -1,-1 -1 -1 -1 1 1 1 1];
gFvaltmp(2)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(2)=1-fcdf(gFvaltmp(2),rank(c),df2);

% the viewing bias towards fearful expression is reduced when the infants were familiarized with fearful faces
c=[1 1 1 1 0 0 0 0,-1 -1 -1 -1 0 0 0 0; ...
    0 0 0 0 1 1 1 1,0 0 0 0 -1 -1 -1 -1];
gFvaltmp(3)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(3)=1-fcdf(gFvaltmp(3),rank(c),df2);

c=[1 -1 0 0 1 -1 0 0,-1 1 0 0 -1 1 0 0; ...
    0 -1 -1 0 0 -1 -1 0,0 1 1 0 0 1 1 0;
    0 -1 0 -1 0 -1 0 -1,0 1 0 1 0 1 0 1];
gFvaltmp(4)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(4)=1-fcdf(gFvaltmp(4),rank(c),df2);

c=[1 -3 -1 -1 0 0 0 0,-1 3 1 1 0 0 0 0; ...
    0 0 0 0 1 -3 -1 -1,0 0 0 0 -1 3 1 1];
gFvaltmp(5)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(5)=1-fcdf(gFvaltmp(5),rank(c),df2);

c=[1 -3 -1 -1 1 -3 -1 -1,-1 3 1 1 -1 3 1 1];
gFvaltmp(6)=((c*betall)'*((c*CovB*c')^-1)*(c*betall))./rank(c);
gPvaltmp(6)=1-fcdf(gFvaltmp(6),rank(c),df2);

addimodel2=dataset(gFvaltmp,gPvaltmp,...
    'Varnames',{'F','p'},...
    'ObsNames',{'Grp(Fear-Happy)','Grp(Famili-Novel)','Famili(Grp)','minialOwnFear1','minialOwnFear2','minialOwnFear3'});
disp('Linear Contrast')
disp(addimodel2)
