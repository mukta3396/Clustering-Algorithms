%% data preprocessing
dataset=SPECTFNew;
row=size(dataset,1);
col=size(dataset,2);
training=dataset(1:row,1:col-1);
classlabel=dataset(1:row,col);
%% initialisation
unity1=zeros(1,row)+1;
initialmem=zeros(2,row);
for i=1:row
initialmem(1,:)=rand(1,row);
initialmem(2,:)=unity1-initialmem(1,:);
end
center=zeros(2,col-1);
assign=zeros(row,1);
stop=0;
m=1.25;
unity=zeros(row,1)+1;
theta=zeros(row,2)+0.5;
%% Kmean clustering
while (stop~=100)
    stop=stop+1;
    center=(initialmem*training)./repmat(sum(initialmem,2),1,col-1);
    c1=repmat(center(1,:),row,1);
    c2=repmat(center(2,:),row,1);
    memc1=sqrt(sum((training-c1).*(training-c1),2));
    memc2=sqrt(sum((training-c2).*(training-c2),2));
    newmem1=(unity./((memc1./memc2).^(2/m-1)+(memc1./memc1).^(2/m-1)));
    newmem2=(unity./((memc2./memc2).^(2/m-1)+(memc2./memc1).^(2/m-1)));
    newmem=cat(2,newmem1,newmem2);
    if ((newmem-transpose(initialmem))<theta | newmem==transpose(initialmem))
        break;
    end    
end
%% accuracy check
for i=1:row
    if (newmem(i,1)>newmem(i,2))
        assign(i)=1;
    else
        assign(i)=0;
    end
end
assign_pos=assign;
assign_neg=1-assign;
confusionmatrix_pos = confusionmat(classlabel,assign_pos);
confusionmatrix_neg = confusionmat(classlabel,assign_neg); 
accuracy_pos=(confusionmatrix_pos(1,1)+confusionmatrix_pos(2,2))/sum(sum(confusionmatrix_pos));
accuracy_neg=(confusionmatrix_neg(1,1)+confusionmatrix_neg(2,2))/sum(sum(confusionmatrix_neg));
if(accuracy_pos>accuracy_neg)
    display(accuracy_pos*100);
    precision=confusionmatrix_pos(1,1)/(confusionmatrix_pos(1,1)+confusionmatrix_pos(1,2));
    recall=confusionmatrix_pos(1,1)/(confusionmatrix_pos(1,1)+confusionmatrix_pos(2,1));
else
    display(accuracy_neg*100);
    precision=confusionmatrix_neg(1,1)/(confusionmatrix_neg(1,1)+confusionmatrix_neg(1,2));
    recall=confusionmatrix_neg(1,1)/(confusionmatrix_neg(1,1)+confusionmatrix_neg(2,1));
end