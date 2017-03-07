%% data preprocessing
dataset=SPECTFNew;
row=size(dataset,1);
col=size(dataset,2);
training=dataset(1:row,1:col-1);
classlabel=dataset(1:row,col);
%% initialisation
dist1=zeros(row,1);
dist2=zeros(row,1);
assign=zeros(row,1);
oldr1=zeros(1,col-1);
oldr2=zeros(1,col-1);
k=2;
a = 1;
b = row;
r1 = floor((b-a).*rand(1,1) + a);
r2 = floor((b-a).*rand(1,1) + a);
newr1=training(r1,:);
newr2=training(r2,:);
stop=0;
%% Kmean clustering
while (stop~=100)
    count0=1;
    count1=1;
    stop=stop+1;
    if(oldr1==newr1 & oldr2==newr2)
        break;
    end
    for i=1:row
        dist1(i,1)=sqrt(sum((training(i,1:col-1)-newr1(1,1:col-1)).*(training(i,1:col-1)-newr1(1,1:col-1))) );
        dist2(i,1)=sqrt(sum((training(i,1:col-1)-newr2(1,1:col-1)).*(training(i,1:col-1)-newr2(1,1:col-1))));
    end
    oldr1=newr1;
    oldr2=newr2;
    for i=1:row
        if (dist1(i,1)<dist2(i,1))
            assign(i)=1;
            newr1(1,:)=newr1(1,:)+training(i,:);
            count0=count0+1;
        else
            assign(i)=0;
            newr2(1,:)=newr2(1,:)+training(i,:);
            count1=count1+1;
        end
    end
    newr1=newr1/count0;
    newr2=newr2/count1;
end
%% accuracy check
assign_pos=assign;
assign_neg=1-assign;
confusionmatrix_pos = confusionmat(classlabel,assign_pos);
confusionmatrix_neg = confusionmat(classlabel,assign_neg); 
accuracy_neg=(confusionmatrix_neg(1,1)+confusionmatrix_neg(2,2))/sum(sum(confusionmatrix_neg));
if(accuracy_pos>accuracy_neg)
    display(accuracy_pos*100);
else
    display(accuracy_neg*100);
end
