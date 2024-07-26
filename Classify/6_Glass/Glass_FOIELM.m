clear;clc;
Yita = 0.0004;            %  起始学习率
FailRate = 0.01;        %  衰减率
expnum = 1;            % 实验室次数
HiddenNum = 22;     %  隐节点数
experimentals = zeros(expnum,6);
for gogo = 1:expnum
%%%%%%%%%%%%%%%%%%%%%%%%% Adjustment parameters %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Learning Rate
% if gogo>1
%     Yita = Yita/(1+FailRate*gogo);
% end
% HiddenNum = HiddenNum + 1;
%%%%%%%%%%%%%%%%%%%%%%%%% Data processing %%%%%%%%%%%%%%%%%%%%%%%
data = load("Glass.txt");
Inputdim = size(data,2)-1;           %   (属性数)列数 - 1 （原因是最后一列为分类列）
DataNum = size(data,1);                     %   (样本数)行数
sorted_target = sort(data(:,Inputdim+1));  %   有序标签
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:DataNum                           %   确定数据集的类别数量
    if sorted_target(i,1) ~= label(1,j)     %   如果 排列标签(1,2)不=标签(1,1)    //matlab中 ~= 表 不等于
        j=j+1;                              %   j+1 = 2
        label(1,j) = sorted_target(i,1);    %   标签(1,2)=排列标签(1,2)
    end
end
ClassNum=j;                                 %   类别数量
OutputNeuronsNum=ClassNum;                  %   输出神经元数量
rowrank = randperm(DataNum);                %   打乱矩阵
data = data(rowrank,:);
AttributeData = data(:,1:Inputdim);  %   属性
TableData = data(:,Inputdim+1);      %   标签
TrainNum = round(0.6*DataNum);                     %   训练样本选60%
TestNum = DataNum - TrainNum;               %   测试样本为剩余
Train_Atttibute = AttributeData(1:TrainNum,:);              %   训练样本属性
Train_Table = TableData(1:TrainNum,:);                      %   训练样本标签
Test_Atttibute = AttributeData((TrainNum+1):DataNum,:);     %   测试样本属性
Test_Table = TableData((TrainNum+1):DataNum,:);             %   测试样本标签

Train_Table_Hot=zeros(OutputNeuronsNum, TrainNum);          %   独热编码预备全0矩阵(类别数量, 训练样本数量)
for i = 1:TrainNum
    for j = 1:ClassNum
        if label(1,j) == Train_Table(i,1)                   %   确定标签的类别 1.2.3...
            break;
        end
    end
    Train_Table_Hot(j,i)=1;                                 %   独热编码100 010 001 ...
end
Train_Table = Train_Table_Hot'*2 - 1;                         

Test_Table_Hot=zeros(OutputNeuronsNum, TestNum);            %   同上
for i = 1:TestNum
    for j = 1:ClassNum
        if label(1,j) == Test_Table(i,1)
            break;
        end
    end
    Test_Table_Hot(j,i)=1;                               
end
Test_Table = Test_Table_Hot'*2 - 1;

%%%%%%%%%%%%%%%%%%%%%%%%% Training Input Datas %%%%%%%%%%%%%%%%%%%%%%%
TrainInputXC=Train_Atttibute;           % 输入<中心>数据  
TrainInputXR=0.1*rand(TrainNum,Inputdim);      %  随机产生区间输入<半径>数据 
for i=1:TrainNum  
    if TrainInputXR(i,Inputdim)==0
        TrainInputXR(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end

TrainIdealOutputC = Train_Table;    %  <中心>数据理想输出
TrainIdealOutputR = 0.05*rand(TrainNum,ClassNum)+0.05;   %  <半径>数据理想输出，[0.05,0.1]公式 r = a + (b-a)*rand(m,n)其中[a,b]是范围

TestInputXC=Test_Atttibute;         %  输入<中心>数据  
TestInputXR=0.1*rand(TestNum,Inputdim);  %  随机产生区间输入<半径>数据 
for i=1:TestNum  
    if TestInputXR(i,Inputdim)==0
        TestInputXR(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end

TestIdealOutputC = Test_Table;    %  <中心>数据理想输出
TestIdealOutputR = 0.05*rand(TestNum,ClassNum)+0.05;   %  <半径>数据理想输出，[0.05,0.1]公式 r = a + (b-a)*rand(m,n)其中[a,b]是范围

%%%%%%%%%%%%%%%%%%%%%%%%% Parameters Set %%%%%%%%%%%%%%%%%%%%%%%
Alpha = 9/11;             %  分数阶阶数

Iterations = 500;       %  迭代步数
TrainMSEsave = zeros(1,Iterations);                         %   存储训练每次迭代的均方误差
TrainAccsave = zeros(1,Iterations);                         %   存储训练每次迭代的准确率
TestMSEsave = zeros(1,Iterations);                    %   存储训练每次迭代的梯度范数
TestAccsave = zeros(1,Iterations);                    %   存储训练每次迭代的准确率
MinError = 0.0000001;    %  最小误差
InputWeight = rand(Inputdim, HiddenNum)*2-1;     %   输入层权重W 总体[-1, 1] 随机 rand(隐节点数, 输入样本数量)
InputBias = rand(1, HiddenNum);           %   阈值b[0, 1]
OutputXita = rand(HiddenNum, ClassNum)*2-1;      %   输出层权重θ 总体[-1, 1] 随机 rand(隐节点数, 输入样本数量)
Delta_Xita = OutputXita*0;          %   每次迭代θ变值
%%%%%%%%%%%%%%%%%%%%%%%%% Training Datas %%%%%%%%%%%%%%%%%%%%%%%
tic;

for k = 1:Iterations
    Delta_Xita = Delta_Xita*0;                     %   每次迭代θ变值
    TrainInputZc = zeros(TrainNum, HiddenNum);     %   <中心>数据加权Sc
    TrainInputZr = zeros(TrainNum, HiddenNum);     %   <半径>数据加权Sr
    TrainInputHc = zeros(TrainNum, HiddenNum);     %   待激活<中心>数据输出Hc
    TrainInputHr = zeros(TrainNum, HiddenNum);     %   待激活<半径>数据输出Hr
    TrainOutputC = zeros(TrainNum, ClassNum);      %   输出<中心>
    TrainOutputR = zeros(TrainNum, ClassNum);      %   输出<半径>
    TestInputZc = zeros(TestNum, HiddenNum);     %   <中心>数据加权Sc
    TestInputZr = zeros(TestNum, HiddenNum);     %   <半径>数据加权Sr
    TestInputHc = zeros(TestNum, HiddenNum);     %   待激活<中心>数据输出Hc
    TestInputHr = zeros(TestNum, HiddenNum);     %   待激活<半径>数据输出Hr
    TestOutputC = zeros(TestNum, ClassNum);      %   输出<中心>
    TestOutputR = zeros(TestNum, ClassNum);      %   输出<半径>
    for j = 1:TrainNum

        for l = 1:HiddenNum    % 输入层-隐藏层
            for n = 1:Inputdim
                TrainInputZc(j,l) = InputWeight(n,l) * TrainInputXC(j,n) + TrainInputZc(j,l);
                TrainInputZr(j,l) = abs(InputWeight(n,l)) * TrainInputXR(j,n) + TrainInputZr(j,l);
            end
            TrainInputZc(j,l) =  TrainInputZc(j,l) + InputBias(1,l);
            ZCha = TrainInputZc(j,l) - TrainInputZr(j,l);
            ZHe = TrainInputZc(j,l) + TrainInputZr(j,l);
            TrainInputHc(j,l) = (act(ZCha, 1) + act(ZHe, 1))/2;   % 修改编号更改激活函数
            TrainInputHr(j,l) = (act(ZHe, 1) - act(ZCha, 1))/2;  % 修改编号更改激活函数
        end

        for m = 1:ClassNum    % 隐藏层-输出层
            for  l = 1:HiddenNum
                TrainOutputC(j,m) = act(TrainInputHc(j,l)*OutputXita(l,m), 0) + TrainOutputC(j,m);   % 修改编号更改激活函数
                TrainOutputR(j,m) = act(TrainInputHr(j,l)*abs(OutputXita(l,m)), 0) + TrainOutputR(j,m);   % 修改编号更改激活函数
            end
        end

    end

    TrainMSE = 0;                                      %   训练均方误差
    for j = 1:TrainNum
        for m = 1:ClassNum
            TrainMSE = (TrainOutputC(j,m) - TrainIdealOutputC(j,m))^2 + (TrainOutputR(j,m) - TrainIdealOutputR(j,m))^2 + TrainMSE;
        end
    end
    TrainMSE = (1/TrainNum)*TrainMSE;
    TrainMSEsave(k) = TrainMSE;                           %   记录训练每次迭代均方误差
    
    if TrainMSE<MinError                              %   判断当前误差不小于最小误差
        break
    end

    TrainClassifySuccess = 0;                                %   记录分类成功个数
    TrainFindI = TrainIdealOutputC';
    TrainFindR = TrainOutputC';
    TrainIdealResults = zeros(TrainNum,1);
    TrainRealResults = zeros(TrainNum,1);
    for i = 1 : TrainNum
        [~, TrainIdealLable]=max(TrainFindI(:,i));
        [~, TrainRealLable]=max(TrainFindR(:,i));
        TrainIdealResults(i,1) = TrainIdealLable;
        TrainRealResults(i,1) = TrainRealLable;
        if TrainRealLable == TrainIdealLable
            TrainClassifySuccess=TrainClassifySuccess+1;
        end
    end
    TrainAccuracy = TrainClassifySuccess/TrainNum;
    TrainAccsave(k) = TrainAccuracy;                           %   记录训练每次迭代准确率



    MinXita = min(min(min(OutputXita)));                %   寻找最小θ
    for j = 1:TrainNum                                  %   更新参数θ
        for l = 1:HiddenNum
            for m = 1:ClassNum
                DifC = TrainOutputC(j,m) - TrainIdealOutputC(j,m);
                DifR = TrainOutputR(j,m) - TrainIdealOutputR(j,m);
                Delta_Xita(l,m) = 2 * (1/((1-Alpha)*gamma(1-Alpha))) * (DifC*TrainInputHc(j,l) + DifR*TrainInputHr(j,l)*sgn(OutputXita(l,m))) * ((OutputXita(l,m) - MinXita)^(1-Alpha)) + Delta_Xita(l,m);
%                 Delta_Xita(l,m) = 2 * (DifC*TrainInputHc(j,l) + DifR*TrainInputHr(j,l)*sgn(OutputXita(l,m))) + Delta_Xita(l,m);
            end
        end
    end
    OutputXita = OutputXita - Yita*Delta_Xita;
    
 %%%%%%%%%  测试
    for j = 1:TestNum

        for l = 1:HiddenNum    % 输入层-隐藏层
            for n = 1:Inputdim
                TestInputZc(j,l) = InputWeight(n,l)*TestInputXC(j,n) + TestInputZc(j,l);
                TestInputZr(j,l) = abs(InputWeight(n,l))*TestInputXR(j,n) + TestInputZr(j,l);
            end
            TestInputZc(j,l) = TestInputZc(j,l)+ InputBias(1,l);
            ZCha = TestInputZc(j,l) - TestInputZr(j,l);
            ZHe = TestInputZc(j,l) + TestInputZr(j,l);
            TestInputHc(j,l) = (act(ZCha, 1) + act(ZHe, 1))/2;   % 修改编号更改激活函数
            TestInputHr(j,l) = (act(ZHe, 1) - act(ZCha, 1))/2;  % 修改编号更改激活函数
        end

        for m = 1:ClassNum    % 隐藏层-输出层
            for  l = 1:HiddenNum
                TestOutputC(j,m) = act(TestInputHc(j,l)*OutputXita(l,m), 0) + TestOutputC(j,m);   % 修改编号更改激活函数
                TestOutputR(j,m) = act(TestInputHr(j,l)*abs(OutputXita(l,m)), 0) + TestOutputR(j,m);   % 修改编号更改激活函数
            end
        end

    end

    TestMSE = 0;                                      %   训练均方误差
    for j = 1:TestNum
        for m = 1:ClassNum
            TestMSE = (TestOutputC(j,m) - TestIdealOutputC(j,m))^2 + (TestOutputR(j,m) - TestIdealOutputR(j,m))^2 + TestMSE;
        end
    end
    TestMSE = (1/TestNum)*TestMSE;
    TestMSEsave(k) = TestMSE;                           %   记录训练每次迭代均方误差
    

   TestClassifySuccess = 0;                                %   记录分类成功个数
    TestFindI = TestIdealOutputC';
    TestFindR = TestOutputC';
    TestIdealResults = zeros(TestNum,1);
    TestRealResults = zeros(TestNum,1);
    for i = 1 : TestNum
        [~, TestIdealLable]=max(TestFindI(:,i));
        [~, TestRealLable]=max(TestFindR(:,i));
        TestIdealResults(i,1) = TestIdealLable;
        TestRealResults(i,1) = TestRealLable;
        if TestRealLable == TestIdealLable
            TestClassifySuccess=TestClassifySuccess+1;
        end
    end
    TestAccuracy = TestClassifySuccess/TestNum;
    TestAccsave(k) = TestAccuracy;                           %   记录训练每次迭代准确率

end
toc;
FinalTime = toc;
%%%%%%%%%%%%%%%%%%%%%%%%% Do Your Picture %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% 训练误差和梯度范数
xian11 = figure(1000+gogo);%+gogo
plot(1:Iterations,TrainMSEsave);
hold on
xlabel("Number of Iterations");
ylabel("Training MSE");

xian12 = figure(2000+gogo);%+gogo
plot(1:Iterations,TestMSEsave);
hold on
xlabel("Number of Iterations");
ylabel("Testing MSE");

xian21 = figure(3000+gogo);%+gogo
plot(1:Iterations,TrainAccsave);
hold on
xlabel("Number of Iterations");
ylabel("Training Accuracy");

xian22 = figure(4000+gogo);%+gogo
plot(1:Iterations,TestAccsave);
hold on
xlabel("Number of Iterations");
ylabel("Testing Accuracy");

filename = [num2str(TrainAccuracy) '_' num2str(TestAccuracy) '_' num2str(FinalTime) '.mat'];
    save(filename);
end

%%%%%%%%%%%%%%%%%%%%%%%%% 超参数纪录 %%%%%%%%%%%%%%%%%%%%%%%
experimentals(gogo,1) = Yita;
experimentals(gogo,2) = TrainMSE;
experimentals(gogo,3) = TestMSE;
experimentals(gogo,4) = TrainAccuracy;
experimentals(gogo,5) = TestAccuracy;
experimentals(gogo,6) = FinalTime;



%%%%%%%%%%%%%%%%%%%%%%%%% Function Set %%%%%%%%%%%%%%%%%%%%%%%
function y1 = sgn(x1)  % 符号函数
    if x1>0
        y1 = 1;
    elseif x1<0
        y1 = -1;
    else 
        y1 = 0;
    end
end

function y2 = act(x2,type) 
    if type == 0        % 无激活函数 编号0
        y2 = x2;
    elseif type == 1        % sigmoid激活函数 编号 1
        y2 = 1 / (1 + exp(-x2));
    elseif type == 2     % ReLU激活函数 编号 2
        y2 = max(0,x2);
    end
end


