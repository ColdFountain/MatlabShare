clear;clc;
Yita = 0.0003;            %  起始学习率
FailRate = 0.01;        %  衰减率
HiddenNum = 26;     %  隐节点数

expnum = 1;            % 实验室次数
Iterations = 1000;       %  迭代步数
experimentals = zeros(expnum,3);
for gogo = 1:expnum
%%%%%%%%%%%%%%%%%%%%%%%%% Adjustment parameters %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Learning Rate
% if gogo>1
%     Yita = Yita/(1+FailRate*gogo);
% end
% HiddenNum = HiddenNum + 1;
%%%%%%%%%%%%%%%%%%%%%%%%% Training Input Datas %%%%%%%%%%%%%%%%%%%%%%%
% Fai = sqrt(5);
% GuassA=1/(Fai*sqrt(2*pi));
% GuassB = 0;
% GuassC = Fai;


TrainNum = 200;        %输入训练样本数量
Inputdim = 1;  %输入训练样本维度
ClassNum = 1;  %输出维度
% TrainInputXC=sort(10*rand(TrainNum,Inputdim) - 5);%  随机产生区间输入<中心>数据fx1
% TrainInputXC=sort(4*rand(TrainNum,Inputdim) - 2);%  随机产生区间输入<中心>数据  
TrainInputXC=sort(2*rand(TrainNum,Inputdim) - 1);%  随机产生区间输入<中心>数据  

for i=1:TrainNum  
    if TrainInputXC(i,Inputdim)==0
        TrainInputXC(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end
TrainInputXR=0.1*rand(TrainNum,1);  %  随机产生区间输入<半径>数据 
for i=1:TrainNum  
    if TrainInputXR(i,Inputdim)==0
        TrainInputXR(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end

% TrainIdealOutputC = sin(TrainInputXC).*(TrainInputXC.^(-1));    %  <中心>数据理想输出，函数公式sin(x)/x fx1

%    TrainIdealOutputC(i,1) = 1/(1+exp(-TrainInputXC(i,1)));     %  fx2
% TrainIdealOutputC(i,1) = GuassA*exp(-((TrainInputXC(i,1)-GuassB)^2)/(2*(GuassC^2)));   % fx3
for i=1:TrainNum
TrainIdealOutputC(i,1) = 1/(1+exp(-TrainInputXC(i,1)));
end
TrainIdealOutputR = 0.05*rand(TrainNum,1)+0.05;   %  <半径>数据理想输出，[0.05,0.1]公式 r = a + (b-a)*rand(m,n)其中[a,b]是范围

TestNum = TrainNum; %输入训练样本数量
% TestInputXC=sort(10*rand(TestNum,Inputdim) - 5);%  随机产生区间输入<中心>数据  
% TestInputXC=sort(4*rand(TestNum,Inputdim) - 2);%  随机产生区间输入<中心>数据 
TestInputXC=sort(2*rand(TestNum,Inputdim) - 1);%  随机产生区间输入<中心>数据  


for i=1:TestNum  
    if TestInputXC(i,Inputdim)==0
        TestInputXC(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end
TestInputXR=0.1*rand(TestNum,1);  %  随机产生区间输入<半径>数据 
for i=1:TestNum  
    if TestInputXR(i,Inputdim)==0
        TestInputXR(i,Inputdim)=0.000001;         %   防止出现0数据影响计算
    end
end

% TestIdealOutputC = sin(TestInputXC).*(TestInputXC.^(-1));    %  <中心>数据理想输出，函数公式sin(x)/x  fx1
%    TestIdealOutputC(i,1) = 1/(1+exp(-TestInputXC(i,1)));     %  fx2
% TestIdealOutputC(i,1) = GuassA*exp(-((TestInputXC(i,1)-GuassB)^2)/(2*(GuassC^2)));   % fx3

for i=1:TestNum
TestIdealOutputC(i,1) = 1/(1+exp(-TestInputXC(i,1)));
end

TestIdealOutputR = 0.05*rand(TestNum,1)+0.05;   %  <半径>数据理想输出，[0.05,0.1]公式 r = a + (b-a)*rand(m,n)其中[a,b]是范围

%%%%%%%%%%%%%%%%%%%%%%%%% Parameters Set %%%%%%%%%%%%%%%%%%%%%%%


TrainMSEsave1 = zeros(1,Iterations);                         %   存储训练每次迭代的均方误差
TestMSEsave1 = zeros(1,Iterations);                    %   存储训练每次迭代的梯度范数
MinError = 0.0000001;    %  最小误差
InputWeight = rand(Inputdim, HiddenNum)*2-1;     %   输入层权重W 总体[-1, 1] 随机 rand(隐节点数, 输入样本数量)
InputBias = rand(1, HiddenNum);           %   阈值b[0, 1]
OutputXita = rand(HiddenNum, ClassNum)*2-1;      %   输出层权重θ 总体[-1, 1] 随机 rand(隐节点数, 输入样本数量)
Delta_Xita = OutputXita*0;          %   每次迭代θ变值

%%%%%%%%%%%%%%%%%%%%%%%%% add Alpha %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Alpha = 9/11;             %  分数阶阶数


%%%%%%%%%%%%%%%%%%%%%%%%% Training Datas %%%%%%%%%%%%%%%%%%%%%%%
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

end

%%%%%%%%%%%%%%%%%%%%%%%%% Do Your Picture %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% 训练误差和梯度范数
xian = figure(100+gogo);%+gogo
set(gcf, 'unit', 'centimeters', 'position', [10 10 15 12]);
%二图各一线
% subplot(1,2,1)
plot(1:300,TrainMSEsave(1:300));
hold on
xlabel("Number of Iterations");
ylabel("Training MSE");
% subplot(1,2,2)
% plot(1:Iterations,TrainNormBeta);
% xlabel("Number of Iterations");
% ylabel("Norm of Gradient");
%一图二线
% plot(1:Iterations,TrainMSEsave,'r');
% hold on
% plot(1:Iterations,TrainNormBeta,'b');
% xlabel("Number of Iterations");
% legend('Training MSE','Norm of Gradient');

%%%%%%%%%%%%%%% Train区域散点曲线图
dian = figure(200+gogo);%+gogo
subplot(1,2,1);
IO = TrainIdealOutputC;
IOup = (TrainIdealOutputC + TrainIdealOutputR)';
IOdown = (TrainIdealOutputC - TrainIdealOutputR)';
RO = TrainOutputC;
plot(TrainInputXC,IO,'Color',[0.8500 0.3250 0.0980],'LineWidth', 1);
hold on
plot(TrainInputXC,IOup,'Color',[0.6010 0.7450 0.9330],'LineWidth', 1);
hold on
plot(TrainInputXC,IOdown,'Color',[0.3010 0.7450 0.9330],'LineWidth', 1);
hold on
x = TrainInputXC';
h=fill([x,fliplr(x)],[IOup,fliplr(IOdown)],'black');
set(h,'edgealpha',0,'facealpha',0.1); 
sz = mean(TrainOutputR) + 10;
scatter(TrainInputXC,RO,sz,[0.4660 0.6740 0.1880],"filled")

xlabel("Number of Training Input");
ylabel("Training OutPut");
legend('TrainIdealOutput','TrainRealOutputMean');
legend('TrainIdealOutputCenter','TrainIdealOutputHigh','TrainIdealOutputLow','TrainIdealOutputInterval','TrainRealOutput');

%%%%%%%%%%%%%%% Test区域散点曲线图
subplot(1,2,2);
IO = TestIdealOutputC;
IOup = (TestIdealOutputC + TestIdealOutputR)';
IOdown = (TestIdealOutputC - TestIdealOutputR)';
RO = TestOutputC;
plot(TestInputXC,IO,'Color',[0.8500 0.3250 0.0980],'LineWidth', 1);
hold on
plot(TestInputXC,IOup,'Color',[0.6010 0.7450 0.9330],'LineWidth', 1);
hold on
plot(TestInputXC,IOdown,'Color',[0.3010 0.7450 0.9330],'LineWidth', 1);
hold on
x = TestInputXC';
h=fill([x,fliplr(x)],[IOup,fliplr(IOdown)],'black');
set(h,'edgealpha',0,'facealpha',0.1); 
sz = mean(TestOutputR) + 10;
scatter(TestInputXC,RO,sz,[0.4660 0.6740 0.1880],"filled")

xlabel("Number of Testing Input");
ylabel("Testing OutPut");
legend('TestIdealOutput','TestRealOutputMean');
legend('TestIdealOutputCenter','TestIdealOutputHigh','TestIdealOutputLow','TestIdealOutputInterval','TestRealOutput');

% filename = [num2str(gogo) '.mat'];
% save(filename);
end


experimentals(gogo,1) = Yita;
experimentals(gogo,2) = TrainMSE;
experimentals(gogo,3) = TestMSE;


%%%%%%%%%%%%%%%%%%%%%%%%% 超参数纪录 %%%%%%%%%%%%%%%%%%%%%%%

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


