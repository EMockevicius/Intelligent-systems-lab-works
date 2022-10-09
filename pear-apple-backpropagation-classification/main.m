clear;clc;

%Classification using perceptron

%Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%Calculate for each image, colour and roundness
%For Apples
%1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
%2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
%3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
%4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
%5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
%6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
%7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
%8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
%9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1];

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% calculate wieghted sum with randomly generated parameters
%v1 = <...>; % write your code here
% calculate current output of the perceptron

%study 1    
v1=x1(1)*w1+x2(1)*w2+b*1;

if v1 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e1 = T(1) - y;

%-----------------------
%study 2
v2=x1(2)*w1+x2(2)*w2+b*1;

if v2 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e2 = T(2) - y;

%-----------------------
%study 3
v3=x1(3)*w1+x2(3)*w2+b*1;

if v3 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e3 = T(3) - y;

%-----------------------
%study 4
v4=x1(4)*w1+x2(4)*w2+b*1;

if v4 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e4 = T(4) - y;

%-----------------------
%study 5
v5=x1(5)*w1+x2(5)*w2+b*1;

if v5 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e5 = T(5) - y;

%-------------------------------------

% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);
err = [e1;e2;e3;e4;e5];

% mokymo zingsnis
mz = 0.02;

% write training algorithm

while e ~= 0 % executes while the total error is not 0 
	
    for j = 1:20
        for k = 1:5
     w1=w1+mz*err(k)*x1(k);
     w2=w2+mz*err(k)*x2(k);
     b=b+mz*err(k)*1;
        end
    end
    
%study 1    
v1=x1(1)*w1+x2(1)*w2+b*1;

if v1 > 0
	y1 = 1;
else
	y1 = -1;
end
% calculate the error
e1 = T(1) - y1;

%-----------------------
%study 2
v2=x1(2)*w1+x2(2)*w2+b*1;

if v2 > 0
	y2 = 1;
else
	y2 = -1;
end
% calculate the error
e2 = T(2) - y2;

%-----------------------
%study 3
v3=x1(3)*w1+x2(3)*w2+b*1;

if v3 > 0
	y3 = 1;
else
	y3 = -1;
end
% calculate the error
e3 = T(3) - y3;

%-----------------------
%study 4
v4=x1(4)*w1+x2(4)*w2+b*1;

if v4 > 0
	y4 = 1;
else
	y4 = -1;
end
% calculate the error
e4 = T(4) - y4;

%-----------------------
%study 5
v5=x1(5)*w1+x2(5)*w2+b*1;

if v5 > 0
	y5 = 1;
else
	y5 = -1;
end
% calculate the error
e5 = T(5) - y5;

err = [e1;e2;e3;e4;e5];

e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);

vv = [v1;v2;v3;v4;v5];
yy = [y1;y2;y3;y4;y5];


%applying weights for new imputs

%adding new imputs
x01=[hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x02=[metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];

%checking A4    
v1=x01(1)*w1+x02(1)*w2+b*1;

if v1 > 0
	y1 = 1;
else
	y1 = -1;
end

%checking A5
v2=x01(2)*w1+x02(2)*w2+b*1;

if v2 > 0
	y2 = 1;
else
	y2 = -1;
end

%checking A6
v3=x01(3)*w1+x02(3)*w2+b*1;

if v3 > 0
	y3 = 1;
else
	y3 = -1;
end

%checking A7
v4=x01(4)*w1+x02(4)*w2+b*1;

if v4 > 0
	y4 = 1;
else
	y4 = -1;
end

%checking A8
v5=x01(5)*w1+x02(5)*w2+b*1;

if v5 > 0
	y5 = 1;
else
	y5 = -1;
end

%checking A9    
v6=x01(6)*w1+x02(6)*w2+b*1;

if v6 > 0
	y6 = 1;
else
	y6 = -1;
end

%checking P3    
v7=x01(7)*w1+x02(7)*w2+b*1;

if v7 > 0
	y7 = 1;
else
	y7 = -1;
end

%checking P4   
v8=x01(8)*w1+x02(8)*w2+b*1;

if v8 > 0
	y8 = 1;
else
	y8 = -1;
end

% Displaying results

yy=[y1;y2;y3;y4;y5;y6;y7;y8]

end

% Apples on plot
X1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9];
X2=[metric_A1 metric_A2 metric_A3 metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9];

% Pears on plot
X3=[hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4];
X4=[metric_P1 metric_P2 metric_P3 metric_P4];

plot(X1,X2,'ro',X3,X4,'gd')
grid on;
hold on

% Decision boundary w1x + w2y + b = 0
% w1x + b = 0 and w2y + b = 0
% thus x = -b/w1 and y=-b/w2

X=[0, -(b/w1)];
Y=[-(b/w2),0];
plot(X,Y);
legend('obuoliai', 'kriaušės', 'decision boundary');