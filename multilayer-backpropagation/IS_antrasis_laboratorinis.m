clc;clear;close all;

%pridėti papildomą neuroną, jei neveiks, tuomet sluoksnį;

% ----------------------initial values----------------------------

% 2 inputs; 4 perceptrons in hidden layers(sigmoid); 1 output;

% N - inputs F - number of iterations; n - learning rate; 
% Wxyz weights x -layer y - from z - to; bias
N = 400;
F = 100000;
n = 0.1;

%assigning random weight and bias values
w111 = rand(1); % for y1
w112 = rand(1);
b11 = rand(1);

w121 = rand(1); % for y2
w122 = rand(1);
b12 = rand(1);

w131 = rand(1); % for y3
w132 = rand(1);
b13 = rand(1);

w141 = rand(1); % for y4
w142 = rand(1);
b14 = rand(1);

w211 = rand(1); % output layer
w212 = rand(1);
w213 = rand(1);
w214 = rand(1);
b21 = rand(1);

%20 random values for both inputs X1 and X2
for i = 1:N
    X1 = rand(1,N);
    X2 = rand(1,N);
end

% target formula d
%d = (1+0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2; given
d = (0.4 * tanh(pi * X1 / 0.6) + 1 + 0.6 * sin(X2 * 0.2 / pi));
        % ----------------------training----------------------------

for i = 1:F
    for j = 1:N
        
        % ----------------------forward-pass----------------------------
        
        % finding sigmoids
        v11 = X1(j) * w111 + X2(j) * w112 + b11;
        v12 = X1(j) * w121 + X2(j) * w122 + b12;
        v13 = X1(j) * w131 + X2(j) * w132 + b13;
        v14 = X1(j) * w141 + X2(j) * w142 + b14;
        
        % finding hidden layer outputs
       y11 = 1/(1+exp(-v11));       % 1/(1+exp(v11)) or other activation function depending on network
       y12 = 1/(1+exp(-v12));
%        y11 = v11*pi;                % different type of activation function
%        y12 = tanh(v12);
        y13 = 1/(1+exp(-v13));
        y14 = 1/(1+exp(-v14));
        
        % finding output layer output
        y21 = y11*w211 + y12*w212 + y13*w213 + y14*w214 + b21;

% if its sigmoid and not sum
% v21 = y11*w211 + y12*w212 + y13*w213 + y14*w214 + b21;
% y21 = 1/(1+exp(y21)); 
        
        % calculating error
        e = d(j) - y21;
        
        % ----------------------back-pass----------------------------
        
        % updating output layer weights
        w211 = w211 + n * e * y11;  
        w212 = w212 + n * e * y12;  
        w213 = w213 + n * e * y13;  
        w214 = w214 + n * e * y14;  
        b21 = b21 + n * e;
        
        % finding gradient change for each perceptron 
        % y1*(1-y1)  ==  diff(y1) 
        g1 = y11*(1-y11) * e * w211;
        g2 = y12*(1-y12) * e * w212;
        g3 = y13*(1-y13) * e * w213;
        g4 = y14*(1-y14) * e * w214;
        
        % updating input layer weights
        w111 = w111 + n * g1 * X1(j); %for y1
        w112 = w112 + n * g1 * X2(j);
        b11 = b11 + n * g1;
        
        w121 = w121 + n * g2 * X1(j); %for y2
        w122 = w122 + n * g2 * X2(j); 
        b12 = b12 + n * g2;
        
        w131 = w131 + n * g3 * X1(j); %for y3
        w132 = w132 + n * g3 * X2(j); 
        b13 = b13 + n * g3;
        
        w141 = w141 + n * g4 * X1(j); %for y4
        w142 = w142 + n * g4 * X2(j);         
        b14 = b14 + n * g4;
        end
end

% ----------------------using trained backpropagation neural network----------------------------

for j = 1:N
        % finding sigmoids
        v11 = X1(j) * w111 + X2(j) * w112 + b11;
        v12 = X1(j) * w121 + X2(j) * w122 + b12;
        v13 = X1(j) * w131 + X2(j) * w132 + b13;
        v14 = X1(j) * w141 + X2(j) * w142 + b14;
        
        % finding hidden layer outputs
        y11 = 1/(1+exp(-v11));         % 1/(1+exp(v11)) or other activation function depending on network
        y12 = 1/(1+exp(-v12));
%         y11 = v11*pi;                   % different type of activation function
%         y12 = tanh(v12);
        y13 = 1/(1+exp(-v13));
        y14 = 1/(1+exp(-v14));
        
        % finding output layer output
        y21(j) = y11*w211 + y12*w212 + y13*w213 + y14*w214 + b21;
end

% ----------------------Plotting answer----------------------------
figure('name','Rezultatas');
plot3(X1,X2,d,'go', X1,X2,y21,'r+')
legend ({'inputs', 'outputs'})
grid on;

