% note bias added manually below
X = [1 2 4;1 1 1; 1 2 2; 1 1.4 0.5]
Theta1 = [1 2 3; 2 3 4; 3 4 5; 4 5 6; 5 6 7]
Theta2 = [6 5 4 3 2 1]
a2 = [ones(size(X,1),1) sigmoid(X*Theta1')]
a3 = sigmoid(a2*Theta2')
