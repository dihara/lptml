d = 2;
n = 100;

P = zeros(d,0);
L = zeros(1,0);
t = 0;
for k=1:n
    a = randi(100);
    if (a>50)
        P = [P randn(d, 1)+[3;0]];
        L = [L 0];
        t = t + 1;
    else
        P = [P randn(d, 1)+[-3;0]];
        L = [L 1];
    end
end

x = P;
y = L;
S = [];
S5 = [];
S10 = [];
S15 = [];
S20 = [];
S30 = [];
S40 = [];
S50 = [];
S60 = [];
S70 = [];
S80 = [];
S90 = [];
S100 = [];
D = [];
D5 = [];
D10 = [];
D15 = [];
D20 = [];
D30 = [];
D40 = [];
D50 = [];
D60 = [];
D70 = [];
D80 = [];
D90 = [];
D100 = [];

for k=1:n
    a = randi(100);
    if (a<=5)
        y5(k) = 1-L(k);
    else
        y5(k) = L(k);
    end
    
    if (a<=10)
        y10(k) = 1-L(k);
    else
        y10(k) = L(k);
    end   

    if (a<=15)
        y15(k) = 1-L(k);
    else
        y15(k) = L(k);
    end   
    
    if (a<=20)
        y20(k) = 1-L(k);
    else
        y20(k) = L(k);
    end       
    
    for j=k+1:n
        a = randi(100);
        if y(k)==y(j)
            if (a<=5)
                D5 = [D5 [j;k]];
            else
                S5 = [S5 [j;k]];
            end
            
            if (a<=10)
                D10 = [D10 [j;k]];
            else
                S10 = [S10 [j;k]];
            end   

            if (a<=15)
                D15 = [D15 [j;k]];
            else
                S15 = [S15 [j;k]];
            end
            
            if (a<=20)
                D20 = [D20 [j;k]];
            else
                S20 = [S20 [j;k]];
            end
            
            if (a<=30)
                D30 = [D30 [j;k]];
            else
                S30 = [S30 [j;k]];
            end            
            
            if (a<=40)
                D40 = [D40 [j;k]];
            else
                S40 = [S40 [j;k]];
            end   
            
            if (a<=50)
                D50 = [D50 [j;k]];
            else
                S50 = [S50 [j;k]];
            end   
            
            if (a<=60)
                D60 = [D60 [j;k]];
            else
                S60 = [S60 [j;k]];
            end   
            
            if (a<=70)
                D70 = [D70 [j;k]];
            else
                S70 = [S70 [j;k]];
            end     
            
            if (a<=80)
                D80 = [D80 [j;k]];
            else
                S80 = [S80 [j;k]];
            end   
            
            if (a<=90)
                D90 = [D90 [j;k]];
            else
                S90 = [S90 [j;k]];
            end           
            
            S = [S [j;k]];
        else
            if (a<=5)
                S5 = [S5 [j;k]];
            else
                D5 = [D5 [j;k]];
            end
            
            if (a<=10)
                S10 = [S10 [j;k]];
            else
                D10 = [D10 [j;k]];
            end   

            if (a<=15)
                S15 = [S15 [j;k]];
            else
                D15 = [D15 [j;k]];
            end
            
            if (a<=20)
                S20 = [S20 [j;k]];
            else
                D20 = [D20 [j;k]];
            end            

            if (a<=30)
                S30 = [S30 [j;k]];
            else
                D30 = [D30 [j;k]];
            end   
            
            if (a<=40)
                S40 = [S40 [j;k]];
            else
                D40 = [D40 [j;k]];
            end   
            
            if (a<=50)
                S50 = [S50 [j;k]];
            else
                D50 = [D50 [j;k]];
            end          
            
            if (a<=60)
                S60 = [S60 [j;k]];
            else
                D60 = [D60 [j;k]];
            end   
            
            if (a<=70)
                S70 = [S70 [j;k]];
            else
                D70 = [D70 [j;k]];
            end   
            
            if (a<=80)
                S80 = [S80 [j;k]];
            else
                D80 = [D80 [j;k]];
            end   
            
            if (a<=90)
                S90 = [S90 [j;k]];
            else
                D90 = [D90 [j;k]];
            end                        
            D = [D [j;k]];
        end
    end
end

D100 = S
S100 = D

%G = [1, 0; 0, 40];
%PD1 = G*P;

%G = randn(2, 2)
%PR1 = G*P
%G = randn(2, 2)
%PR2 = G*P
%G = randn(2, 2)
%PR3 = G*P

%figure('Name', 'Original data')
%plot(P(1,:), P(2,:), '.');
% figure('Name', 'Transformation diagonal matrix')
% plot(PD1(1,:), PD1(2,:), '.');
% figure('Name', 'Random linear transformation 1')
% plot(PR1(1,:), PR1(2,:), '.');
% figure('Name', 'Random linear transformation 2')
% plot(PR2(1,:), PR2(2,:), '.');
% figure('Name', 'Random linear transformation 3')
% plot(PR3(1,:), PR3(2,:), '.');


Gy = [1, 0; 0, 40];
xt = Gy*P;
%figure('Name', 'Y scaling');
%axis equal
%plot(PR4(1,:), PR4(2,:), '.');