

Pc = 200*10^5;%pa
gamma = 1.2; %as assumed for design of parabolic nozzle

Dt = 0.8;
Rc = 0.5*Dt; %curvature rad as assumed for parabolic nozzle design
AR = 50; %assumed area ratio
De = sqrt(AR*Dt^2);
At = (3.14*Dt^2)/4;

A = [ 0.85, 1.29, 1.63, 2.22,2.66, 3.4, 3.9, 4.7, 5.5, 6.5, 7.3, 8, 8.9, 9.7, 10.6, 11.3, 12.1, 13, 13.9, 14.7, 15.5, 16.3, 17, 17.9, 18.56, 19.3, 20, 20.8, 21.4, 22, 22.6, 23.25, 23.6, 24.1, 24.6, 25.1];%between length 0 to 4m at 0.2m interval
L = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2];
%At O/F = 2.4 for RP-1/O2 at frozen cond.
Mc = 1.1327;
K = 3.6862; %CONDUCTIVITY
Pr = 0.6389; 
%for AR = 35 from CEA o/p
Cp = 2.0792; 
Ivac = 3347.8; % in m/s
Isp = 3228.7; % in m/s
CSTAR = 1788.3; %characteristic velocity
Cf = 1.8054;
hlist = [];

%upon finding h we move on to find deltaT

F = 100000;%N
ueq = 3288.25; %0.5*(Ivac+Isp)
DelV = 3000; %m/s
MR = 0.40; %exp(-DelV/ueq)
initial_mass = 30000; 
fin_mass = MR*initial_mass;
prop_mass = initial_mass-fin_mass;
mdot = F/ueq;
Twall = 800;
Tch = 3600;

for ii = 1:length(A)
    Area = A(ii);
    h = 0.026*((Mc.^0.2*Cp)/(Pr.^0.6*Dt.^0.2))*((Pc/CSTAR).^0.8)*(At/Area).^0.9;
    hlist(end+1) = h;
  
end
%disp(hlist);
for h = 1:length(hlist)
    q = hlist(h)*(Tch - Twall)*10^-3;
    D= sqrt(4*A(h)/3.14);
    Qdot = 3.14*D*q*0.2; %as we consider areas at 0.2m intervals
    %disp(Qdot);%in MW/m2
    delT = Qdot/(mdot*Cp);
    %disp(delT);
    Nu = 0.03*((Pc*Dt)/(CSTAR*Mc)).^0.8*(L(h)/Dt).^0.8*(Dt/D).^1.6*(Pr)^0.33;
    %disp(Nu);
    num = Nu*K; %num=hc*d
    %opdisp(num);
    qcool = num*delT*10^-3;
    %disp(qcoold);
    %disp(Qdot);
    n= Qdot/(3.14*0.2*qcool);%n = N*d^2
    syms x positive
    eqn = (2.512*x^3)*(2.612*x^2)/(x+0.04) == n;
    d_ch = vpasolve(eqn,x,[0 Inf]);
    %disp(d_ch);
    N = n/d_ch^2;
    disp(N);
   
end

