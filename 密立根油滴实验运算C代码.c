#include <stdio.h>

#include <math.h>

int main()

{

float q, t, r, u, x, x1, x2, n1, e1;

float p1 = 981,g = 9.80,p0=1.293,l = 2.00e-3,b = 8.23e-3,p = 0.101e6,d = 5.00e-3,n = 1.83e-5;

int n2;

double e2, w,e0=1.602e-19;

printf("先输入时间,再输入电压\n");

scanf("%f %f",&t,&u);

r = sqrt(9 * n*l / (2 * (p1-p0)*g*t));

x1 = ((n*l) / (t*(1 + b / (p*r))));

x2 = x1*x1*x1;

x = sqrt(x2);

q = 18 * 3.14*x*d / (sqrt(2 * (p1-p0)*g)*u);

n1=q/(1.6e-19);

n2=(int)n1;

if(n1-n2>=0.5)

n2=n2+1;

else if(n1-n2<0.5||n1-n2>=0)

n2=n2;

else

printf("error");

e2=q/n2;

w=(e2-e0)/e0*100;

printf("半径r= %e\n",r);

printf("电量q= %e\n",q);

printf("油滴所带基本电荷n= %d\n",n2);

printf("实验数据所得电荷量e= %e\n",e2);

printf("实验误差w= %f \n",w);

return 0;

}

