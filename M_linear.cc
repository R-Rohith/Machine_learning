#include<iostream>
/*
 * YAY!! Completed a program to fit a set of points to a linear function. This is a very basic program, and just a small stepping stone to Machine Learning.
 * Next make a program for logistic regression.
 */
int main()              //No main fn.,....b'cause MACRO!
{
    float x[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21},y[]={2.25,3.75,4.2,5,6.9,7.1,8.3,9,9.9,10.8,11,12,13,14,15,16,17,18,19,20,21},w=0,b=0,stepb=0.1,stepw=0.1,prev_loss_w,prev_loss_b,curr_loss_b,curr_loss_w;
    float calc_loss(float*,float*,int,float,float);
    //y=w*x+b               //Linear Regression

    FILE *f;
    f=fopen("plot.txt","w");

    prev_loss_w=calc_loss(x,y,21,w,b);
    prev_loss_b=prev_loss_w;
    for(int epoch=0;epoch<2000;epoch++)
    {
    curr_loss_w=calc_loss(x,y,21,w+stepw,b);
    if (curr_loss_w<prev_loss_w)
        w+=stepw;
    else if (curr_loss_w>prev_loss_w)
    {
        stepw*=(-0.5);
        w+=stepw;
    }                                                        //APPARENTLY RUNNING THE LEARNING INDEPENDENTLY IS BETTER....OR IS IT?...GOTTA CHECK: It's true because
    prev_loss_w=curr_loss_w;                                 //the change in loss because of new wieght also changes the loss of bias.
    //Need a code to change step according to w
    //fprintf(f,"%f %f\n",w,curr_loss_w);
    std::cout<<"\nw= "<<w<<" b= "<<b<<" wloss= "<<curr_loss_w;
    }
    for(int epoch=0;epoch<2000;epoch++)
    {
    curr_loss_b=calc_loss(x,y,21,w,b+stepb);
    if (curr_loss_b<prev_loss_b)
        b+=stepb;
    else if (curr_loss_b>prev_loss_b)
    {
        stepb*=(-0.5);
        b+=stepb;
    }
    prev_loss_b=curr_loss_b;
    //Need a code to change step according to w
    //fprintf(f,"%f %f\n",w,curr_loss_w);
    std::cout<<"w= "<<w<<"b= "<<b<<" bloss= "<<curr_loss_b;
    }
    return 0;
}
float calc_loss(float *x,float *y,int size,float w, float b)
{
    float loss=0,t;
    for(int i=0;i<size;i++)
        {
            t=y[i]-w*x[i]-b;
            loss+=t*t;
        }
    loss=loss/size;
    return loss;
}
