# **PID Controller Project**


---

**PID Controller**

The goal of this project is to implement a PID controller in C++ to maneuver the vehicle around the track!
The simulator will provide you the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle.


[//]: # (Image References)

[image1]: ./img_Folder/img.png "Throttle "


---

### Reflection

Describe the effect each of the P, I, D components had in your implementation.PID stands for Proportional-Integral-Derivative.

Here is the final [video](https://share.icloud.com/photos/0ctTnnKjjVquueToX6fAJq6_Q#Nairobi)


### 1. The P - Component 

This controller requires biasing or manual reset when used alone. This is because it never reaches the steady state condition. It provides stable operation but always maintains the steady state error.In the [video](https://share.icloud.com/photos/0KAk9JjaIRREtS5K_9BrYE6YQ#Nairobi) oscillation behavior is clearly observed when ONLY used. 

---

### 2. The I - Component 

Due to limitation of p-controller where there always exists an offset between the process variable and set point, I-controller is needed, which provides necessary action to eliminate the steady state error.  It integrates the error over a period of time until error value reaches to zero. It holds the value to final control device at which error becomes zero.

From the [video](https://share.icloud.com/photos/0LzwLmCIvqnU3-VT33Uf4GjbA#Nairobi) you can see that with I-controller ONLY the vehicle can correct the steering angle error. But the correction seems not fast enough and stuggles on the corners. 

### 3. The D - Component 

I-controller doesn’t have the capability to predict the future behavior of error. So it reacts normally once the set point is changed. D-controller overcomes this problem by anticipating future behavior of the error. Its output depends on rate of change of error with respect to time, multiplied by derivative constant. It gives the kick start for the output thereby increasing system response.

In our case only using the D-controller leaves the car to crash. it has the same effect as the I-controller.

See [video](https://share.icloud.com/photos/0nBYYyFVQBYrzyfwASwI3eE_A#Nairobi). 


### 4. Describe how the final hyperparameters were chosen.

Trial and Error Method: It is a simple method of PID controller tuning. While system or controller is working, we can tune the controller. In this method, first we have to set Ki and Kd values to zero and increase proportional term (Kp) until system reaches to oscillating behavior. Once it is oscillating, adjust Ki (Integral term) so that oscillations stops and finally adjust D to get fast response.

In my case i choose the parameter according the simulation result. After trial and erorr and consultations with classmates i ended up with the following values;

` Kp = 0.06
  Ki = 0.00031
  Kd = 1.29`


I also played around with the throttle and speed to enable stay on the track.

![alt text][image1]


Reference :

https://www.elprocus.com/the-working-of-a-pid-controller/

https://github.com/tiaotiaohum/CarND-PID-Control-Project#component-p






```python

```
