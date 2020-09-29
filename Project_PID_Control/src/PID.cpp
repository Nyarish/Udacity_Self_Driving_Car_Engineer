#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  PID::Kp = Kp_;
  PID::Ki = Ki_;
  PID::Kd = Kd_;
  
  PID::p_error = 0;
  PID::i_error = 0;
  PID::d_error = 0;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  float pre_cte = p_error;
  PID::d_error = cte - pre_cte;
  PID::p_error = cte;
  PID::i_error += cte;
  

}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  double total_error = -PID::Kp * PID::p_error - PID::Ki * PID::i_error - PID::Kd * PID::d_error;
  
  return total_error;  // TODO: Add your total error calc here!
}