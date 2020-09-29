/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  std::default_random_engine generator;
  
  for(int i=0; i<num_particles; i++){
    Particle parti;
    parti.id = i;
    parti.x = dist_x(generator);
    parti.y = dist_y(generator);
    parti.theta = dist_theta(generator);
    parti.weight = 1;
    particles.push_back(parti);
    
  }
  is_initialized = true;
  std::cout<< "init state is: "<<is_initialized <<std::endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine generator;
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);
  
  for(int i=0; i<num_particles; i++){
    if (fabs(yaw_rate) <0.0000001){
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + noise_x(generator);
      
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + noise_y(generator);
      particles[i].theta += yaw_rate * delta_t + noise_theta(generator);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + noise_x(generator);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + noise_y(generator);
      particles[i].theta += yaw_rate * delta_t + noise_theta(generator);
    }
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(int i=0; i<(observations.size()); i++){
    double min_distance = std::numeric_limits<float>::max();
    for(int j=0; j<int(predicted.size()); j++){
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(distance < min_distance){
        min_distance = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(int i=0; i < particles.size(); i++){
    particles[i].weight = 1.0;
  }
  
  for(int i=0; i < int(particles.size()); i++){
    vector<LandmarkObs> predi_valid;
    
    for(int j=0; j<int(map_landmarks.landmark_list.size()); j++){
      double distance_parti_landm = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if(distance_parti_landm <= sensor_range){
        predi_valid.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f });
      }
    }
    
    vector<LandmarkObs> tran_xy_set;
    
    for(int n=0; n<int (observations.size());n++){
      LandmarkObs tran_xy;
      tran_xy.id = observations[n].id;
      tran_xy.x = observations[n].x * cos(particles[i].theta) - observations[n].y * sin(particles[i].theta) + particles[i].x;
       tran_xy.y = observations[n].x * sin(particles[i].theta) + observations[n].y * cos(particles[i].theta) + particles[i].y;
      tran_xy_set.push_back(tran_xy);
      
    }
    
    ParticleFilter::dataAssociation(predi_valid, tran_xy_set);
    
    for(int m=0; m<int(tran_xy_set.size()); m++){
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(tran_xy_set[m].id-1);
      double x_diff = pow(tran_xy_set[m].x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_diff = pow(tran_xy_set[m].y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double weight_temp = exp(-(x_diff + y_diff)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      particles[i].weight *= weight_temp;
    }
    
  predi_valid.clear();
  tran_xy_set.clear();
    
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  vector<double> p_weights;
  
  for (int i=0; i < particles.size(); i++){
    p_weights.push_back(particles[i].weight);
  }
  std::discrete_distribution<> weight_dist(p_weights.begin(), p_weights.end());
  vector<Particle> resa_parti;
  
  int index;
  
  for(int i=0; i<num_particles; i++){
    index= weight_dist(gen);
    resa_parti.push_back(particles[index]);
  }
  particles = resa_parti;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}