//

// Created by x00425595 on 2019/9/18.

//



#include "face_swap/kalman_filter.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/video/tracking.hpp>


namespace face_swap 
{
    MineKalmanFilter::MineKalmanFilter()
{

//        prev_x0 = 0

//        prev_y0 = 0

//        prev_x1 = 0

//        prev_y1 = 0



//        KF = cv::KalmanFilter(4, 2, 0)

//        KF.measurementMatrix = (cv::Mat_<float>(4, 2) << 1, 0, 0, 0, 0, 1, 0, 0);

//        KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1)

//        KF.processNoiseCov = kf_noise_coef * (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)



//        setIdentity(KF.measurementMatrix);

//        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));

//        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));

//        setIdentity(KF.errorCovPost, Scalar::all(1));

//        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));



        // Initialize the points of last frame

        for (int i = 0; i < 68; ++i) {

            last_object.push_back(cv::Point(0, 0));

        }

        // Initialize measurement points

        for (int i = 0; i < 68; i++) {

            kalman_points.push_back(cv::Point(0, 0));

        }

        // Initialize prediction points

        for (int i = 0; i < 68; i++) {

            predict_points.push_back(cv::Point(0, 0));

        }



        // Kalman Filter Setup (68 Points Test)

        stateNum = 272;

        const int measureNum = 136;



        cv::KalmanFilter KFF(stateNum, measureNum, 0);

        cv::Mat statet(stateNum, 1, CV_32FC1);

        cv::Mat processNoise(stateNum, 1, CV_32F);

        measurement = cv::Mat::zeros(measureNum, 1, CV_32F);



        // Generate a matrix randomly

        randn(statet, cv::Scalar::all(0), cv::Scalar::all(0));
        state = statet;



        // Generate the Measurement Matrix

        KF.transitionMatrix = cv::Mat::zeros(272, 272, CV_32F);

        for (int i = 0; i < 272; i++) {

            for (int j = 0; j < 272; j++) {

                if (i == j || (j - 136) == i) {

                    KFF.transitionMatrix.at<float>(i, j) = 1.0;

                } else {

                    KFF.transitionMatrix.at<float>(i, j) = 0;

                }

            }

        }



        //!< measurement matrix (H) Measurement Model

        cv::setIdentity(KFF.measurementMatrix);

        //!< process noise covariance matrix (Q)

        cv::setIdentity(KFF.processNoiseCov, cv::Scalar::all(1e-5));



        //!< measurement noise covariance matrix (R)

        cv::setIdentity(KFF.measurementNoiseCov, cv::Scalar::all(1e-1));



        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix

        cv::setIdentity(KFF.errorCovPost, cv::Scalar::all(1));



        cv::randn(KFF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));



        this->KF = KFF;

    }


    std::vector<cv::Point> MineKalmanFilter::getPredictPoints(std::vector<cv::Point> newPoints) {

        kalman_points = newPoints;



        // Kalman Prediction

        cv::Mat prediction = KF.predict();

        for (int i = 0; i < 68; i++) {

            predict_points[i].x = prediction.at<float>(i * 2);

            predict_points[i].y = prediction.at<float>(i * 2 + 1);

        }



        // Update Measurement

        for (int i = 0; i < 136; i++) {

            if (i % 2 == 0) {

                measurement.at<float>(i) = (float) kalman_points[i / 2].x;

            } else {

                measurement.at<float>(i) = (float) kalman_points[(i - 1) / 2].y;

            }

        }


        measurement += KF.measurementMatrix * state;


        // Correct Measurement

        KF.correct(measurement);

        return predict_points;

    }

}
