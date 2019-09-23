#ifndef FACE_SWAP_KALMAN_FILTER_H
#define FACE_SWAP_KALMAN_FILTER_H


#include <list>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

namespace face_swap
{
    class MineKalmanFilter
    {
    protected:
//        int prev_x0;
//        int prev_x1;
//        int prev_y0;
//        int prev_y1;

        std::vector<cv::Point> last_object;
        std::vector<cv::Point> kalman_points;
        std::vector<cv::Point> predict_points;
        cv::Mat measurement;
        cv::Mat state;
	int stateNum;
        cv::KalmanFilter KF;

    public:

        MineKalmanFilter();


        std::vector<cv::Point> getPredictPoints(std::vector<cv::Point> newPoints);

    };

}

#endif //FACE_SWAP_KALMAN_FILTER_H


