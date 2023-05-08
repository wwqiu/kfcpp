/**
 *
 * This example compare the kfcpp kalman filter and the opencv kalman filter.
 *
 */

#include <opencv2/opencv.hpp>
#include "kfcpp.hpp"

void on_mouse(int e, int x, int y, int d, void* ptr)
{
    cv::Point* p = (cv::Point*)ptr;
    p->x = x;
    p->y = y;
}

kfcpp::Matrix2d toMatrix2d(const cv::Mat& src) {
    kfcpp::Matrix2d dst = kfcpp::Matrix2d(src.rows, src.cols);
    memcpy(dst.data, src.data, sizeof(float) * src.rows * src.cols);
    return dst;
}

cv::Mat toCvMat(const kfcpp::Matrix2d& src) {
    cv::Mat dst(src.rows, src.cols, CV_32FC1);
    memcpy(dst.data, src.data, sizeof(float) * src.rows * src.cols);
    return dst;
}

int main()
{
    int state_size = 4;    // [x, y, v_x, v_y]
    int meas_size = 2;     // [z_x, z_y] 
    int control_size = 0;  // no control input
    unsigned int F_type = CV_32F;//or CV_64F
    // opencv kalman filter
    cv::KalmanFilter KF(state_size, meas_size, control_size, F_type);
    cv::Mat state(state_size, 1, F_type);  // [x, y, v_x, v_y] 
    cv::Mat meas(meas_size, 1, F_type);    // [z_x, z_y] 
    cv::setIdentity(KF.transitionMatrix);
    cv::setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(3e-2));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));
    // kfcpp kalman filter
    kfcpp::KalmanFilter mkf;
    mkf.SetF(toMatrix2d(KF.transitionMatrix));
    mkf.SetP(toMatrix2d(KF.errorCovPost));
    mkf.SetQ(toMatrix2d(KF.processNoiseCov));
    mkf.SetH(toMatrix2d(KF.measurementMatrix));
    mkf.SetR(toMatrix2d(KF.measurementNoiseCov));
    kfcpp::Matrix2d init_state = kfcpp::Matrix2d::zeros(4, 1);
    mkf.Init(init_state);

    char ch = 0;
    cv::Mat display_image(600, 800, CV_8UC3);
    cv::namedWindow("Mouse Track");
    cv::Point mouse_pos;
    setMouseCallback("Mouse Track", on_mouse, &mouse_pos);

    while (ch != 'q' && ch != 'Q')
    {
        display_image = cv::Scalar::all(0);
        
        meas.at<float>(0) = mouse_pos.x;
        meas.at<float>(1) = mouse_pos.y;

        // opencv
        KF.predict();
        cv::Mat state_cv = KF.correct(meas);

        // kfcpp
        mkf.Predict();
        mkf.Measure(toMatrix2d(meas));
        cv::Mat state_kfcpp = toCvMat(mkf.GetState());

        // visualize
        cv::Point2f meas_point(meas.at<float>(0), meas.at<float>(1));
        cv::Point2f point0(state_cv.at<float>(0), state_cv.at<float>(1));
        cv::Point2f point1(state_kfcpp.at<float>(0), state_kfcpp.at<float>(1));
        char text[20];
        printf("opencv(%0.2f,%0.2f), kfcpp(%0.2f,%0.2f)\n", point0.x, point0.y, point1.x, point1.y);

        cv::circle(display_image, point0, 10, CV_RGB(0, 255, 0), 1);
        cv::circle(display_image, meas_point, 5, CV_RGB(0, 255, 255), 1);
        cv::circle(display_image, point1, 5, CV_RGB(255, 255, 0), 1);
        cv::imshow("Mouse Track", display_image);
        ch = cv::waitKey(40);
    }
}