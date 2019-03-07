#include <iostream>
#include "LibraryInterface.h"

using namespace std;

int main()
{
    cv::Mat* basePicture = cv::imgcodecs_imread("H:\\\Downloads\\44760113_491303904707328_3975569911538870027_n.jpg", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat* newf = cv::imgcodecs_imread("H:\\\Downloads\\44760113_491303904707328_3975569911538870027_n.jpg", cv::ImreadModes::IMREAD_COLOR);
    cv::Mat* currentPicture = cv::core_Mat_new1();
    cv::Mat* pictureDifference = cv::core_Mat_new1();
    cv::Mat* pictureThreshold = cv::core_Mat_new1();
    cv::Mat* pictureDilated = cv::core_Mat_new1();
    cv::Mat* hierarchy = cv::core_Mat_new1();

    std::vector<int> i = std::vector<int>();
    std::vector<int> *pi = &i;
    std::vector<cv::Mat> **contours = (std::vector<cv::Mat>**)&pi;

    cv::imgproc_cvtColor(&(cv::_InputArray)*newf, &(cv::_OutputArray)*currentPicture, 7, 0);
    cv::core_absdiff(&(cv::_InputArray)*basePicture, &(cv::_InputArray)*currentPicture, &(cv::_OutputArray)*pictureDifference);
    cv::imgproc_threshold(&(cv::_InputArray)*pictureDifference, &(cv::_OutputArray)*pictureThreshold, 25, 255, cv::ThresholdTypes::THRESH_BINARY);
    cv::imgproc_dilate(&(cv::_InputArray)*pictureThreshold, &(cv::_OutputArray)*pictureDilated, &(cv::_InputOutputArray)*cv::core_Mat_new1(), CvPoint(-1, -1), 2, 0, CvScalar());
    cv::imgproc_findContours1_OutputArray(&(cv::_InputOutputArray)*pictureDilated, contours, &(cv::_OutputArray)*hierarchy, 0, 2, CvPoint());
    cv::imgproc_moments(&(cv::_InputArray)*basePicture, 0);
    return 0;
}
