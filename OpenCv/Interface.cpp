#include "LibraryInterface.h"

static cv::_InputArray entity(cv::_InputArray *obj)
{
    return (obj != NULL) ? *obj : static_cast<cv::_InputArray>(cv::noArray());
}

OPENCV_API(cv::Mat*) core_Mat_new1()
{
    return new cv::Mat();
}

OPENCV_API(cv::Mat*) imgcodecs_imread(const char *filename, int flags)
{
    cv::Mat ret = cv::imread(filename, flags);
    return new cv::Mat(ret);
}

OPENCV_API(void) core_Mat_delete(cv::Mat *self)
{
    delete self;
}

OPENCV_API(cv::Mat*) core_Mat_new8(int rows, int cols, int type, void* data, size_t step)
{
    return new cv::Mat(rows, cols, type, data, step);
}

OPENCV_API(cv::_InputArray*) core_InputArray_new_byMat(cv::Mat *mat)
{
    return new cv::_InputArray(*mat);
}

OPENCV_API(void) core_InputArray_delete(cv::_InputArray *ia)
{
    delete ia;
}

OPENCV_API(cv::_OutputArray*) core_OutputArray_new_byMat(cv::Mat *mat)
{
    cv::_OutputArray ia(*mat);
    return new cv::_OutputArray(ia);
}

OPENCV_API(void) core_OutputArray_delete(cv::_OutputArray *oa)
{
    delete oa;
}

OPENCV_API(size_t) vector_Mat_getSize(std::vector<cv::Mat>* vector)
{
    return vector->size();
}

OPENCV_API(void) vector_Mat_assignToArray(std::vector<cv::Mat>* vector, cv::Mat** arr)
{
    for (size_t i = 0; i < vector->size(); i++)
    {
        (vector->at(i)).assignTo(*(arr[i]));
    }
}

OPENCV_API(void) vector_Mat_delete(std::vector<cv::Mat>* vector)
{
    delete vector;
}

OPENCV_API(double) imgproc_contourArea_InputArray(cv::_InputArray *contour, int oriented)
{
    return cv::contourArea(*contour, oriented != 0);
}

OPENCV_API(void) imgproc_findContours1_OutputArray(cv::_InputOutputArray *image, std::vector<cv::Mat> **contours, cv::_OutputArray *hierarchy, int mode, int method, CvPoint offset)
{
    *contours = new std::vector<cv::Mat>;
    cv::findContours(*image, **contours, *hierarchy, mode, method, offset);
}

OPENCV_API(void) imgproc_dilate(cv::_InputArray *src, cv::_OutputArray *dst, cv::_InputArray *kernel, CvPoint anchor, int iterations, int borderType, CvScalar borderValue)
{
    cv::dilate(*src, *dst, entity(kernel), anchor, iterations, borderType, borderValue);
}

OPENCV_API(double) imgproc_threshold(cv::_InputArray *src, cv::_OutputArray *dst, double thresh, double maxval, int type)
{
    return cv::threshold(*src, *dst, thresh, maxval, type);
}

OPENCV_API(void) imgproc_cvtColor(cv::_InputArray *src, cv::_OutputArray *dst, int code, int dstCn)
{
    cv::cvtColor(*src, *dst, code, dstCn);
}

OPENCV_API(void) core_absdiff(cv::_InputArray *src1, cv::_InputArray *src2, cv::_OutputArray *dst)
{
    cv::absdiff(*src1, *src2, *dst);
}

OPENCV_API(int) core_inputarray_kind(cv::_InputArray *src1)
{
    return src1->kind();
}

OPENCV_API(bool) core_mat_data(cv::Mat *src1)
{
    return src1->data == 0;
}

OPENCV_API(bool) core_mat_total(cv::Mat *src1)
{
    return src1->total() == 0;
}

OPENCV_API(bool) core_mat_dims(cv::Mat *src1)
{
    return src1->dims == 0;
}

OPENCV_API(cv::CvMoments) imgproc_moments(cv::_InputArray *arr, int binaryImage)
{
    cv::Moments m = cv::moments(*arr, binaryImage != 0);
    return c(m);
}