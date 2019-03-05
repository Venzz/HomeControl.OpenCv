#include <opencv2/core/core_c.h>
#include <vector>

#ifndef OPENCV_API
#   if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#      define OPENCV_API(rettype) extern "C" __declspec(dllexport) rettype __cdecl 
#   else
#      define OPENCV_API(rettype) extern "C" __attribute__ ((visibility ("default"))) rettype
#   endif
#endif

namespace cv {
    enum ImreadModes {
        IMREAD_UNCHANGED = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
        IMREAD_GRAYSCALE = 0,  //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
        IMREAD_COLOR = 1,  //!< If set, always convert image to the 3 channel BGR color image.
        IMREAD_ANYDEPTH = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
        IMREAD_ANYCOLOR = 4,  //!< If set, the image is read in any possible color format.
        IMREAD_LOAD_GDAL = 8,  //!< If set, use the gdal driver for loading the image.
        IMREAD_REDUCED_GRAYSCALE_2 = 16, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
        IMREAD_REDUCED_COLOR_2 = 17, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
        IMREAD_REDUCED_GRAYSCALE_4 = 32, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
        IMREAD_REDUCED_COLOR_4 = 33, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
        IMREAD_REDUCED_GRAYSCALE_8 = 64, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
        IMREAD_REDUCED_COLOR_8 = 65, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
        IMREAD_IGNORE_ORIENTATION = 128 //!< If set, do not rotate the image according to EXIF's orientation flag.
    };
    enum ThresholdTypes {
        THRESH_BINARY = 0, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
        THRESH_BINARY_INV = 1, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f]
        THRESH_TRUNC = 2, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
        THRESH_TOZERO = 3, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
        THRESH_TOZERO_INV = 4, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
        THRESH_MASK = 7,
        THRESH_OTSU = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
        THRESH_TRIANGLE = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value
    };

    static inline Scalar morphologyDefaultBorderValue() { return Scalar::all(DBL_MAX); }
    Mat imread(const String& filename, int flags = IMREAD_COLOR);
    double contourArea(InputArray contour, bool oriented = false);
    void findContours(InputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset = Point());
    void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue());
    double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type);
    void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);

    OPENCV_API(cv::Mat*) core_Mat_new1();
    OPENCV_API(cv::Mat*) imgcodecs_imread(const char *filename, int flags);
    OPENCV_API(void) core_Mat_delete(cv::Mat *self);
    OPENCV_API(cv::Mat*) core_Mat_new8(int rows, int cols, int type, void* data, size_t step);
    OPENCV_API(cv::_InputArray*) core_InputArray_new_byMat(cv::Mat *mat);
    OPENCV_API(void) core_InputArray_delete(cv::_InputArray *ia);
    OPENCV_API(cv::_OutputArray*) core_OutputArray_new_byMat(cv::Mat *mat);
    OPENCV_API(void) core_OutputArray_delete(cv::_OutputArray *oa);
    OPENCV_API(size_t) vector_Mat_getSize(std::vector<cv::Mat>* vector);
    OPENCV_API(void) vector_Mat_assignToArray(std::vector<cv::Mat>* vector, cv::Mat** arr);
    OPENCV_API(void) vector_Mat_delete(std::vector<cv::Mat>* vector);
    OPENCV_API(double) imgproc_contourArea_InputArray(cv::_InputArray *contour, int oriented);
    OPENCV_API(void) imgproc_findContours1_OutputArray(cv::_InputOutputArray *image, std::vector<cv::Mat> **contours, cv::_OutputArray *hierarchy, int mode, int method, CvPoint offset);
    OPENCV_API(void) imgproc_dilate(cv::_InputArray *src, cv::_OutputArray *dst, cv::_InputArray *kernel, CvPoint anchor, int iterations, int borderType, CvScalar borderValue);
    OPENCV_API(double) imgproc_threshold(cv::_InputArray *src, cv::_OutputArray *dst, double thresh, double maxval, int type);
    OPENCV_API(void) imgproc_cvtColor(cv::_InputArray *src, cv::_OutputArray *dst, int code, int dstCn);
    OPENCV_API(void) core_absdiff(cv::_InputArray *src1, cv::_InputArray *src2, cv::_OutputArray *dst);

    OPENCV_API(int) core_inputarray_kind(cv::_InputArray *src1);
    OPENCV_API(bool) core_mat_data(cv::Mat *src1);
    OPENCV_API(bool) core_mat_total(cv::Mat *src1);
    OPENCV_API(bool) core_mat_dims(cv::Mat *src1);
}