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

    struct CvMoments
    {
        double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03; /* spatial moments */
        double  mu20, mu11, mu02, mu30, mu21, mu12, mu03; /* central moments */
        double  inv_sqrt_m00; /* m00 != 0 ? 1/sqrt(m00) : 0 */
    };

    static CvMoments c(const Moments m)
    {
        CvMoments ret;
        ret.m00 = m.m00; ret.m10 = m.m10; ret.m01 = m.m01;
        ret.m20 = m.m20; ret.m11 = m.m11; ret.m02 = m.m02;
        ret.m30 = m.m30; ret.m21 = m.m21; ret.m12 = m.m12; ret.m03 = m.m03;
        ret.mu20 = m.mu20; ret.mu11 = m.mu11; ret.mu02 = m.mu02;
        ret.mu30 = m.mu30; ret.mu21 = m.mu21; ret.mu12 = m.mu12; ret.mu03 = m.mu03;
        const double am00 = std::abs(m.m00);
        ret.inv_sqrt_m00 = am00 > DBL_EPSILON ? 1. / std::sqrt(am00) : 0;

        return ret;
    }

    static inline Scalar morphologyDefaultBorderValue() { return Scalar::all(DBL_MAX); }
    Mat imread(const String& filename, int flags = IMREAD_COLOR);
    double contourArea(InputArray contour, bool oriented = false);
    void findContours(InputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset = Point());
    void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue());
    double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type);
    void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);
    Moments moments(InputArray array, bool binaryImage = false);

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
    OPENCV_API(cv::CvMoments) imgproc_moments(cv::_InputArray *arr, int binaryImage);

    OPENCV_API(int) core_inputarray_kind(cv::_InputArray *src1);
    OPENCV_API(bool) core_mat_data(cv::Mat *src1);
    OPENCV_API(bool) core_mat_total(cv::Mat *src1);
    OPENCV_API(bool) core_mat_dims(cv::Mat *src1);
}