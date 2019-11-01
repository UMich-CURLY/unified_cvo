#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace cv;
using namespace std;

#define snr 700
#define LEN 3
#define THETA 45

void help();
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);
const String keys =
"{help h usage ? |             | print this message             }"
"{image          |input.png    | input image name               }"
"{LEN            |125          | length of a motion             }"
"{THETA          |0            | angle of a motion in degrees   }"
"{SNR            |700          | signal to noise ratio          }"
;

