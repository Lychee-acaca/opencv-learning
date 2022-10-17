#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//读取，转HSV
	Mat hsvMat;
	Mat input = imread("../../assets/WangSiTu.jpg");
	float scale = 1.0;
	resize(input, input, Size(input.cols * scale, input.rows * scale));
	cvtColor(input, hsvMat, COLOR_BGR2HSV);


	//定义皮肤HSV范围，求图像中的皮肤区域
	double i_minH = 0, i_maxH = 20;
	double i_minS = 43, i_maxS = 255;
	double i_minV = 55, i_maxV = 255;

	Mat maskMat;
	inRange(hsvMat, Scalar(i_minH, i_minS, i_minV), Scalar(i_maxH, i_maxS, i_maxV), maskMat);

	//滤波
	Mat filterOut;

	//双边滤波
	int bilateralFilterVal = 30;
	bilateralFilter(input, filterOut, bilateralFilterVal, bilateralFilterVal * 2, bilateralFilterVal / 2);
	
	//高斯滤波
	//GaussianBlur(input, filterOut, Size(5, 5), 3, 0);

	//中值滤波
	//medianBlur(input, filterOut, 7);

	//均值滤波
	//blur(input, filterOut, Size(3, 3));

	//把滤波图中皮肤区域的部分拷贝到结果图中
	Mat objectMat;
	input.copyTo(objectMat);
	filterOut.copyTo(objectMat, maskMat);

	//显示结果
	imshow("磨皮前", input);
	imshow("磨皮后", objectMat);
	waitKey(0);
}