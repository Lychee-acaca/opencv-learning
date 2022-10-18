#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input_gray;
	Mat input = imread("../../assets/WangSiTu.jpg");
	cvtColor(input, input_gray, COLOR_BGR2GRAY);


	Mat objectMatX, objectMatY, sobel;
	//x方向求梯度，算子大小为3*3
	Sobel(input_gray, objectMatX, CV_16S, 1, 0, 3);
	//这种用法是给矩阵中的每个元素求绝对值
	convertScaleAbs(objectMatX, objectMatX);
	//y方向求梯度，算子大小为3*3
	Sobel(input_gray, objectMatY, CV_16S, 0, 1, 3);
	convertScaleAbs(objectMatY, objectMatY);
	//sobel算子原本需要采用sqrt(x^2+y^2)，为了运算速度可以近似为abs(x)+abs(y)
	sobel = objectMatX + objectMatY;

	//canny算子阈值1为80，阈值2为180
	//如果卷积结果高于阈值2则判定为边缘点
	//如果卷积结果低于阈值2，高于阈值1，与边缘点相连则判定为边缘点
	//如果卷积结果低于阈值1，则判定为非边缘点
	Mat canny;
	Canny(input_gray, canny, 80, 180); //80 180

	imshow("input", input);
	imshow("input_gray", input_gray);
	imshow("objectMatX", objectMatX);
	imshow("objectMatY", objectMatY);
	imshow("sobel", sobel);
	imshow("canny", canny);
	waitKey(0);
}