//应该只能用于矫正边缘没有黑色像素的图片
//而且有些太奇怪的角度也不太行
#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/WangSiTu.jpg");

	//先把图片任意旋转一个角度，作为矫正算法的输入
	//为了防止丢失旋转后的边角信息，将图像进行缩放
	float angel = 35.5;
	float h = sqrt(2) * cos(CV_PI / 4 - angel * 2 * CV_PI / 360);
	float scale = 1 / h;

	Point2f center(input.cols * 0.5, input.rows * 0.5);
	Mat affine_matrix = getRotationMatrix2D(center, angel, scale);
	Mat input_Rotated;
	warpAffine(input, input_Rotated, affine_matrix, input.size());

	//图像矫正部分
	Mat maskMat;
	//获取图片的黑色区域，进行边缘提取
	inRange(input_Rotated, Scalar(0, 0, 0), Scalar(0, 0, 0), maskMat);
	imshow("step1", maskMat);
	Canny(maskMat, maskMat, 1, 10);
	imshow("step2", maskMat);

	//闭运算填充空洞区域
	//不进行闭运算，仅使用canny边缘检测，可能会丢失四个角
	morphologyEx(maskMat, maskMat, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)));
	imshow("step3", maskMat);

	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;

	//提取轮廓
	findContours(maskMat, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));

	//找最大的轮廓的编号
	int index = -1;
	int max = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		//获取轮廓面积
		double area = contourArea(contours[i]);
		if (area > max)
		{
			max = area;
			index = i;
		}
	}

	//最大轮廓求最小外接矩形
	RotatedRect rect = minAreaRect(contours[index]);
	
	//求矩形的四个点
	Point2f rectp[4];
	rect.points(rectp);

	//定义画布的四个顶点
	Point2f pts[] = {
		Point2f(0, 0),
		Point2f(input_Rotated.rows, 0),
		Point2f(input_Rotated.rows, input_Rotated.cols),
		Point2f(0, input_Rotated.cols),
	};

	//进行投影变换
	Mat output;
	Mat perspective_matrix = getPerspectiveTransform(rectp, pts);
	warpPerspective(input_Rotated, output, perspective_matrix, output.size());

	imshow("input", input);
	imshow("矫正前", input_Rotated);
	imshow("矫正后", output);
	waitKey(0);
}
