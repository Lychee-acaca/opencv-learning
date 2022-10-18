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
	//x�������ݶȣ����Ӵ�СΪ3*3
	Sobel(input_gray, objectMatX, CV_16S, 1, 0, 3);
	//�����÷��Ǹ������е�ÿ��Ԫ�������ֵ
	convertScaleAbs(objectMatX, objectMatX);
	//y�������ݶȣ����Ӵ�СΪ3*3
	Sobel(input_gray, objectMatY, CV_16S, 0, 1, 3);
	convertScaleAbs(objectMatY, objectMatY);
	//sobel����ԭ����Ҫ����sqrt(x^2+y^2)��Ϊ�������ٶȿ��Խ���Ϊabs(x)+abs(y)
	sobel = objectMatX + objectMatY;

	//canny������ֵ1Ϊ80����ֵ2Ϊ180
	//���������������ֵ2���ж�Ϊ��Ե��
	//���������������ֵ2��������ֵ1�����Ե���������ж�Ϊ��Ե��
	//���������������ֵ1�����ж�Ϊ�Ǳ�Ե��
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