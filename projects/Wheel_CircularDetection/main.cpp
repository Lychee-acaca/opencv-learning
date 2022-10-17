#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//��ȡ��ת�Ҷȣ���򷨶�ֵ������ɫ
	Mat input, input_gray, thresh_Mat;
	input = imread("../../assets/wheel.png");
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	threshold(input_gray, thresh_Mat, 100, 255, THRESH_OTSU);
	bitwise_not(thresh_Mat, thresh_Mat);

	//�����㣬��ֵ�˲�
	Mat blur_Mat;
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	morphologyEx(thresh_Mat, blur_Mat, MORPH_OPEN, element);
	medianBlur(blur_Mat, blur_Mat, 9);


	//Ѱ������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(blur_Mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//ɸѡ��ȷԲ����ͼ
	int num = 0;
	Mat output;
	input.copyTo(output);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]); //�����������
		double len = arcLength(contours[i], true); //���������ܳ�
		double roundness = (4 * CV_PI * area) / (len * len); //Բ�ζ�

		if (roundness > 0.8)
		{
			num++;
			Point2f center;
			float radius;
			minEnclosingCircle(contours[i], center, radius);
			circle(output, center, (int)radius, Scalar(0, 255, 0), -1);
		}
	}

	//��ʾ
	printf("num = %d", num);
	imshow("input", input);
	imshow("thresh_Mat", thresh_Mat);
	imshow("blur_Mat", blur_Mat);
	imshow("output", output);
	waitKey(0);
}
