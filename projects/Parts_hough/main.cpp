#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//��ȡ��ת�Ҷȣ�canny��Ե���
	Mat input, input_gray, canny;
	input = imread("../../assets/parts.JPG");
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	Canny(input_gray, canny, 150, 250);

	//����任����һ�ַ�ʽ��ͶƱ��ֵ110����¼������
	Mat output1;
	input.copyTo(output1);
	vector<Vec2f> lines1;
	HoughLines(canny, lines1, 1, CV_PI / 180, 110);

	vector<Vec2f>::iterator it1 = lines1.begin();
	for (; it1 != lines1.end(); ++it1)
	{
		float rho = (*it1)[0], theta = (*it1)[1];
		Point pt1, pt2;
		double a = cos(theta);
		double b = sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;
		// 1000���ӳ���
		pt1.x = saturate_cast<int>(x0 + 1000 * (-b));
		pt1.y = saturate_cast<int>(y0 + 1000 * (a));
		pt2.x = saturate_cast<int>(x0 - 1000 * (-b));
		pt2.y = saturate_cast<int>(y0 - 1000 * (a));
		line(output1, pt1, pt2, Scalar(0, 0, 255), 1);
	}

	//����任���ڶ��ַ�ʽ��ͶƱ��ֵ20����С�߳�30����С��϶20����¼�ѿ�������
	Mat output2;
	input.copyTo(output2);
	vector<Vec4f> lines2;
	HoughLinesP(canny, lines2, 1, CV_PI / 180, 20, 30, 20);
	vector<Vec4f>::iterator it2 = lines2.begin();
	for (; it2 != lines2.end(); ++it2)
	{
		Point pt1, pt2;
		pt1.x = (*it2)[0];
		pt1.y = (*it2)[1];
		pt2.x = (*it2)[2];
		pt2.y = (*it2)[3];
		line(output2, pt1, pt2, Scalar(0, 0, 255), 1);
	}
	
	//��ʾ
	imshow("input", input);
	imshow("canny", canny);
	imshow("output1", output1);
	imshow("output2", output2);
	waitKey(0);
}